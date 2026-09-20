#!/usr/bin/env python3
"""Per-category end-to-end XMV-AD-H run for Kaggle.

Kaggle notebooks have only ~20GB on /kaggle/working, which is smaller than the
full fp16 feature-shard set (~37GB), so this driver processes ONE category at a
time, reading the dataset tree directly from the read-only /kaggle/input mount:

  extract DINO/Point-MAE shards for the category
  -> run H1 / H2 / H3 for the category
  -> delete the category's shards
  -> next category

Results accumulate across categories.  H1_RGB.csv / H2_XYZ.csv /
H3_multimodal.csv / M3DM_comparison.csv are written at the end exactly as
run_all.py writes them, using the same cache_utils / metrics / runners, so the
resulting numbers are comparable to a Colab run of the bundle.

Inputs (set by the Kaggle notebook before running this):
  XMVAD_BUNDLE_ROOT     -> /kaggle/working/bundle
  XMVAD_KAGGLE_DATASET  -> root of the extracted MVTec-3D-AD tree under /kaggle/input
"""

import gc
import os
import time

import torch

import cache_utils
from cache_utils import (CacheManifest, ensure_dirs,
                         load_blur, load_config, torch_device)
import bootstrap
import extract_features
import run_all
import run_h1
import run_h2
import run_h3
from metrics import mean_row, write_results_csv


def _banner(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66, flush=True)


def np_mean(vals):
    import numpy as np
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _pick_zip():
    zp = os.environ.get("XMVAD_KAGGLE_ZIP", "").strip()
    if zp and os.path.isfile(zp):
        return zp
    return None


def _locate_dataset(cfg):
    """Prefers the Kaggle input mount (XMVAD_KAGGLE_DATASET / known layout)."""
    env = os.environ.get("XMVAD_KAGGLE_DATASET", "").strip()
    if env and os.path.isdir(os.path.join(env, "bagel")):
        return env
    import glob
    cands = sorted(glob.glob("/kaggle/input/**/mvtec_3d_anomaly_detection", recursive=True))
    if not cands:
        for p in sorted(glob.glob("/kaggle/input/datasets/*/*")):
            if os.path.isdir(os.path.join(p, "bagel")):
                cands.append(p)
    if not cands:
        return cfg["dataset_dir"]
    return cands[0]


def _clear_manifest_category(cfg, category):
    m = CacheManifest(cfg["features_dir"])
    feats = m.data.setdefault("features", {})
    if category in feats:
        del feats[category]
        m.save()
        print("[manifest] reset category %s (will re-extract)" % category)


def _delete_category(cfg, category):
    # The dataset tree lives on the read-only /kaggle/input mount on Kaggle;
    # drop only this category's feature shards to keep /kaggle/working small.
    for f in os.listdir(cfg["features_dir"]):
        if f.startswith(category + "_") and f.endswith(".pt") and not f.endswith(".tmp"):
            try:
                os.remove(os.path.join(cfg["features_dir"], f))
            except OSError:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _final_h1_csv(cfg, rows):
    rows = list(rows)
    rows.append(mean_row(rows))
    path = os.path.join(cfg["results_dir"], "H1_RGB.csv")
    write_results_csv(path, rows, provenance="measured_by_us", model="H1_RGB")
    m = rows[-1]
    print("\n---> H1 MEAN  I=%.4f P=%.4f AUPRO=%s  -> %s"
          % (m["I"] or float("nan"), m["P"] or float("nan"),
             ("%.4f" % m["AUPRO"]) if m["AUPRO"] is not None else "  n/a", path))


def _final_h2_csv(cfg, rows):
    rows = list(rows)
    rows.append(mean_row(rows))
    path = os.path.join(cfg["results_dir"], "H2_XYZ.csv")
    write_results_csv(path, rows, provenance="measured_by_us", model="H2_XYZ")
    m = rows[-1]
    print("\n---> H2 MEAN  I=%.4f P=%.4f AUPRO=%s  -> %s"
          % (m["I"] or float("nan"), m["P"] or float("nan"),
             ("%.4f" % m["AUPRO"]) if m["AUPRO"] is not None else "  n/a", path))


def _final_h3_csv(cfg, cat_rows):
    import os
    rows = list(cat_rows)
    for v in run_h3.VARIANTS:
        sub = [r for r in rows if r["variant"] == v]
        rows.append({"model": "H3_MULTIMODAL", "category": "MEAN", "variant": v,
                     "I": np_mean([r["I"] for r in sub]),
                     "P": np_mean([r["P"] for r in sub]),
                     "AUPRO": np_mean([r["AUPRO"] for r in sub])})
    path = os.path.join(cfg["results_dir"], "H3_multimodal.csv")
    write_results_csv(path, rows, provenance="measured_by_us", model="H3_MULTIMODAL",
                      extra_fields=["variant"])
    for v in run_h3.VARIANTS:
        r = [x for x in rows if x["variant"] == v and x["category"] == "MEAN"][0]
        print("\n---> H3 %-22s MEAN  I=%.4f P=%.4f AUPRO=%.4f"
              % (v, r["I"], r["P"], r["AUPRO"]))
    print("    -> %s" % path)


def run_kaggle(cfg=None, categories=None, zip_path=None):
    import numpy as np
    cfg = cfg or load_config()
    categories = categories or cfg["categories"]
    dataset_dir = _locate_dataset(cfg)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError("dataset dir missing: %s" % dataset_dir)
    cfg["dataset_dir"] = dataset_dir

    device = torch_device()
    print("=" * 66)
    print("XMV-AD-H run_kaggle   gpu=%s  colab=%s" % (device, cache_utils.is_colab()))
    print("dataset              : %s" % dataset_dir)
    print("=" * 66, flush=True)

    ensure_dirs([cfg["features_dir"], cfg["results_dir"],
                 cfg["logs_dir"], cfg["checkpoints_dir"]])
    bootstrap.ensure_pointmae_checkpoint(cfg)

    blur = load_blur()
    h1_rows, h2_rows, h3_rows = [], [], []
    t_all = time.time()

    for category in categories:
        t_cat = time.time()
        _banner("CATEGORY %s" % category)

        _clear_manifest_category(cfg, category)

        extract_features.run_extraction(cfg, [category])
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        r1 = run_h1.run_h1(cfg, [category], blur)
        if r1:
            h1_rows.append(r1[0])
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        r2 = run_h2.run_h2(cfg, [category], blur)
        if r2:
            h2_rows.append(r2[0])
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        r3 = run_h3.run_h3(cfg, [category], blur)
        for row in (r3 or []):
            if row.get("category") == category:
                h3_rows.append(row)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _delete_category(cfg, category)
        print("[category] %s finished in %.0fs" % (category, time.time() - t_cat),
              flush=True)

        # Update CSVs incrementally after each completed category
        if h1_rows:
            _final_h1_csv(cfg, h1_rows)
        if h2_rows:
            _final_h2_csv(cfg, h2_rows)
        if h3_rows:
            _final_h3_csv(cfg, h3_rows)
        try:
            path, comp_rows = run_all._build_comparison_csv(cfg)
            run_all._print_comparison_table(cfg, comp_rows)
        except Exception as e:
            pass

    # Comparison vs published M3DM.
    print("\n")
    path, rows = run_all._build_comparison_csv(cfg)
    run_all._print_comparison_table(cfg, rows)

    print("\n===== DONE in %.0fs =====" % (time.time() - t_all))
    print("results -> %s" % cfg["results_dir"])
    return cfg


if __name__ == "__main__":
    try:
        run_kaggle()
    except KeyboardInterrupt:
        print("\n[interrupted]", file=__import__("sys").stderr)
        __import__("sys").exit(130)