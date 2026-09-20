#!/usr/bin/env python3
"""H3: multimodal baselines

  H3A  calibrated 50/50 fusion of H1 and H2 scores/maps,
  H3B  max fusion of H1 and H2 scores/maps,
  H3C  aligned concat bank (DINO 56x56-repeated RGB + Point-MAE XYZ, 1920-D).

RGB patches are up-sampled to the 56x56 XYZ grid with the repo's canonical
repeat layout (cache_utils.align_rgb_descriptors) so each concat row truly
aligns RGB and XYZ at the same spatial cell.  Output: results/H3_multimodal.csv
"""

import os
import time

import numpy as np
import torch

import cache_utils
from cache_utils import (align_rgb_descriptors, concat_descriptors, cfg_get,
                         load_blur, load_config,
                         load_split_samples, make_memory_index, normalize_and_bank,
                         score_samples, ensure_dir)
from metrics import compute_metrics, write_results_csv

MODEL = "H3_MULTIMODAL"
VARIANTS = ["H3A_Calibrated_50_50", "H3B_Max_Fusion", "H3C_Concat_Fusion"]
CAT_KEY = "category"


def _prepare_bank(cfg, train_desc, grid, tag, cache_root):
    bank_t, stats = normalize_and_bank(train_desc, cfg, tag, cache_root)
    bank_np = bank_t.detach().cpu().numpy().astype(np.float32)
    index, bank_np = make_memory_index(bank_np)
    return bank_np, index, stats


def _concat_test_samples(rgb_samples, xyz_samples):
    """Per-sample aligned concat descriptors on the 56x56 grid."""
    out = []
    for rgb_s, xyz_s in zip(rgb_samples, xyz_samples):
        rgb_aligned = align_rgb_descriptors(rgb_s["desc"])
        xyz_d = xyz_s["desc"]
        concat = torch.cat([rgb_aligned, xyz_d], dim=1)
        d = dict(xyz_s)
        d["desc"] = concat
        out.append(d)
    return out


def _rows_for(categories, results):
    rows = []
    for cat in categories:
        for v in VARIANTS:
            row = {"model": MODEL, "category": cat, "variant": v,
                   "I": results[cat][v]["I"], "P": results[cat][v]["P"],
                   "AUPRO": results[cat][v]["AUPRO"]}
            rows.append(row)
    return rows


def run_h3(cfg, categories=None, blur=None):
    categories = categories or cfg["categories"]
    results_dir = cfg["results_dir"]
    ensure_dir(results_dir)

    if blur is None:
        blur = load_blur()
    cfg_data = cfg
    cache_root = cfg["features_dir"]

    all_rows = []
    results = {}
    for category in categories:
        t0 = time.time()
        rgb_train = load_split_samples(cfg["features_dir"], category, "train", "rgb")
        xyz_train = load_split_samples(cfg["features_dir"], category, "train", "xyz")
        rgb_test = load_split_samples(cfg["features_dir"], category, "test", "rgb")
        xyz_test = load_split_samples(cfg["features_dir"], category, "test", "xyz")
        if not rgb_train or not xyz_train:
            print("[skip] %s: empty train shards" % category)
            continue

        # --- RGB bank (28x28) --------------------------------
        rgb_train_desc = concat_descriptors(rgb_train)
        rgb_bank_np, rgb_index, rgb_stats = _prepare_bank(
            cfg_data, rgb_train_desc, (28, 28), "%s_h3_rgb" % category, cache_root)
        # --- XYZ bank (56x56) --------------------------------
        xyz_train_desc = concat_descriptors(xyz_train)
        xyz_bank_np, xyz_index, xyz_stats = _prepare_bank(
            cfg_data, xyz_train_desc, (56, 56), "%s_h2_xyz" % category, cache_root)
        # --- Concat bank (56x56, 1920-D) ---------------------
        concat_train_desc = torch.cat([
            align_rgb_descriptors(rgb_train_desc), xyz_train_desc], dim=1)
        concat_bank_np, concat_index, concat_stats = _prepare_bank(
            cfg_data, concat_train_desc, (56, 56), "%s_h3_concat" % category, cache_root)

        # --- Score every variant ------------------------------
        s_rgb, map_rgb, _l1, _n1 = score_samples(
            rgb_test, rgb_bank_np, rgb_index, rgb_stats, (28, 28), blur, cfg_data)
        s_xyz, map_xyz, labels, names = score_samples(
            xyz_test, xyz_bank_np, xyz_index, xyz_stats, (56, 56), blur, cfg_data)
        concat_test = _concat_test_samples(rgb_test, xyz_test)
        s_con, map_con, _l3, _n3 = score_samples(
            concat_test, concat_bank_np, concat_index, concat_stats, (56, 56), blur, cfg_data)

        gts = [np.asarray(s["gt"], dtype=np.float32) for s in xyz_test]
        pix_labels = np.concatenate([g.ravel() for g in gts]) if gts else np.array([])
        lim = float(cfg_get(cfg, "aupro_integration_limit", 0.3))
        nthr = int(cfg_get(cfg, "aupro_num_thresholds", 100))

        m_a = compute_metrics(labels, 0.5 * np.array(s_rgb) + 0.5 * np.array(s_xyz),
                              pix_labels, np.concatenate([(0.5 * x + 0.5 * y).ravel()
                                                          for x, y in zip(map_rgb, map_xyz)]),
                              gts, [0.5 * x + 0.5 * y for x, y in zip(map_rgb, map_xyz)],
                              lim, nthr)
        m_b = compute_metrics(labels, np.maximum(np.array(s_rgb), np.array(s_xyz)),
                              pix_labels, np.concatenate([np.maximum(x, y).ravel()
                                                          for x, y in zip(map_rgb, map_xyz)]),
                              gts, [np.maximum(x, y) for x, y in zip(map_rgb, map_xyz)],
                              lim, nthr)
        m_c = compute_metrics(labels, np.array(s_con),
                              pix_labels, np.concatenate([x.ravel() for x in map_con]),
                              gts, map_con, lim, nthr)

        results[category] = {"H3A_Calibrated_50_50": m_a,
                             "H3B_Max_Fusion": m_b, "H3C_Concat_Fusion": m_c}
        print("[H3] %-12s A:%.4f B:%.4f C:%.4f (I) (%.1fs)"
              % (category, m_a["I"] or float("nan"), m_b["I"] or float("nan"),
                 m_c["I"] or float("nan"), time.time() - t0), flush=True)

    if not results:
        print("[H3] no results produced")
        return all_rows

    rows = _rows_for([c for c in categories if c in results], results)
    # Per-variant MEAN rows.
    for v in VARIANTS:
        sub = [r for r in rows if r["variant"] == v]
        rows.append({"model": MODEL, "category": "MEAN", "variant": v,
                     "I": float(np.mean([r["I"] for r in sub if r["I"] is not None]) or 0),
                     "P": float(np.mean([r["P"] for r in sub if r["P"] is not None]) or 0),
                     "AUPRO": float(np.mean([r["AUPRO"] for r in sub if r["AUPRO"] is not None]) or 0)})

    path = os.path.join(results_dir, "H3_multimodal.csv")
    write_results_csv(path, rows, provenance="measured_by_us", model=MODEL,
                      extra_fields=["variant"])
    for v in VARIANTS:
        r = [x for x in rows if x["variant"] == v and x["category"] == "MEAN"][0]
        print("\n---> H3 %-22s MEAN  I=%.4f P=%.4f AUPRO=%.4f"
              % (v, r["I"], r["P"], r["AUPRO"]))
    print("    -> %s" % path)
    return rows


if __name__ == "__main__":
    cfg = load_config()
    run_h3(cfg)