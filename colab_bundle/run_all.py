#!/usr/bin/env python3
"""End-to-end XMV-AD-H run on Colab (and locally).

  1. bootstrap  (dataset locate/extract, checkpoint, Drive cache restore)
  2. extract    (DINO RGB + Point-MAE XYZ features -> shards + manifest)
  3. H1         (RGB memory bank)            -> results/H1_RGB.csv
  4. H2         (XYZ memory bank)            -> results/H2_XYZ.csv
  5. H3         (H3A 50/50, H3B max, H3C concat) -> results/H3_multimodal.csv
  6. compare    (best-of-ours vs published M3DM)  -> results/M3DM_comparison.csv

Published M3DM values are reported by prior work (reported_prior_work); all
ours are measured_by_us.  Logged run is captured by the notebook cell:
    python -u run_all.py 2>&1 | tee drive/logs/run_all.log
"""

import csv
import os
import shutil
import sys
import time

import numpy as np

import cache_utils
from cache_utils import cfg_get, ensure_dir, load_config, torch_device
import bootstrap
import extract_features
import status as status_mod
import run_h1
import run_h2
import run_h3


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _read_rows(cfg, name):
    path = os.path.join(cfg["results_dir"], name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _best_per_category(cfg):
    """Per-category best of H1, H2, H3C (I first, then P, then AUPRO)."""
    h1 = _read_rows(cfg, "H1_RGB.csv") or []
    h2 = _read_rows(cfg, "H2_XYZ.csv") or []
    h3 = _read_rows(cfg, "H3_multimodal.csv") or []
    cand = {}
    for r in h1:
        if r.get("category") and r["category"] != "MEAN":
            cand.setdefault(r["category"], []).append((r, "H1_RGB"))
    for r in h2:
        if r.get("category") and r["category"] != "MEAN":
            cand.setdefault(r["category"], []).append((r, "H2_XYZ"))
    for r in h3:
        if r.get("category") and r["category"] != "MEAN" and r.get("variant") == "H3C_Concat_Fusion":
            cand.setdefault(r["category"], []).append((r, "H3C_Concat_Fusion"))
    best = {}
    for cat, options in cand.items():
        options.sort(key=lambda x: (-(_safe_float(x[0].get("I")) or -1),
                                    -(_safe_float(x[0].get("P")) or -1),
                                    -(_safe_float(x[0].get("AUPRO")) or -1)))
        score, variant = options[0]
        best[cat] = {"variant": variant, "row": score}
    return best


def _build_comparison_csv(cfg):
    m3dm = cfg.get("m3dm_published", {})
    best = _best_per_category(cfg)
    out_dir = cfg["results_dir"]
    ensure_dir(out_dir)

    rows = []
    cats = [c for c in cfg["categories"] if c in best]
    for cat in cats:
        r = best[cat]["row"]
        for metric, label in (("I", "I-AUROC"), ("P", "P-AUROC"), ("AUPRO", "AUPRO@0.3")):
            ours = _safe_float(r.get(metric))
            pub = _safe_float(m3dm.get(metric))
            gap = (ours - pub) if (ours is not None and pub is not None) else None
            rows.append({
                "category": cat, "metric": label, "variant": best[cat]["variant"],
                "ours": "%.4f" % ours if ours is not None else "n/a",
                "m3dm_reported_prior_work": "%.4f" % pub if pub is not None else "n/a",
                "gap": "+%.4f" % gap if gap is not None else "n/a",
                "note": "" ,
            })
    rows[0]["note"] = "M3DM=REPORTED BY PRIOR WORK; ours=measured_by_us"

    # MEAN row from our best values this run.
    if cats:
        for metric, label in (("I", "I-AUROC"), ("P", "P-AUROC"), ("AUPRO", "AUPRO@0.3")):
            vals = [_safe_float(best[c]["row"].get(metric)) for c in cats]
            vals = [v for v in vals if v is not None]
            pub = _safe_float(m3dm.get(metric))
            if vals:
                ours = float(np.mean(vals))
                gap = (ours - pub) if pub is not None else None
                rows.append({
                    "category": "MEAN", "metric": label, "variant": "(best of H1/H2/H3C)",
                    "ours": "%.4f" % ours,
                    "m3dm_reported_prior_work": "%.4f" % pub if pub is not None else "n/a",
                    "gap": "+%.4f" % gap if gap is not None else "n/a",
                    "note": "",
                })

    path = os.path.join(out_dir, "M3DM_comparison.csv")
    fields = ["category", "metric", "variant", "ours",
              "m3dm_reported_prior_work", "gap", "note"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path, rows


def _print_comparison_table(cfg, rows):
    m3dm = cfg.get("m3dm_published", {})
    best = _best_per_category(cfg)
    print("\n" + "=" * 66)
    print("FINAL: best-of-ours per category  vs  published M3DM")
    print("=" * 66)
    head = "%-13s %-18s %-8s %-8s %-8s" % ("category", "best_method", "I", "P", "AUPRO")
    print(head)
    print("-" * len(head))
    for cat in cfg["categories"]:
        b = best.get(cat)
        if not b:
            print("%-13s %-18s %-8s %-8s %-8s" % (cat, "no results", "-", "-", "-"))
            continue
        r = b["row"]
        print("%-13s %-18s %-8s %-8s %-8s" % (
            cat, b["variant"],
            "%.4f" % _safe_float(r.get("I")), "%.4f" % _safe_float(r.get("P")),
            "%.4f" % _safe_float(r.get("AUPRO"))))
    print("-" * len(head))

    ours_means = {}
    for metric, label in (("I", "I-AUROC"), ("P", "P-AUROC"), ("AUPRO", "AUPRO@0.3")):
        vals = [_safe_float(best[c]["row"].get(metric)) for c in best]
        vals = [v for v in vals if v is not None]
        ours_means[label] = float(np.mean(vals)) if vals else None

    print("\n  %-20s %-11s %-11s %-11s" % ("", "I-AUROC", "P-AUROC", "AUPRO@0.3"))
    print("  %-20s %-11s %-11s %-11s" % (
        "best-of-ours (measured)", _fmt(ours_means.get("I-AUROC")),
        _fmt(ours_means.get("P-AUROC")), _fmt(ours_means.get("AUPRO@0.3"))))
    print("  %-20s %-11s %-11s %-11s" % (
        "M3DM (reported by prior work)", _fmt(_safe_float(m3dm.get("I"))),
        _fmt(_safe_float(m3dm.get("P"))), _fmt(_safe_float(m3dm.get("AUPRO")))))
    print("\nsummary CSV -> %s" % os.path.join(cfg["results_dir"], "M3DM_comparison.csv"))


def _fmt(v):
    return "  n/a " if v is None else "%.4f" % v


def _mirror_to_drive(cfg, key):
    """Copy results/ or logs/ into cfg.persistent[key] (Drive on Colab)."""
    src = cfg.get(key + "_dir")
    dst = cfg["persistent"].get(key)
    if not src or not dst or not os.path.isdir(src):
        return
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    ensure_dir(dst)
    n = 0
    for f in sorted(os.listdir(src)):
        s = os.path.join(src, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst, f))
            n += 1
    if n:
        print("[mirror] %d files %s -> %s" % (n, src, dst))


def run_all(cfg=None, categories=None, skip_setup=False):
    cfg = cfg or load_config()
    categories = categories or cfg["categories"]
    t_start = time.time()

    print("=" * 66)
    print("XMV-AD-H  run_all   gpu=%s  colab=%s" % (torch_device(), cache_utils.is_colab()))
    print("=" * 66)

    if not skip_setup:
        bootstrap.setup(cfg, categories)

    extract_features.run_extraction(cfg, categories)

    run_h1.run_h1(cfg, categories)
    run_h2.run_h2(cfg, categories)
    run_h3.run_h3(cfg, categories)

    path, rows = _build_comparison_csv(cfg)
    _print_comparison_table(cfg, rows)

    _mirror_to_drive(cfg, "results")
    _mirror_to_drive(cfg, "logs")

    # Final status view.
    print("\n")
    status_mod.run_status(cfg, categories)

    print("\n===== DONE in %.0fs =====" % (time.time() - t_start))
    print("results -> %s" % cfg["results_dir"])
    print("log     -> %s" % cfg["logs_dir"])
    return cfg


if __name__ == "__main__":
    try:
        run_all()
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        sys.exit(130)