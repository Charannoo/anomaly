#!/usr/bin/env python3
"""Status overview for XMV-AD-H: GPU, extraction checkmarks, per-method
results, published M3DM comparator and gaps vs our best.  Used as the final
notebook cell so the run summary is visible without opening CSVs.
"""

import csv
import os

import cache_utils
from cache_utils import (CacheManifest, cfg_get, is_colab, load_config,
                         local_time, torch_device)


def _fmt(v):
    return "  n/a " if v is None else "%.4f" % float(v)


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _best_variant_rows(h3_rows):
    """Pick per-category best H3 variant by I, then P, then AUPRO."""
    best = {}
    for r in h3_rows or []:
        if r.get("category") == "MEAN":
            continue
        key = r["category"]
        cand = best.get(key)
        if cand is None:
            best[key] = r
            continue
        for metric in ("I", "P", "AUPRO"):
            ca, cb = _safe_float(cand.get(metric)), _safe_float(r.get(metric))
            if cb is not None and (ca is None or cb > ca):
                best[key] = r
                break
            if ca is not None and (cb is None or ca > cb):
                break
    return list(best.values())


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def run_status(cfg=None, categories=None):
    cfg = cfg or load_config()
    categories = categories or cfg["categories"]
    gpu_name = "?"
    if torch_device() == "cuda":
        import torch
        gpu_name = torch.cuda.get_device_name(0)
    print("=" * 62)
    print("XMV-AD-H STATUS   colab=%s  gpu=%s  (%s)"
          % (is_colab(), gpu_name, local_time()))
    print("=" * 62)

    # ---- Extraction checkmarks ----------------------------------------
    manifest = CacheManifest(cfg["features_dir"])
    feats = cfg["features_dir"]
    print("\n[features] %s" % feats)
    for cat in categories:
        marks = []
        for split in ("train", "test"):
            for mod in ("rgb", "xyz"):
                ok = manifest.is_complete(cat, split, mod)
                marks.append("%s_%s=%s" % (split, mod,
                                           ["--", "OK"][int(bool(ok))]))
        print("   %-13s %s" % (cat, "  ".join(marks)))

    # ---- Results -------------------------------------------------------
    m3dm = cfg.get("m3dm_published", {})
    h1 = read_csv(os.path.join(cfg["results_dir"], "H1_RGB.csv"))
    h2 = read_csv(os.path.join(cfg["results_dir"], "H2_XYZ.csv"))
    h3 = read_csv(os.path.join(cfg["results_dir"], "H3_multimodal.csv"))

    best_by_cat = _best_variant_rows(h3)
    grid = "%-13s %-16s %-8s %-8s %-8s %-12s" % (
        "category", "method", "I", "P", "AUPRO", "provenance")
    print("\n[results] summary (best H3 variant per category + M3DM comparator)")
    print(grid)
    print("-" * len(grid))
    for cat in categories:
        b = next((x for x in best_by_cat if x["category"] == cat), None)
        if b:
            print("%-13s %-16s %-8s %-8s %-8s %-12s" % (
                cat, b.get("variant", "?"), _fmt(_safe_float(b.get("I"))),
                _fmt(_safe_float(b.get("P"))), _fmt(_safe_float(b.get("AUPRO"))),
                "measured_by_us"))
        print("%-13s %-16s %-8s %-8s %-8s %-12s" % (
            cat, "M3DM", _fmt(_safe_float(m3dm.get("I"))),
            _fmt(_safe_float(m3dm.get("P"))), _fmt(_safe_float(m3dm.get("AUPRO"))),
            m3dm.get("provenance", "reported_prior_work")))

    # ---- CSV files -----------------------------------------------------
    print("\n[CSV outputs]")
    for name in ("H1_RGB.csv", "H2_XYZ.csv", "H3_multimodal.csv", "M3DM_comparison.csv"):
        p = os.path.join(cfg["results_dir"], name)
        print("   %-22s %s" % (name, p if os.path.exists(p) else "MISSING"))

    # ---- Gaps vs published M3DM ----------------------------------------
    print("\n[vs published M3DM]")
    if best_by_cat and m3dm.get("I") is not None:
        for metric, label in (("I", "I-AUROC"), ("P", "P-AUROC"), ("AUPRO", "AUPRO@0.3")):
            ours_vals = [_safe_float(b.get(metric)) for b in best_by_cat]
            ours_vals = [v for v in ours_vals if v is not None]
            ours_mean = sum(ours_vals) / len(ours_vals) if ours_vals else None
            pub = _safe_float(m3dm.get(metric))
            if ours_mean is not None and pub is not None:
                gap = ours_mean - pub
                print("   %-9s ours_mean=%.4f  m3dm=%s  gap=%+.4f"
                      % (label, ours_mean, ("%.4f (reported by prior work)" % pub), gap))
    else:
        print("   no H3 results yet - run run_all.py first.")


def main():
    run_status()


if __name__ == "__main__":
    main()