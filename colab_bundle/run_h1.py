#!/usr/bin/env python3
"""H1: DINO ViT-B/8 RGB memory-bank baseline (M3DM protocol).

One coreset bank per category (f_coreset=0.1, eps=0.9), exact-L2 nearest
neighbour scoring, PatchCore-style reweighted score + 224x224 MapCore map,
I-AUROC / P-AUROC / AUPRO@0.3.  Output: results/H1_RGB.csv.
"""

import os
import time

import numpy as np
import torch

import cache_utils
from cache_utils import (concat_descriptors, evaluate_bank, load_blur,
                         load_config, load_split_samples, cfg_get, ensure_dir)
from metrics import mean_row, write_results_csv

MODEL = "H1_RGB"


def run_h1(cfg, categories=None, blur=None):
    categories = categories or cfg["categories"]
    results_dir = cfg["results_dir"]
    ensure_dir(results_dir)

    if blur is None:
        blur = load_blur()

    rows = []
    for category in categories:
        t0 = time.time()
        train_desc = concat_descriptors(load_split_samples(cfg["features_dir"], category, "train", "rgb"))
        test_samples = load_split_samples(cfg["features_dir"], category, "test", "rgb")
        if train_desc.numel() == 0:
            print("[skip] %s: empty rgb train shard" % category)
            continue
        row, _s, _m, _l, _n = evaluate_bank(cfg, category, train_desc, (28, 28),
                                            test_samples, "%s_h1_rgb" % category,
                                            cfg["features_dir"], blur)
        rows.append(row)
        print("[H1] %-12s I=%.4f P=%.4f AUPRO=%s (%.1fs)"
              % (category, row["I"] or float("nan"), row["P"] or float("nan"),
                 ("%.4f" % row["AUPRO"]) if row["AUPRO"] is not None else "  n/a", time.time() - t0),
              flush=True)

    if not rows:
        print("[H1] no results produced")
        return rows

    rows.append(mean_row(rows))
    path = os.path.join(results_dir, "H1_RGB.csv")
    write_results_csv(path, rows, provenance="measured_by_us", model=MODEL)
    m = rows[-1]
    print("\n---> H1 MEAN  I=%.4f P=%.4f AUPRO=%s  -> %s"
          % (m["I"] or float("nan"), m["P"] or float("nan"),
             ("%.4f" % m["AUPRO"]) if m["AUPRO"] is not None else "  n/a", path))
    return rows


if __name__ == "__main__":
    cfg = load_config()
    run_h1(cfg)