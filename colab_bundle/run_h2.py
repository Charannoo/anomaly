#!/usr/bin/env python3
"""H2: Point-MAE XYZ memory-bank baseline (M3DM protocol).

Same coreset + exact-NN protocol as H1 but on 56x56 interpolated Point-MAE
tokens (3136 x 1152 per sample).  Output: results/H2_XYZ.csv.
"""

import os
import time

import cache_utils
from cache_utils import (concat_descriptors, evaluate_bank, load_blur,
                         load_config, load_split_samples, ensure_dir)
from metrics import mean_row, write_results_csv

MODEL = "H2_XYZ"


def run_h2(cfg, categories=None, blur=None):
    categories = categories or cfg["categories"]
    results_dir = cfg["results_dir"]
    ensure_dir(results_dir)

    if blur is None:
        blur = load_blur()

    rows = []
    for category in categories:
        t0 = time.time()
        train_desc = concat_descriptors(load_split_samples(cfg["features_dir"], category, "train", "xyz"))
        test_samples = load_split_samples(cfg["features_dir"], category, "test", "xyz")
        if train_desc.numel() == 0:
            print("[skip] %s: empty xyz train shard" % category)
            continue
        row, _s, _m, _l, _n = evaluate_bank(cfg, category, train_desc, (56, 56),
                                            test_samples, "%s_h2_xyz" % category,
                                            cfg["features_dir"], blur)
        rows.append(row)
        print("[H2] %-12s I=%.4f P=%.4f AUPRO=%s (%.1fs)"
              % (category, row["I"] or float("nan"), row["P"] or float("nan"),
                 ("%.4f" % row["AUPRO"]) if row["AUPRO"] is not None else "  n/a", time.time() - t0),
              flush=True)

    if not rows:
        print("[H2] no results produced")
        return rows

    rows.append(mean_row(rows))
    path = os.path.join(results_dir, "H2_XYZ.csv")
    write_results_csv(path, rows, provenance="measured_by_us", model=MODEL)
    m = rows[-1]
    print("\n---> H2 MEAN  I=%.4f P=%.4f AUPRO=%s  -> %s"
          % (m["I"] or float("nan"), m["P"] or float("nan"),
             ("%.4f" % m["AUPRO"]) if m["AUPRO"] is not None else "  n/a", path))
    return rows


if __name__ == "__main__":
    cfg = load_config()
    run_h2(cfg)