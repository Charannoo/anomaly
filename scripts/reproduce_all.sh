#!/usr/bin/env bash
# Full reproduction pipeline (requires dataset + weights; Phase 1 checks data stage only).
set -euo pipefail
echo "== XMV-AD reproduce_all (Phase 1: data validation only) =="
python scripts/inspect_dataset.py --data data/raw/mvtec3d
python scripts/visualize_sample.py --data data/raw/mvtec3d --out experiments/figures/data_checks --num-per-category 25
echo "Later phases append: preprocess, train E1-E7, evaluate, benchmark, report."
