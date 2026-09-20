"""Mask-survival audit (Phase 2 hardening H2).

Question: does the conservative valid-mask downsampling (area + 0.999 rule)
throw away absurd amounts of valid geometry before the depth loss/maps?

For every TRAIN-NORMAL sample (no test data involved), report the fraction
of valid pixels at:
  - input resolution (config input_size, NEAREST resize of the raw mask)
  - stage8  (input/8,  e.g. 32x32 @256)
  - stage16 (input/16, e.g. 16x16 @256)

Writes experiments/tables/mask_survival.csv with mean/min per category.

Usage:
    python scripts/check_mask_survival.py --data data/raw/mvtec3d --size 256 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from xmvad.data.depth import load_xyz, valid_mask_from_xyz
from xmvad.data.mvtec3d import discover_categories, discover_split
from xmvad.losses.distillation import downsample_valid_mask


def mask_at_input_size(valid: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    img = Image.fromarray(valid.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
    return np.array(img) > 127


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/mvtec3d")
    ap.add_argument("--size", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--categories", nargs="*", default=None)
    args = ap.parse_args()

    root = Path(args.data)
    if not root.is_dir():
        print(f"Dataset root not found: {root}")
        return 2
    H, W = args.size
    cats = args.categories or discover_categories(str(root))
    rows = []
    for cat in cats:
        recs = [r for r in discover_split(root, cat, "train") if r.label == 0]
        vin, v8, v16 = [], [], []
        for r in tqdm(recs, desc=cat, leave=False):
            try:
                v = valid_mask_from_xyz(load_xyz(r.xyz_path))
            except Exception:
                continue
            v_in = mask_at_input_size(v, (H, W))
            t = torch.from_numpy(v_in).unsqueeze(0).unsqueeze(0)
            vin.append(v_in.mean())
            v8.append(downsample_valid_mask(t, (H // 8, W // 8)).float().mean().item())
            v16.append(downsample_valid_mask(t, (H // 16, W // 16)).float().mean().item())
        vin, v8, v16 = np.array(vin), np.array(v8), np.array(v16)
        rows.append({
            "category": cat, "n_train": len(vin),
            "input_valid_mean%": round(100 * vin.mean(), 2),
            "input_valid_min%": round(100 * vin.min(), 2),
            "stage8_valid_mean%": round(100 * v8.mean(), 2),
            "stage8_valid_min%": round(100 * v8.min(), 2),
            "stage16_valid_mean%": round(100 * v16.mean(), 2),
            "stage16_valid_min%": round(100 * v16.min(), 2),
        })
    df = pd.DataFrame(rows)
    out = REPO_ROOT / "experiments" / "tables" / "mask_survival.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
