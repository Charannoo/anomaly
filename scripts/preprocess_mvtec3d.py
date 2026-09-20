"""Preprocessing: fit train-normal depth stats + cache resized tensors (Phase 1).

- Fits per-category robust depth stats on TRAIN-NORMAL Z pixels only.
- Optionally writes resized (RGB/depth/valid/GT) .npz files to processed root.
- Never writes into data/raw; never touches test data for stat fitting.

Usage:
    python scripts/preprocess_mvtec3d.py --input data/raw/mvtec3d --output data/processed/mvtec3d
    python scripts/preprocess_mvtec3d.py --input data/raw/mvtec3d --output data/processed/mvtec3d --stats-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from PIL import Image
from tqdm import tqdm

from xmvad.data.depth import extract_depth, fit_depth_stats, load_xyz, normalize_depth, valid_mask_from_xyz
from xmvad.data.mvtec3d import discover_categories, discover_split, load_gt_mask


def fit_category_stats(root: Path, category: str, method: str, max_files: int = 0) -> dict:
    recs = [r for r in discover_split(root, category, "train") if r.label == 0]
    if max_files > 0:
        recs = recs[:max_files]
    pooled: list[np.ndarray] = []
    for r in tqdm(recs, desc=f"fit {category}", leave=False):
        try:
            xyz = load_xyz(r.xyz_path)
        except Exception:
            continue
        d, v = extract_depth(xyz), valid_mask_from_xyz(xyz)
        if v.any():
            pooled.append(d[v].ravel().astype(np.float64))
    if not pooled:
        raise RuntimeError(f"No valid train-normal depth pixels for category {category!r}.")
    stats = fit_depth_stats(pooled, method=method)
    stats["n_files"] = len(recs)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--method", default="robust_median", choices=["robust_median", "percentile"])
    ap.add_argument("--size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--max-files", type=int, default=0, help="Cap train files per category for stat fit (0=all)")
    ap.add_argument("--categories", nargs="*", default=None)
    args = ap.parse_args()

    root, out = Path(args.input), Path(args.output)
    if not root.is_dir():
        print(f"Input root not found: {root}")
        print("Dataset required: MVTec 3D-AD. Download it from the official MVTec 3D-AD")
        print("dataset page and extract/copy it to data/raw/mvtec3d/.")
        return 2
    cats = args.categories or discover_categories(root)
    out.mkdir(parents=True, exist_ok=True)
    H, W = args.size

    all_stats: dict = {}
    for cat in cats:
        stats = fit_category_stats(root, cat, args.method, args.max_files)
        all_stats[cat] = stats
        print(f"[{cat}] median/scale or percentiles: {stats}")
    with open(out / "depth_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Wrote {out / 'depth_stats.json'} (fit on train-normal only).")

    if args.stats_only:
        return 0

    # Cache resized tensors per record.
    for cat in cats:
        stats = all_stats[cat]
        for split in ("train", "validation", "test"):
            for rec in tqdm(discover_split(root, cat, split), desc=f"cache {cat}/{split}", leave=False):
                try:
                    rgb = Image.open(rec.rgb_path).convert("RGB").resize((W, H), Image.BILINEAR)
                    xyz = load_xyz(rec.xyz_path)
                except Exception as e:
                    print(f"skip {rec.sample_id}: {e}")
                    continue
                from PIL import Image as _I

                d = np.array(Image.fromarray(extract_depth(xyz), mode="F").resize((W, H), _I.NEAREST), dtype=np.float32)
                v = np.array(Image.fromarray((valid_mask_from_xyz(xyz).astype(np.uint8)) * 255).resize((W, H), _I.NEAREST)) > 127
                dn = normalize_depth(d, stats, valid_mask=v, clip=(-3.0, 3.0))
                gt = load_gt_mask(rec.gt_path, size=(W, H))
                dest = out / cat / split / rec.defect / (Path(rec.rgb_path).stem + ".npz")
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, rgb=np.array(rgb, dtype=np.uint8), depth=dn.astype(np.float32),
                                    valid=v, gt=gt if gt is not None else np.zeros((H, W), dtype=bool),
                                    has_gt=gt is not None)
    print(f"Cached resized tensors under {out} (H={H}, W={W}). Raw data untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
