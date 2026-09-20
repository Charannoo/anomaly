"""Visual data validation: RGB / depth / masks / edges grid (Phase 1).

Saves, per sampled record, an 8-panel figure:
  RGB | raw depth | normalized depth
  valid mask | GT (or blank) | RGB edges
  depth edges | RGB/depth edge overlay

Usage:
    python scripts/visualize_sample.py --data data/raw/mvtec3d --out experiments/figures/data_checks --num-per-category 25 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from xmvad.data.depth import (
    compute_robust_stats,
    depth_to_uint8,
    extract_depth,
    load_xyz,
    normalize_depth,
    valid_mask_from_xyz,
)
from xmvad.data.mvtec3d import discover_categories, discover_split, load_gt_mask


def _edges(gray_u8: np.ndarray) -> np.ndarray:
    try:
        import cv2

        return cv2.Canny(gray_u8, 50, 150)
    except ImportError:
        gy, gx = np.gradient(gray_u8.astype(np.float64))
        mag = np.hypot(gx, gy)
        thr = np.percentile(mag, 95) if mag.size else 1.0
        return (mag > thr).astype(np.uint8) * 255


def render_sample(rgb: Image.Image, xyz: np.ndarray, gt: np.ndarray | None,
                  stats: dict | None, save_path: Path, title: str) -> None:
    rgb_arr = np.array(rgb.convert("RGB"))
    depth = extract_depth(xyz)
    valid = valid_mask_from_xyz(xyz)
    depth_u8 = depth_to_uint8(depth, valid)
    if stats is not None:
        norm = normalize_depth(depth, stats, valid_mask=valid, clip=(-3.0, 3.0))
    else:
        norm = np.zeros_like(depth, dtype=np.float32)
    norm_u8 = depth_to_uint8(norm, valid)
    gray = np.array(rgb.convert("L"))
    rgb_e = _edges(gray)
    depth_e = _edges(depth_u8)
    overlay = np.zeros((*gray.shape, 3), dtype=np.uint8)
    overlay[..., 0] = rgb_e  # red = RGB edges
    overlay[..., 1] = depth_e  # green = depth edges

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(title, fontsize=10)
    panels = [
        ("RGB", rgb_arr, "image"),
        ("raw depth (u8)", depth_u8, "gray"),
        ("normalized depth (u8)", norm_u8, "gray"),
        ("valid-depth mask", valid.astype(np.uint8) * 255, "gray"),
        ("GT mask", (gt.astype(np.uint8) * 255) if gt is not None else np.zeros_like(gray), "gray"),
        ("RGB edges", rgb_e, "gray"),
        ("depth edges", depth_e, "gray"),
        ("edge overlay R=RGB G=depth", overlay, "image"),
    ]
    for ax, (name, arr, kind) in zip(axes.ravel()[:8], panels):
        if kind == "image":
            ax.imshow(arr)
        else:
            ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    axes.ravel()[8].axis("off")
    axes.ravel()[8].text(0.05, 0.5,
                         f"valid={valid.mean() * 100:.1f}%  depth_med~{np.median(depth[valid]) if valid.any() else float('nan'):.4f}",
                         fontsize=9, transform=axes.ravel()[8].transAxes, va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render RGB/depth validation grids.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="experiments/figures/data_checks")
    ap.add_argument("--num-per-category", type=int, default=25)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.data)
    out = Path(args.out)
    if not root.is_dir():
        print(f"Dataset root not found: {root}")
        print("Dataset required: MVTec 3D-AD. Download it from the official MVTec 3D-AD")
        print("dataset page and extract/copy it to data/raw/mvtec3d/.")
        return 2
    rng = np.random.default_rng(args.seed)
    cats = args.categories or discover_categories(root)
    if not cats:
        print(f"No categories under {root}.")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for cat in cats:
        # Fit quick per-category robust stats on a few train-normal depths for display.
        train_recs = [r for r in discover_split(root, cat, "train") if r.label == 0][:50]
        vals = []
        for r in train_recs:
            try:
                xyz = load_xyz(r.xyz_path)
                d, v = extract_depth(xyz), valid_mask_from_xyz(xyz)
                vals.append(d[v].ravel().astype(np.float64))
            except Exception:
                continue
        stats = compute_robust_stats(np.concatenate(vals), "robust_median") if vals else None
        pool = []
        for split in ("train", "validation", "test"):
            pool.extend(discover_split(root, cat, split))
        if not pool:
            print(f"[{cat}] no records, skipping.")
            continue
        idx = rng.choice(len(pool), size=min(args.num_per_category, len(pool)), replace=False)
        cat_out = out / cat
        cat_out.mkdir(parents=True, exist_ok=True)
        for i in sorted(idx):
            rec = pool[int(i)]
            try:
                rgb = Image.open(rec.rgb_path).convert("RGB")
                xyz = load_xyz(rec.xyz_path)
                gt = load_gt_mask(rec.gt_path, size=rgb.size)
            except Exception as e:
                print(f"[{cat}] skip {rec.sample_id}: {e}")
                continue
            save = cat_out / f"{rec.split}_{rec.defect}_{Path(rec.rgb_path).stem}.png"
            render_sample(rgb, xyz, gt, stats, save, rec.sample_id)
            total += 1
        print(f"[{cat}] wrote {min(args.num_per_category, len(pool))} figures -> {cat_out}")
    print(f"Done: {total} figures under {out}. Review overlays for registration errors")
    print("before any fusion experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
