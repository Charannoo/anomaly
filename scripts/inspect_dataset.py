"""Dataset inspection for MVTec 3D-AD (Phase 1).

Prints per category:
- split counts, normal/anomalous counts
- RGB shape, XYZ shape
- depth min/max (valid pixels), invalid depth %
- mask shape, alignment warnings

Usage:
    python scripts/inspect_dataset.py --data data/raw/mvtec3d
    python scripts/inspect_dataset.py --data data/raw/mvtec3d --categories bagel rope --limit 20
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from PIL import Image

from xmvad.data.depth import depth_range, extract_depth, invalid_fraction, load_xyz, valid_mask_from_xyz
from xmvad.data.mvtec3d import (
    EXPECTED_CATEGORIES,
    discover_categories,
    discover_split,
    find_pairing_gaps,
    load_gt_mask,
)


def _rgb_shape(path: Path) -> tuple[int, int, int] | None:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return (im.size[1], im.size[0], 3)
    except Exception:
        return None


def inspect_category(root: Path, category: str, limit: int = 0) -> dict:
    """Inspect one category across train/validation/test; return summary dict."""
    summary: dict = {"category": category, "splits": {}, "gaps": [], "warnings": []}
    for split in ("train", "validation", "test"):
        records = discover_split(root, category, split)
        if limit > 0:
            records = records[:limit]
        labels = Counter(r.label for r in records)
        defects = Counter(r.defect for r in records)
        summary["splits"][split] = {
            "total": len(records),
            "normal": labels.get(0, 0),
            "anomalous": labels.get(1, 0),
            "defects": dict(defects),
        }
        gaps = find_pairing_gaps(root, category, split)
        summary["gaps"].extend(gaps)
    return summary


def probe_samples(root: Path, category: str, n_probe: int = 5) -> list[dict]:
    """Load up to n_probe samples per split and report array-level stats."""
    rows: list[dict] = []
    for split in ("train", "validation", "test"):
        records = discover_split(root, category, split)[:n_probe]
        for rec in records:
            row: dict = {"sample_id": rec.sample_id, "label": rec.label}
            rgb_s = _rgb_shape(rec.rgb_path)
            row["rgb_shape"] = rgb_s
            try:
                xyz = load_xyz(rec.xyz_path)
                row["xyz_shape"] = tuple(xyz.shape)
                valid = valid_mask_from_xyz(xyz)
                depth = extract_depth(xyz)
                dmin, dmax = depth_range(depth, valid)
                row["depth_min"] = dmin
                row["depth_max"] = dmax
                row["invalid_pct"] = round(invalid_fraction(valid) * 100, 2)
                if rgb_s is not None and (xyz.shape[0], xyz.shape[1]) != (rgb_s[0], rgb_s[1]):
                    row["align_warn"] = f"RGB {rgb_s[0]}x{rgb_s[1]} vs XYZ {xyz.shape[0]}x{xyz.shape[1]}"
                else:
                    row["align_warn"] = ""
            except Exception as e:  # keep inspection robust to corrupt files
                row["xyz_shape"] = None
                row["depth_min"] = row["depth_max"] = float("nan")
                row["invalid_pct"] = float("nan")
                row["align_warn"] = f"XYZ load error: {e}"
            gt = load_gt_mask(rec.gt_path) if rec.gt_path else None
            if rec.label == 1 and rec.gt_path is None:
                row["mask_shape"] = None
                row["mask_warn"] = "anomalous sample without GT file"
            elif gt is not None:
                row["mask_shape"] = tuple(gt.shape)
                if rgb_s is not None and tuple(gt.shape) != (rgb_s[0], rgb_s[1]):
                    row["mask_warn"] = f"GT {gt.shape} vs RGB {(rgb_s[0], rgb_s[1])}"
                else:
                    row["mask_warn"] = ""
            else:
                row["mask_shape"] = None
                row["mask_warn"] = ""
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect MVTec 3D-AD dataset structure + arrays.")
    ap.add_argument("--data", required=True, help="Path to data/raw/mvtec3d")
    ap.add_argument("--categories", nargs="*", default=None, help="Subset of categories (default: all found)")
    ap.add_argument("--limit", type=int, default=0, help="Max records per split for counting (0 = all)")
    ap.add_argument("--probe", type=int, default=5, help="Samples per split to open and measure")
    args = ap.parse_args()

    root = Path(args.data)
    if not root.is_dir():
        print(f"Dataset root not found: {root}")
        print("Dataset required: MVTec 3D-AD. Download it from the official MVTec 3D-AD")
        print("dataset page and extract/copy it to data/raw/mvtec3d/.")
        return 2

    found = discover_categories(root)
    if not found:
        print(f"No category subdirectories found under {root}. Expected e.g. bagel/, rope/, ...")
        print("Dataset required: MVTec 3D-AD. Download it from the official MVTec 3D-AD")
        print("dataset page and extract/copy it to data/raw/mvtec3d/.")
        return 2

    cats = args.categories or found
    missing_expected = [c for c in EXPECTED_CATEGORIES if c not in found]
    extra = [c for c in found if c not in EXPECTED_CATEGORIES]
    print(f"Root: {root}")
    print(f"Categories found ({len(found)}): {', '.join(found)}")
    if missing_expected:
        print(f"WARNING: expected categories missing: {', '.join(missing_expected)}")
    if extra:
        print(f"NOTE: extra unexpected dirs: {', '.join(extra)}")

    try:
        from tabulate import tabulate

        have_tab = True
    except ImportError:
        have_tab = False

    all_ok = True
    for cat in cats:
        print(f"\n{'=' * 70}\nCategory: {cat}")
        summary = inspect_category(root, cat, limit=args.limit)
        rows = [[s, v["total"], v["normal"], v["anomalous"], ", ".join(f"{k}:{n}" for k, n in v["defects"].items())]
                for s, v in summary["splits"].items()]
        if have_tab:
            print(tabulate(rows, headers=["split", "total", "normal", "anomalous", "defect counts"]))
        else:
            for r in rows:
                print("  ", r)
        if summary["gaps"]:
            all_ok = False
            print(f"  PAIRING GAPS ({len(summary['gaps'])}):")
            for g in summary["gaps"][:20]:
                print(f"    {g['split']}/{g['defect']}/{g['stem']}: {g['kind']}")
            if len(summary["gaps"]) > 20:
                print(f"    ... and {len(summary['gaps']) - 20} more")
        probes = probe_samples(root, cat, n_probe=args.probe)
        prows = [[p["sample_id"].split("/")[-2] + "/" + p["sample_id"].split("/")[-1], p["rgb_shape"],
                  p["xyz_shape"], p.get("depth_min"), p.get("depth_max"), p.get("invalid_pct"),
                  p.get("mask_shape"), (p.get("align_warn") or "") + (" | " + p["mask_warn"] if p.get("mask_warn") else "")]
                 for p in probes]
        print("  Sample probes (RGB HxWxC, XYZ HxWx3, depth valid min/max, invalid %, GT shape):")
        if have_tab:
            print(tabulate(prows, headers=["sample", "RGB", "XYZ", "dmin", "dmax", "inv%", "GT", "warnings"],
                           floatfmt=".3f"))
        else:
            for r in prows:
                print("   ", r)
        warns = [p for p in probes if p.get("align_warn") or p.get("mask_warn")]
        if warns:
            all_ok = False
            print(f"  WARNINGS: {len(warns)} probed samples with alignment/GT warnings (see above).")

    print(f"\n{'=' * 70}")
    if all_ok:
        print("OK: no pairing gaps or alignment warnings in probed samples.")
    else:
        print("DONE with warnings — review pairing gaps / alignment warnings above before training.")
        print("Do not begin fusion experiments if obvious RGB-depth registration errors exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
