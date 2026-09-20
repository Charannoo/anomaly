"""E3 late fusion (Phase 3A): calibrated RGB+Depth anomaly fusion, no retraining.

Pipeline:
  1. Load frozen Phase 2 test predictions (RGB + depth .npz sets).
  2. Pair by sample_id with LOUD validation (ids, GT, label, defect, shape).
  3. Calibrate each map with validation-normal median/MAD
     (experiments/calibration/E3_late_fusion_calibration.json):
         Z_m = (A_m - median_m) / scale_m        (no clamping; documented)
  4. Fuse (invalid depth -> RGB only, documented):
         mean: where valid: 0.5*Zr + 0.5*Zd else Zr
         max:  where valid: max(Zr, Zd)         else Zr
  5. Metrics (verified implementations): I-AUROC x3 variants, P-AUROC,
     AUPRO, I/P-AUPRC. Tables + deltas vs RGB / Depth / BestSingle.
  6. Bootstrap 95% CI for image AUROC (1000 resamples, seed 0).
  7. Defect-level fusion AUROC (extends defect_level_auroc.csv logic).
  8. Complementarity classes A-G via validation-calibrated max thresholds.
  9. Representative 6-panel figures under experiments/figures/late_fusion/.

Usage:
    python scripts/run_late_fusion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from xmvad.data.depth import depth_to_uint8, extract_depth, load_xyz, valid_mask_from_xyz
from xmvad.data.mvtec3d import discover_split
from xmvad.metrics.detection import image_auprc, image_auroc
from xmvad.metrics.localization import pixel_auprc, pixel_auroc
from xmvad.metrics.pro import aupro
from xmvad.metrics.scores import SCORE_VARIANTS, variant_scores


# ---------------------------------------------------------------- pairing

def load_pred_set(pred_dir: Path, category: str) -> dict[str, dict]:
    """Load one modality's .npz set; reconstruct sample_id from filenames.

    Filenames are ``{category}_{split}_{defect}_{stem}`` (Phase 2 did not
    store sample_id inside the npz). Stems are numeric; defect names contain
    no underscores, but parsing tolerates them via first/last splitting.
    """
    out = {}
    for f in sorted(pred_dir.glob("*.npz")):
        rest = f.stem
        prefix = category + "_"
        if not rest.startswith(prefix):
            raise ValueError(f"[{category}] unexpected prediction filename: {f.name}")
        rest = rest[len(prefix):]
        parts = rest.split("_")
        if len(parts) < 3:
            raise ValueError(f"[{category}] cannot parse prediction filename: {f.name}")
        split, filestem = parts[0], parts[-1]
        defect = "_".join(parts[1:-1])
        sid = f"{category}/{split}/{defect}/{filestem}"
        z = np.load(f, allow_pickle=True)
        out[sid] = {"map": z["map"].astype(np.float32), "gt": z["gt"].astype(bool),
                    "label": int(z["label"]), "defect": str(z["defect"]),
                    "score_max": float(z["score_max"] if "score_max" in z else z["score"])}
    return out


def pair_predictions(rgb_set: dict, depth_set: dict, category: str) -> list[dict]:
    """Pair RGB/depth predictions; FAIL LOUDLY on any mismatch."""
    if set(rgb_set) != set(depth_set):
        only_rgb = sorted(set(rgb_set) - set(depth_set))[:5]
        only_d = sorted(set(depth_set) - set(rgb_set))[:5]
        raise ValueError(f"[{category}] ID mismatch: rgb-only={only_rgb} depth-only={only_d}")
    pairs = []
    for sid in sorted(rgb_set):
        r, d = rgb_set[sid], depth_set[sid]
        if r["map"].shape != d["map"].shape:
            raise ValueError(f"[{sid}] map shape {r['map'].shape} vs {d['map'].shape}")
        if r["gt"].shape != d["gt"].shape or not np.array_equal(r["gt"], d["gt"]):
            raise ValueError(f"[{sid}] GT mismatch between modalities")
        if r["label"] != d["label"] or r["defect"] != d["defect"]:
            raise ValueError(f"[{sid}] label/defect mismatch")
        pairs.append({"sample_id": sid, "rgb_map": r["map"], "depth_map": d["map"],
                      "gt": r["gt"], "label": r["label"], "defect": r["defect"],
                      "rgb_score": r["score_max"], "depth_score": d["score_max"]})
    return pairs


def xyz_path_for(root: Path, sample_id: str) -> Path:
    """Resolve the raw xyz file backing a sample_id (tolerates .tiff/.tif)."""
    parts = sample_id.split("/")
    category, split, defect, stem = parts[0], parts[1], parts[-2], parts[-1]
    xyz_dir = root / category / split / defect / "xyz"
    for ext in (".tiff", ".tif"):
        p = xyz_dir / f"{stem}{ext}"
        if p.exists():
            return p
    cands = sorted(xyz_dir.glob(stem + ".tif*"))
    if not cands:
        raise FileNotFoundError(f"XYZ backing not found for {sample_id}")
    return cands[0]


def valid_mask_for(category: str, sample_id: str, root: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    """Reload the depth validity mask from RAW xyz (no model involved)."""
    v = valid_mask_from_xyz(load_xyz(xyz_path_for(root, sample_id)))
    h, w = shape_hw
    return np.array(Image.fromarray((v.astype(np.uint8)) * 255).resize((w, h), Image.NEAREST)) > 127


# ---------------------------------------------------------------- bootstrap

def bootstrap_auroc(y: np.ndarray, s: np.ndarray, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        from sklearn.metrics import roc_auc_score

        vals.append(float(roc_auc_score(y[idx], s[idx])))
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- figures

def save_fusion_fig(path: Path, rgb_u8, depth_u8, gt, rgb_map, depth_map, fused, score, title):
    import matplotlib.cm as cm

    def overlay(base, m):
        m = np.clip(m, 0, 1)
        col = (cm.jet(m)[..., :3] * 255).astype(np.uint8)
        return (0.55 * base.astype(float) + 0.45 * col.astype(float)).astype(np.uint8)

    gt_show = (gt.astype(np.uint8) * 255)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(title, fontsize=9)
    axes[0, 0].imshow(rgb_u8)
    axes[0, 0].set_title("RGB")
    axes[0, 1].imshow(depth_u8, cmap="gray")
    axes[0, 1].set_title("depth")
    axes[0, 2].imshow(gt_show, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("GT")
    axes[1, 0].imshow(np.clip(rgb_map, 0, 1), cmap="jet", vmin=0, vmax=1)
    axes[1, 0].set_title("RGB map (0-1 raw)")
    axes[1, 1].imshow(np.clip(depth_map, 0, 1), cmap="jet", vmin=0, vmax=1)
    axes[1, 1].set_title("Depth map (0-1 raw)")
    axes[1, 2].imshow(overlay(rgb_u8, fused / (np.abs(fused).max() + 1e-9) * 0.5 + 0.25))
    axes[1, 2].set_title(f"fused overlay s={score:.2f}")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-preds", default="experiments/predictions/E1_baseline_rgb")
    ap.add_argument("--depth-preds", default="experiments/predictions/E2_baseline_depth")
    ap.add_argument("--calibration", default="experiments/calibration/E3_late_fusion_calibration.json")
    ap.add_argument("--data", default="data/raw/mvtec3d")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    with open(REPO_ROOT / args.calibration, encoding="utf-8") as f:
        calib = json.load(f)
    root = REPO_ROOT / args.data
    rgb_base = REPO_ROOT / args.rgb_preds
    depth_base = REPO_ROOT / args.depth_preds
    cats = args.categories or sorted(calib["categories"])
    out_pred = REPO_ROOT / "experiments" / "predictions" / "E3_late_fusion"
    fig_base = REPO_ROOT / "experiments" / "figures" / "late_fusion"

    metric_rows, boot_rows, defect_rows, compl_rows = [], [], [], []
    for cat in cats:
        print(f"== {cat} ==")
        pairs = pair_predictions(load_pred_set(rgb_base / cat, cat),
                                 load_pred_set(depth_base / cat, cat), cat)
        cm, sm = calib["categories"][cat]["rgb"]["median"], calib["categories"][cat]["rgb"]["scale"]
        dm, smd = calib["categories"][cat]["depth"]["median"], calib["categories"][cat]["depth"]["scale"]
        thr = calib["categories"][cat]["score_thresholds"]
        H, W = pairs[0]["rgb_map"].shape
        fused_mean_maps, fused_max_maps, gts, labels = [], [], [], []
        v_max, v_t1, v_t01 = [], [], []
        rgb_u8s, depth_u8s = [], []
        (out_pred / cat).mkdir(parents=True, exist_ok=True)
        for p in tqdm(pairs, desc=f"fuse {cat}", leave=False):
            v = valid_mask_for(cat, p["sample_id"], root, (H, W))
            zr = (p["rgb_map"].astype(np.float64) - cm) / sm
            zd = (p["depth_map"].astype(np.float64) - dm) / smd
            fm = np.where(v, 0.5 * zr + 0.5 * zd, zr)  # invalid -> RGB only
            fx = np.where(v, np.maximum(zr, zd), zr)
            fused_mean_maps.append(fm.astype(np.float32))
            fused_max_maps.append(fx.astype(np.float32))
            gts.append(p["gt"])
            labels.append(p["label"])
            safe = p["sample_id"].replace("/", "_")
            np.savez_compressed(out_pred / cat / f"{safe}.npz", map_mean=fm.astype(np.float32),
                                map_max=fx.astype(np.float32), gt=p["gt"],
                                label=np.int64(p["label"]), defect=p["defect"],
                                sample_id=p["sample_id"])
        labels_np = np.array(labels)
        ms = {name: np.array([variant_scores(m, None)[name] for m in fused_mean_maps])
              for name, _ in SCORE_VARIANTS}
        xs = {name: np.array([variant_scores(m, None)[name] for m in fused_max_maps])
              for name, _ in SCORE_VARIANTS}
        row: dict = {"category": cat}
        for name in ms:
            row[f"mean_image_auroc_{name}"] = image_auroc(labels_np, ms[name])
            row[f"mean_image_auprc_{name}"] = image_auprc(labels_np, ms[name])
            row[f"max_image_auroc_{name}"] = image_auroc(labels_np, xs[name])
            row[f"max_image_auprc_{name}"] = image_auprc(labels_np, xs[name])
        row["mean_pixel_auroc"] = pixel_auroc(gts, fused_mean_maps)
        row["mean_pixel_auprc"] = pixel_auprc(gts, fused_mean_maps)
        row["mean_aupro"] = aupro(gts, fused_mean_maps)
        row["max_pixel_auroc"] = pixel_auroc(gts, fused_max_maps)
        row["max_pixel_auprc"] = pixel_auprc(gts, fused_max_maps)
        row["max_aupro"] = aupro(gts, fused_max_maps)
        row["num_test_images"] = len(pairs)
        row["num_anomalous"] = int(labels_np.sum())
        metric_rows.append(row)

        # Bootstrap CI on primary max-variant image AUROC.
        for tag, arr in (("rgb", np.array([p["rgb_score"] for p in pairs])),
                         ("depth", np.array([p["depth_score"] for p in pairs])),
                         ("fused_mean", ms["max"])):
            mu, lo, hi = bootstrap_auroc(labels_np, arr, n_boot=args.n_boot)
            boot_rows.append({"category": cat, "method": tag, "auroc": round(mu, 4),
                              "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
        # Fusion-minus-branch diff CIs (paired resampling).
        rng = np.random.default_rng(0)
        for tag in ("fused-rgb", "fused-depth"):
            base = np.array([p["rgb_score"] for p in pairs]) if tag == "fused-rgb" else np.array([p["depth_score"] for p in pairs])
            diffs = []
            for _ in range(args.n_boot):
                idx = rng.integers(0, len(labels_np), len(labels_np))
                if np.unique(labels_np[idx]).size < 2:
                    continue
                from sklearn.metrics import roc_auc_score

                diffs.append(float(roc_auc_score(labels_np[idx], ms["max"][idx])
                                   - roc_auc_score(labels_np[idx], base[idx])))
            diffs = np.array(diffs)
            boot_rows.append({"category": cat, "method": f"diff_{tag}", "auroc": round(float(diffs.mean()), 4),
                              "ci_lo": round(float(np.percentile(diffs, 2.5)), 4),
                              "ci_hi": round(float(np.percentile(diffs, 97.5)), 4)})

        # Defect-level fusion AUROC (normal vs each defect, max variant).
        normal_idx = [i for i, l in enumerate(labels) if l == 0]
        n_scores = ms["max"][normal_idx]
        by_defect: dict[str, list[int]] = {}
        for i, p in enumerate(pairs):
            if p["label"] == 1:
                by_defect.setdefault(p["defect"], []).append(i)
        for dft, idxs in sorted(by_defect.items()):
            s = ms["max"][idxs]
            y = np.array([0] * len(n_scores) + [1] * len(s))
            from sklearn.metrics import roc_auc_score

            defect_rows.append({"category": cat, "defect": dft, "n_normal": len(n_scores),
                                "n_defect": len(s),
                                "fusion_image_auroc": round(float(roc_auc_score(y, np.concatenate([n_scores, s]))), 4)})

        # Complementarity classes via validation-calibrated max thresholds.
        t_r, t_d, t_f = thr["rgb_max_p95"], thr["depth_max_p95"], thr["fused_max_p95"]
        (fig_base / cat).mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(pairs):
            cr = (p["rgb_score"] >= t_r) == bool(p["label"])
            cd = (p["depth_score"] >= t_d) == bool(p["label"])
            cf = (ms["max"][i] >= t_f) == bool(p["label"])
            if cr and cd:
                cls = "C_both_correct"
            elif cr and not cd:
                cls = "A_rgb_only"
            elif cd and not cr:
                cls = "B_depth_only"
            else:
                cls = "D_both_wrong"
            if cf and not cr:
                cls += "+E_fixes_rgb"
            if cf and not cd:
                cls += "+F_fixes_depth"
            if not cf and (cr or cd):
                cls += "+G_breaks"
            compl_rows.append({"sample_id": p["sample_id"], "category": cat, "label": p["label"],
                               "class": cls})
            # Save figures for informative classes (limit per class).
        # Representative figures: first 3 of E/F/G + 2 of A/B per category.
        want: dict[str, int] = {}
        for r in [r for r in compl_rows if r["category"] == cat]:
            key = ("FIX" if ("+E" in r["class"] or "+F" in r["class"]) else
                   ("BREAK" if "+G" in r["class"] else
                    ("A" if r["class"] == "A_rgb_only" else ("B" if r["class"] == "B_depth_only" else None))))
            if key is None:
                continue
            if want.get(key, 0) >= 3:
                continue
            want[key] = want.get(key, 0) + 1
            i = next(i for i, p in enumerate(pairs) if p["sample_id"] == r["sample_id"])
            p = pairs[i]
            rec = discover_split(root, cat, p["sample_id"].split("/")[1])
            rgb_path = next(x.rgb_path for x in rec if x.sample_id == p["sample_id"])
            from PIL import Image as I

            rgb_u8 = np.array(I.open(rgb_path).convert("RGB").resize((W, H)))
            d_raw = load_xyz(xyz_path_for(root, p["sample_id"]))
            from xmvad.data.depth import valid_mask_from_xyz as vmf
            vv = np.array(I.fromarray((vmf(d_raw).astype(np.uint8)) * 255).resize((W, H), I.NEAREST)) > 127
            depth_u8 = depth_to_uint8(extract_depth(d_raw), vmf(d_raw))
            depth_u8 = np.array(I.fromarray(depth_u8).resize((W, H)))
            save_fusion_fig(fig_base / cat / f"{key}_{p['sample_id'].replace('/', '_')}.png",
                            rgb_u8, depth_u8, p["gt"], p["rgb_map"], p["depth_map"],
                            fused_mean_maps[i], ms["max"][i], f"{r['class']} {p['sample_id']}")

    tables = REPO_ROOT / "experiments" / "tables"
    pd.DataFrame(metric_rows).to_csv(tables / "E3_late_fusion_metrics.csv", index=False)

    # Comparison vs frozen Phase 2 tables.
    rgb_df = pd.read_csv(tables / "E1_baseline_rgb_metrics.csv")
    depth_df = pd.read_csv(tables / "E2_baseline_depth_metrics.csv")
    fused_df = pd.DataFrame(metric_rows)
    comp = []
    for _, fr in fused_df.iterrows():
        cat = fr["category"]
        rr = rgb_df[rgb_df.category == cat].iloc[0]
        dr = depth_df[depth_df.category == cat].iloc[0]
        for metric, rcol, dcol, fkey in (
                ("I-AUROC(max)", "image_auroc_max", "image_auroc_max", "mean_image_auroc_max"),
                ("I-AUROC(topk1%)", "image_auroc_topk_1pct", "image_auroc_topk_1pct", "mean_image_auroc_topk_1pct"),
                ("P-AUROC", "pixel_auroc", "pixel_auroc", "mean_pixel_auroc"),
                ("AUPRO", "aupro", "aupro", "mean_aupro")):
            rv, dv, fv = float(rr[rcol]), float(dr[dcol]), float(fr[fkey])
            best = max(rv, dv)
            comp.append({"category": cat, "metric": metric, "RGB": round(rv, 4),
                         "Depth": round(dv, 4), "LateFusion": round(fv, 4),
                         "d_vs_RGB": round(fv - rv, 4), "d_vs_Depth": round(fv - dv, 4),
                         "d_vs_BestSingle": round(fv - best, 4)})
    # MEAN row
    for metric, fkey, rcol, dcol in (
            ("I-AUROC(max)", "mean_image_auroc_max", "image_auroc_max", "image_auroc_max"),
            ("I-AUROC(topk1%)", "mean_image_auroc_topk_1pct", "image_auroc_topk_1pct", "image_auroc_topk_1pct"),
            ("P-AUROC", "mean_pixel_auroc", "pixel_auroc", "pixel_auroc"),
            ("AUPRO", "mean_aupro", "aupro", "aupro")):
        fvals = fused_df[fkey].to_numpy(float)
        rvals = rgb_df[rgb_df.category != "MEAN"][rcol].to_numpy(float)
        dvals = depth_df[depth_df.category != "MEAN"][dcol].to_numpy(float)
        fv, rv, dv = float(np.nanmean(fvals)), float(np.nanmean(rvals)), float(np.nanmean(dvals))
        comp.append({"category": "MEAN", "metric": metric, "RGB": round(rv, 4),
                     "Depth": round(dv, 4), "LateFusion": round(fv, 4),
                     "d_vs_RGB": round(fv - rv, 4), "d_vs_Depth": round(fv - dv, 4),
                     "d_vs_BestSingle": round(fv - max(rv, dv), 4)})
    pd.DataFrame(comp).to_csv(tables / "E3_vs_baselines.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(tables / "E3_bootstrap_ci.csv", index=False)
    pd.DataFrame(defect_rows).to_csv(tables / "E3_defect_level.csv", index=False)
    pd.DataFrame(compl_rows).to_csv(tables / "E3_complementarity.csv", index=False)
    print("wrote E3 tables + figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
