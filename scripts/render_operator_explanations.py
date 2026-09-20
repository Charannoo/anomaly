"""Phase 5, todo 11: operator explanation figures.

Six deterministic explanation figures:
  1. case1_synthetic_rgb_dominant    - bagel, synthetic color_patch (strong)
  2. case2_synthetic_depth_dominant  - bagel, synthetic dent      (strong)
  3. case3_synthetic_joint           - bagel, synthetic joint_blob (strong)
  4. case4_real_rgb_stronger         - foam color contamination (stored E4 maps)
  5. case5_real_depth_stronger       - rope open 001 (stored E4 maps)
  6. case6_failure                   - cookie crack 000 (stored E4 maps, weakest)

Synthetic maps come from re-running the FROZEN E4 on deterministically
re-injected train-normal inputs (same seed rule as the generator). Real maps
are the stored E4 prediction npz (256-res). Evidence region = GT mask
(real) or injected mask (synthetic). Evidence is relative, not causal.

Panels per figure: input RGB + GT contour, calibrated A_RGB, A_DEPTH,
A_FINAL (mean). Caption = class, p_rgb/p_depth, E_rgb/E_depth, S_full,
|Delta_RGB|, |Delta_DEPTH|.

Writes PNGs + montage to experiments/figures/explainability_failures/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from xmvad.data.depth import extract_depth, load_xyz, normalize_depth, valid_mask_from_xyz
from xmvad.data.mvtec3d import discover_split, load_rgb
from xmvad.data.stats import fit_train_depth_stats
from xmvad.data.transforms import SynchronizedTransform
from xmvad.fusion.score_fusion import calibrate
from xmvad.metrics.attribution import decision_ablation, evidence_class, region_evidence, relative_evidence
from xmvad.metrics.scores import variant_scores
from xmvad.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "experiments" / "figures" / "explainability_failures"
PRED_DIR = REPO_ROOT / "experiments" / "predictions" / "E4_concat_fusion"
CALIB = json.load(open(REPO_ROOT / "experiments" / "tables" / "E4_concat_calibration.json",
                       encoding="utf-8"))["categories"]

_GEN = importlib.util.spec_from_file_location(
    "generate_attribution_data", REPO_ROOT / "scripts" / "generate_attribution_data.py")
gen = importlib.util.module_from_spec(_GEN)
sys.modules["generate_attribution_data"] = gen
_GEN.loader.exec_module(gen)
PLAN, det_seed, apply_variant, tensorize, build_e4 = (
    gen.PLAN, gen.det_seed, gen.apply_variant, gen.tensorize, gen.build_e4)


def resize_arr(a, H, W, nearest=False):
    if a.ndim == 2:
        im = Image.fromarray(np.asarray(a, dtype=np.float32), mode="F")
        if nearest:
            return np.array(im.resize((W, H), Image.NEAREST), dtype=np.float64)
        return np.array(im.resize((W, H), Image.BILINEAR), dtype=np.float64)
    pil = Image.fromarray(np.asarray(a).astype(np.uint8))
    return np.array(pil.resize((W, H), Image.BILINEAR), dtype=np.float64)


def panel(ax, img, title, cmap=None, vmax=None, contour=None, cbar=False):
    if cmap is None:
        ax.imshow(img.astype(np.float64) / 255.0)
    elif cbar:
        im = ax.imshow(img, cmap=cmap, vmin=0.0, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.imshow(img, cmap=cmap, vmin=0.0, vmax=vmax)
    if contour is not None and np.asarray(contour, bool).any():
        ax.contour(np.asarray(contour, bool), colors="red", linewidths=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def render_fig(name, rgb_small, gt, zr, zd, fused, cap, vmax):
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.4))
    hl = rgb_small.astype(np.float64).copy()
    if gt.any():
        hl[gt] = hl[gt] * 0.55 + 0.45 * 255
    panel(axes[0], hl, "input RGB + GT mask", contour=None)
    panel(axes[1], zr, "A_RGB (calibrated)", cmap="jet", vmax=vmax, contour=gt, cbar=True)
    panel(axes[2], zd, "A_DEPTH (calibrated)", cmap="jet", vmax=vmax, contour=gt, cbar=True)
    panel(axes[3], fused, "A_FINAL (50/50 mean)", cmap="jet", vmax=vmax, contour=gt, cbar=True)
    fig.suptitle(cap, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    pth = FIG_DIR / f"{name}.png"
    fig.savefig(pth, dpi=130)
    plt.close(fig)
    return pth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/concat_fusion.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    root = Path(data_cfg.get("root", "data/raw/mvtec3d"))
    input_size = tuple(data_cfg.get("input_size", [256, 256]))
    H, W = input_size
    rgb_mean = tuple(data_cfg.get("rgb_mean", [0.485, 0.456, 0.406]))
    rgb_std = tuple(data_cfg.get("rgb_std", [0.229, 0.224, 0.225]))
    clip = tuple(data_cfg["depth_clip"]) if data_cfg.get("depth_clip") else None
    mc = cfg.get("model", {})

    def cal(cat):
        c = CALIB[cat]
        return c["rgb"]["median"], c["rgb"]["scale"], c["depth"]["median"], c["depth"]["scale"]

    # ---- synthetic helper: epsilon, tensorize path left to build_e4 outputs ----
    def synth_maps(cat, kind, subtype, sev, source_stem):
        depth_stats = fit_train_depth_stats(root, cat, data_cfg.get("depth_norm", "robust_median"),
                                            verbose=False)
        rec = next(r for r in discover_split(root, cat, "train") if r.label == 0
                   and r.xyz_path.stem == source_stem)
        rgb0 = np.array(load_rgb(rec.rgb_path))
        d0 = extract_depth(load_xyz(rec.xyz_path))
        v0 = valid_mask_from_xyz(load_xyz(rec.xyz_path))
        kw = next(kw for k, st, sv, kw in PLAN if k == kind and sv == sev)
        sid = f"{cat}/train/good/{source_stem}"
        seed = det_seed(sid, kind, subtype, sev)
        rgb_p, depth_p, valid_p, mask = apply_variant(rgb0, d0, v0, kind, subtype, seed, kw)
        tf = SynchronizedTransform(input_size=input_size, rgb_mean=rgb_mean, rgb_std=rgb_std,
                                   depth_stats=None, depth_clip=clip, train=False)
        rgb_t, depth_t, valid_t = tensorize(Image.fromarray(rgb_p), depth_p, valid_p,
                                            tf, depth_stats, clip, H, W)
        e4 = build_e4(cfg, cat, "checkpoints/E1_baseline_rgb", "checkpoints/E2_baseline_depth",
                      "checkpoints/E4_concat_fusion", torch.device("cpu"))
        with torch.no_grad():
            out = e4.predict(rgb_t, depth_t, valid_t)
        gt = resize_arr(mask, H, W, nearest=True) > 0.5
        v = valid_t[0, 0].numpy().astype(bool)
        zr = calibrate(out["rgb_map"][0, 0].cpu().numpy().astype(np.float64), *cal(cat)[:2])
        zd = calibrate(out["depth_map"][0, 0].cpu().numpy().astype(np.float64), *cal(cat)[2:])
        fused = np.where(v, 0.5 * zr + 0.5 * zd, zr)
        return rgb_p, zr, zd, fused, gt, v

    cases_table = []
    # Synthetic cases (deterministic first train-normal for bagel).
    bagel_stem = next(r.xyz_path.stem for r in discover_split(root, "bagel", "train")
                      if r.label == 0)
    cases_table.append(("case1_synthetic_rgb_dominant", "bagel", "rgb_only", "color_patch", "strong",
                        bagel_stem, None))
    cases_table.append(("case2_synthetic_depth_dominant", "bagel", "depth_only", "dent", "strong",
                        bagel_stem, None))
    cases_table.append(("case3_synthetic_joint", "bagel", "joint", "joint_blob", "strong",
                        bagel_stem, None))
    # Real cases (stored E4 predictions).
    cases_table.append(("case4_real_rgb_stronger", "foam", None, None, None, None,
                        "foam/test/color/000"))
    cases_table.append(("case5_real_depth_stronger", "rope", None, None, None, None,
                        "rope/test/open/001"))
    cases_table.append(("case6_failure", "cookie", None, None, None, None,
                        "cookie/test/crack/000"))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, cat, kind, subtype, sev, stem, sid in cases_table:
        if sid is not None:
            with np.load(PRED_DIR / cat / (sid.replace("/", "_") + ".npz"), allow_pickle=True) as z:
                der = {k: z[k] for k in z.files}
            gt = np.asarray(der["gt"], dtype=bool)
            v = np.ones(gt.shape, bool)
            med_r, sc_r, med_d, sc_d = cal(cat)
            zr = calibrate(np.asarray(der["rgb_map"], dtype=np.float64), med_r, sc_r)
            zd = calibrate(np.asarray(der["depth_map"], dtype=np.float64), med_d, sc_d)
            fused = 0.5 * zr + 0.5 * zd
            # GT colocalized depth validity is unknown; keep simple 50/50 as stored map_mean.
            fused = np.asarray(der["map_mean"], dtype=np.float64)
            rec = next(r for r in discover_split(root, cat, "test") if r.label == 1
                       and r.xyz_path.stem == sid.split("/")[-1])
            rgb = np.array(load_rgb(rec.rgb_path))
            e_r = region_evidence(zr, gt, mode="topk", frac=0.01)
            e_d = region_evidence(zd, gt, mode="topk", frac=0.01)
        else:
            mp = synth_maps(cat, kind, subtype, sev, stem)
            rgb = mp[0]
            zr, zd, fused, gt, v = mp[1], mp[2], mp[3], mp[4], mp[5]
            e_r = region_evidence(zr, gt, mode="topk", frac=0.01)
            e_d = region_evidence(zd, gt, mode="topk", frac=0.01)
        p_rgb, p_depth = relative_evidence(e_r, e_d)
        cls = evidence_class(e_r, e_d)
        s_full = variant_scores(fused, None)["max"]
        da = decision_ablation(s_full, variant_scores(zr, None)["max"],
                               variant_scores(zd, None)["max"])
        cap = (f"[{name}]  class={cls}   p_rgb={p_rgb:.3f}   p_depth={p_depth:.3f}\n"
               f"E_rgb={e_r:.2f}   E_depth={e_d:.2f}   region=GT   S_full={s_full:.3f}\n"
               f"Delta_RGB={da['Delta_RGB']:+.3f}   Delta_DEPTH={da['Delta_DEPTH']:+.3f}")
        rgb_small = resize_arr(rgb, H, W) if rgb.shape[:2] != (H, W) else rgb.astype(np.float64)
        vmax = max(float(np.percentile(zr[gt] if gt.any() else zr, 99.5)), 1.0)
        written.append(render_fig(name, rgb_small, gt, zr, zd, fused, cap, vmax))
        print(f"wrote {written[-1]}")

    # Montage.
    n = len(written)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5.4 * rows))
    axes = np.atleast_1d(np.atleast_2d(axes).ravel())
    for ax, pth in zip(axes, written):
        ax.imshow(plt.imread(pth)); ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "operator_explanations_montage.png", dpi=100)
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'operator_explanations_montage.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())