"""E3B normal-state reliability statistics (validation-normal ONLY).

For each category x modality, runs the FROZEN Phase 2 baselines over the
validation-normal split, calibrates maps with the FROZEN E3 calibration, and
records noise/coverage statistics. No test sample is touched. No retraining.

Noise definition (single, explicit, fixed before any test evaluation):
    noise_m = mean over validation images of
              (fraction of evidence pixels with calibrated Z > 3)
where evidence pixels = all pixels (RGB) or valid pixels (depth).
Static base weights: base_m = 1/(noise_m + eps), normalized to sum to one.

Writes experiments/calibration/E3B_reliability_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xmvad.data.mvtec3d import MVTec3DDataset, discover_categories, discover_split
from xmvad.data.stats import fit_train_depth_stats
from xmvad.data.transforms import SynchronizedTransform
from xmvad.fusion.score_fusion import EPS, calibrate, static_weights
from xmvad.losses.distillation import downsample_valid_mask
from xmvad.models.anomaly.teacher_student import TeacherStudentAD
from xmvad.training.checkpoints import load_checkpoint
from xmvad.utils.config import load_config

Z_THRESHOLDS = (2.0, 3.0, 5.0)
NOISE_Z = 3.0
PEAK_Z = 5.0


def load_model(ckpt_path: Path, cfg: dict, device: torch.device) -> TeacherStudentAD:
    model_cfg = cfg.get("model", {})
    probe = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    modality = probe.get("modality") or model_cfg.get("modality", "rgb")
    model = TeacherStudentAD(
        modality=modality,
        level_weights=tuple(model_cfg.get("level_weights", [0.5, 0.5])),  # type: ignore[arg-type]
        pretrained=False,
        score_mode=model_cfg.get("score_mode", "max"),
        topk_frac=float(model_cfg.get("topk_frac", 0.01)),
        smooth_sigma=float(model_cfg.get("smooth_sigma", 0.0)),
    )
    load_checkpoint(ckpt_path, model)
    return model.to(device).eval()


def topk_mean(flat: np.ndarray, frac: float) -> float:
    k = max(1, int(round(frac * flat.size)))
    return float(np.sort(flat)[-k:].mean())


@torch.no_grad()
def modality_stats(cfg: dict, ckpt: Path, category: str, median: float, scale: float,
                   device: torch.device) -> dict:
    data_cfg = cfg.get("data", {})
    root = Path(data_cfg.get("root", "data/raw/mvtec3d"))
    input_size = tuple(data_cfg.get("input_size", [256, 256]))
    modality = cfg.get("model", {}).get("modality", "rgb")
    depth_stats = None
    if modality == "depth":
        depth_stats = fit_train_depth_stats(root, category, data_cfg.get("depth_norm", "robust_median"),
                                            verbose=False)
    tf = SynchronizedTransform(
        input_size=input_size,
        rgb_mean=tuple(data_cfg.get("rgb_mean", [0.485, 0.456, 0.406])),
        rgb_std=tuple(data_cfg.get("rgb_std", [0.229, 0.224, 0.225])),
        depth_stats=depth_stats,
        depth_clip=tuple(data_cfg["depth_clip"]) if data_cfg.get("depth_clip") else None,
        train=False,
    )
    recs = [r for r in discover_split(root, category, "validation") if r.label == 0]
    ds = MVTec3DDataset(recs, input_size=input_size, transform=tf, depth_stats=depth_stats)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    model = load_model(ckpt, cfg, device)
    key = "rgb" if modality == "rgb" else "depth"
    pooled, peak_flags, frac_above, t01, t1, valid_fracs = [], [], {z: [] for z in Z_THRESHOLDS}, [], [], []
    s8_fracs, s16_fracs = [], []
    for batch in tqdm(loader, desc=f"rel {category}/{modality}", leave=False):
        x = batch[key].to(device)
        valid = batch["valid"].to(device) if modality == "depth" else None
        am = model.predict(x, valid)["anomaly_map"].cpu().numpy()
        for i in range(x.shape[0]):
            m = am[i, 0].astype(np.float64)
            z = calibrate(m, median, scale)
            if modality == "depth":
                v = batch["valid"][i, 0].numpy().astype(bool)
                ev = z[v]
                valid_fracs.append(float(v.mean()))
                t = torch.from_numpy(v).unsqueeze(0).unsqueeze(0)
                s8_fracs.append(float(downsample_valid_mask(t, (z.shape[0] // 8, z.shape[1] // 8)).float().mean()))
                s16_fracs.append(float(downsample_valid_mask(t, (z.shape[0] // 16, z.shape[1] // 16)).float().mean()))
            else:
                ev = z.ravel()
            pooled.append(ev.ravel())
            peak_flags.append(bool(np.max(ev) > PEAK_Z) if ev.size else False)
            for zt in Z_THRESHOLDS:
                frac_above[zt].append(float((ev > zt).mean()) if ev.size else 0.0)
            t01.append(topk_mean(ev, 0.001))
            t1.append(topk_mean(ev, 0.01))
    pooled_all = np.concatenate(pooled)
    stats = {
        "n_images": len(recs), "n_pixels": int(pooled_all.size),
        "median_cal": float(np.median(pooled_all)),
        "mad_cal": float(np.median(np.abs(pooled_all - np.median(pooled_all)))),
        "p95_cal": float(np.percentile(pooled_all, 95)),
        "p99_cal": float(np.percentile(pooled_all, 99)),
        "extreme_peak_freq": float(np.mean(peak_flags)),
        "spatial_frac_above": {str(zt): float(np.mean(frac_above[zt])) for zt in Z_THRESHOLDS},
        "noise": float(np.mean(frac_above[NOISE_Z])),
        "mean_top01": float(np.mean(t01)),
        "mean_top1": float(np.mean(t1)),
    }
    if modality == "depth":
        stats.update({"valid_frac_mean": float(np.mean(valid_fracs)),
                      "stage8_valid_mean": float(np.mean(s8_fracs)),
                      "stage16_valid_mean": float(np.mean(s16_fracs))})
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-config", default="configs/baseline_rgb.yaml")
    ap.add_argument("--depth-config", default="configs/baseline_depth.yaml")
    ap.add_argument("--rgb-checkpoint", default="checkpoints/E1_baseline_rgb")
    ap.add_argument("--depth-checkpoint", default="checkpoints/E2_baseline_depth")
    ap.add_argument("--calibration", default="experiments/calibration/E3_late_fusion_calibration.json")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="experiments/calibration/E3B_reliability_stats.json")
    args = ap.parse_args()

    rgb_cfg = load_config(args.rgb_config)
    depth_cfg = load_config(args.depth_config)
    with open(REPO_ROOT / args.calibration, encoding="utf-8") as f:
        calib = json.load(f)
    root = Path(rgb_cfg.get("data", {}).get("root", "data/raw/mvtec3d"))
    cats = args.categories or depth_cfg.get("data", {}).get("categories") or discover_categories(str(root))
    device = torch.device(args.device)

    def resolve(ckpt_dir: str, cat: str) -> Path:
        cands = sorted(Path(ckpt_dir).glob(f"{cat}_best.pt"))
        if not cands:
            raise FileNotFoundError(f"No checkpoint for {cat} in {ckpt_dir}")
        return cands[0]

    out: dict = {"fit_on": "validation_normal_only",
                 "noise_definition": f"mean spatial fraction of evidence pixels with Z > {NOISE_Z}",
                 "extreme_peak_definition": f"fraction of images with max Z > {PEAK_Z}",
                 "static_weight_rule": "base_m = 1/(noise_m + eps), normalized",
                 "categories": {}}
    for cat in cats:
        print(f"== {cat} ==")
        cm = calib["categories"][cat]
        r = modality_stats(rgb_cfg, resolve(args.rgb_checkpoint, cat), cat,
                           cm["rgb"]["median"], cm["rgb"]["scale"], device)
        d = modality_stats(depth_cfg, resolve(args.depth_checkpoint, cat), cat,
                           cm["depth"]["median"], cm["depth"]["scale"], device)
        w_r, w_d = static_weights(r["noise"], d["noise"])
        print(f"  noise rgb={r['noise']:.4f} depth={d['noise']:.4f} -> w=({w_r:.3f},{w_d:.3f})")
        out["categories"][cat] = {"rgb": r, "depth": d,
                                  "static_weights": {"w_rgb": w_r, "w_depth": w_d}}
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
