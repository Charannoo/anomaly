"""E3 calibration on NORMAL VALIDATION data only (Phase 3A).

For each category x modality, runs the FROZEN Phase 2 baselines over the
validation-normal split and fits robust normal-calibration parameters from
the inference anomaly maps:

    median_m = median(pooled map values)
    scale_m  = MAD * 1.4826 (fallback: std, then 1.0)

Depth pools VALID pixels only (invalid pixels carry no evidence).
Also records per-image max-score distributions (for validation-calibrated
qualitative thresholds) and fused-mean validation maps/scores.

Writes experiments/calibration/E3_late_fusion_calibration.json.
No test sample is touched. No retraining occurs.
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
from xmvad.models.anomaly.teacher_student import TeacherStudentAD
from xmvad.training.checkpoints import load_checkpoint
from xmvad.utils.config import load_config


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


def robust_scale(values: np.ndarray) -> tuple[float, float]:
    """Return (median, MAD*1.4826 with std/1.0 fallbacks)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("No finite values for calibration.")
    median = float(np.median(v))
    mad = float(np.median(np.abs(v - median)))
    scale = mad * 1.4826
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(v))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return median, float(scale)


@torch.no_grad()
def collect_validation(cfg: dict, ckpt: Path, category: str, device: torch.device):
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
    if not recs:
        raise RuntimeError(f"No validation-normal records for {category}.")
    ds = MVTec3DDataset(recs, input_size=input_size, transform=tf, depth_stats=depth_stats)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    model = load_model(ckpt, cfg, device)
    key = "rgb" if modality == "rgb" else "depth"
    pooled: list[np.ndarray] = []
    max_scores: list[float] = []
    maps: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    for batch in tqdm(loader, desc=f"cal {category}/{modality}", leave=False):
        x = batch[key].to(device)
        valid = batch["valid"].to(device) if modality == "depth" else None
        out = model.predict(x, valid)
        am = out["anomaly_map"].cpu().numpy()
        for i in range(x.shape[0]):
            m = am[i, 0].astype(np.float64)
            maps.append(m)
            if modality == "depth":
                v = batch["valid"][i, 0].numpy().astype(bool)
                valids.append(v)
                pooled.append(m[v].ravel())
            else:
                valids.append(None)  # type: ignore[arg-type]
                pooled.append(m.ravel())
            finite = m[np.isfinite(m)] if modality == "rgb" else m[v]
            max_scores.append(float(finite.max()) if finite.size else 0.0)
    pooled_all = np.concatenate(pooled)
    median, scale = robust_scale(pooled_all)
    return {
        "median": median, "scale": scale,
        "n_images": len(recs), "n_pixels": int(pooled_all.size),
        "p99": float(np.percentile(pooled_all, 99)),
        "max_scores": max_scores,  # kept in memory only (thresholds below)
        "maps": maps, "valids": valids,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-config", default="configs/baseline_rgb.yaml")
    ap.add_argument("--depth-config", default="configs/baseline_depth.yaml")
    ap.add_argument("--rgb-checkpoint", default="checkpoints/E1_baseline_rgb")
    ap.add_argument("--depth-checkpoint", default="checkpoints/E2_baseline_depth")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="experiments/calibration/E3_late_fusion_calibration.json")
    args = ap.parse_args()

    rgb_cfg = load_config(args.rgb_config)
    depth_cfg = load_config(args.depth_config)
    root = Path(rgb_cfg.get("data", {}).get("root", "data/raw/mvtec3d"))
    cats = args.categories or depth_cfg.get("data", {}).get("categories") or discover_categories(str(root))
    device = torch.device(args.device)

    def resolve(ckpt_dir: str, cat: str) -> Path:
        cands = sorted(Path(ckpt_dir).glob(f"{cat}_best.pt"))
        if not cands:
            raise FileNotFoundError(f"No checkpoint for {cat} in {ckpt_dir}")
        return cands[0]

    out: dict = {"method": "robust_median_mad", "fit_on": "validation_normal_only",
                 "note": "Depth pools valid pixels only. No test data used.",
                 "eval_resolution": list(rgb_cfg.get("data", {}).get("input_size", [256, 256])),
                 "categories": {}}
    for cat in cats:
        print(f"== {cat} ==")
        r = collect_validation(rgb_cfg, resolve(args.rgb_checkpoint, cat), cat, device)
        d = collect_validation(depth_cfg, resolve(args.depth_checkpoint, cat), cat, device)
        # Fused-mean validation maps (same rule as E3 inference).
        fused_max = []
        for m_r, m_d, v in zip(r["maps"], d["maps"], d["valids"]):
            zr = (m_r - r["median"]) / r["scale"]
            zd = (m_d - d["median"]) / d["scale"]
            fused = np.where(v, 0.5 * zr + 0.5 * zd, zr)
            fused_max.append(float(np.max(fused[np.isfinite(fused)])))
        out["categories"][cat] = {
            "rgb": {k: r[k] for k in ("median", "scale", "n_images", "n_pixels", "p99")},
            "depth": {k: d[k] for k in ("median", "scale", "n_images", "n_pixels", "p99")},
            "score_thresholds": {
                "rgb_max_p95": float(np.percentile(r["max_scores"], 95)),
                "depth_max_p95": float(np.percentile(d["max_scores"], 95)),
                "fused_max_p95": float(np.percentile(fused_max, 95)),
            },
        }
        print(f"  rgb median={r['median']:.4f} scale={r['scale']:.4f} | "
              f"depth median={d['median']:.4f} scale={d['scale']:.4f}")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
