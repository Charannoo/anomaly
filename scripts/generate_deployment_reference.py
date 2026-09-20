"""Generate PyTorch FP32 deployment reference fixtures (Phase 6).

For each of the 10 frozen E4 categories, takes exactly 2 deterministic test
samples (first normal, first anomalous) and records:
  inputs  : rgb, depth, valid  (the exact tensors sent to the exported graphs)
  outputs : A_rgb, A_depth (raw model maps), A_final (calibrated 50/50 mean
            with depth-invalid -> RGB fallback), score (max over A_final)

These fixtures are the parity baseline: exported ONNX / OpenVINO graphs must
reproduce A_rgb / A_depth within the pre-declared tolerance, and the
post-processing must reproduce A_final / score exactly (same NumPy ops).

Writes experiments/deployment/reference_fixtures/<cat>_{normal,anomalous}.npz
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

from xmvad.data.mvtec3d import MVTec3DDataset, discover_split
from xmvad.data.stats import fit_train_depth_stats
from xmvad.data.transforms import SynchronizedTransform
from xmvad.deployment.model_wrapper import build_deployment_e4
from xmvad.fusion.score_fusion import calibrate
from xmvad.metrics.scores import SCORE_VARIANTS, variant_scores
from xmvad.utils.config import load_config


def robust_params(values: np.ndarray):
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sc = mad * 1.4826
    if not np.isfinite(sc) or sc <= 1e-12:
        sc = float(np.std(v))
    if not np.isfinite(sc) or sc <= 1e-12:
        sc = 1.0
    return med, float(sc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/concat_fusion.yaml")
    ap.add_argument("--rgb-checkpoint", default="checkpoints/E1_baseline_rgb")
    ap.add_argument("--depth-checkpoint", default="checkpoints/E2_baseline_depth")
    ap.add_argument("--fusion-checkpoint", default="checkpoints/E4_concat_fusion")
    ap.add_argument("--categories", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.get("data", {}).get("root", "data/raw/mvtec3d"))
    from xmvad.data.mvtec3d import EXPECTED_CATEGORIES, discover_categories
    cats = args.categories or cfg.get("data", {}).get("categories") or sorted(EXPECTED_CATEGORIES)
    calib = json.loads((REPO_ROOT / "experiments" / "tables" / "E4_concat_calibration.json").read_text(encoding="utf-8"))
    if calib.get("categories", {}).keys() != set(map(str, cats)):
        cats = [c for c in cats if c in calib.get("categories", {})]

    input_size = tuple(cfg.get("data", {}).get("input_size", [256, 256]))
    out_dir = REPO_ROOT / "experiments" / "deployment" / "reference_fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    torch.set_num_threads(1)
    for cat in cats:
        depth_stats = fit_train_depth_stats(root, cat, cfg.get("data", {}).get("depth_norm", "robust_median"), verbose=False)
        r_ckpt = sorted(Path(args.rgb_checkpoint).glob(f"{cat}_best.pt"))[0]
        d_ckpt = sorted(Path(args.depth_checkpoint).glob(f"{cat}_best.pt"))[0]
        f_ckpt = sorted(Path(args.fusion_checkpoint).glob(f"{cat}_best.pt"))[0]
        model = build_deployment_e4(cfg, r_ckpt, d_ckpt, f_ckpt, "cpu")
        model.eval()

        tf = SynchronizedTransform(
            input_size=input_size,
            rgb_mean=tuple(cfg.get("data", {}).get("rgb_mean", [0.485, 0.456, 0.406])),
            rgb_std=tuple(cfg.get("data", {}).get("rgb_std", [0.229, 0.224, 0.225])),
            depth_stats=depth_stats,
            depth_clip=tuple(cfg["data"]["depth_clip"]) if cfg.get("data", {}).get("depth_clip") else None,
            train=False)
        recs = discover_split(root, cat, "test")
        normals = [r for r in recs if r.label == 0]
        anomalies = [r for r in recs if r.label == 1]
        chosen = [("normal", normals[0]), ("anomalous", anomalies[0])]
        ds = MVTec3DDataset([r for _, r in chosen], input_size=input_size, transform=tf, depth_stats=depth_stats)

        med_r = calib["categories"][cat]["rgb"]["median"]
        sc_r = calib["categories"][cat]["rgb"]["scale"]
        med_d = calib["categories"][cat]["depth"]["median"]
        sc_d = calib["categories"][cat]["depth"]["scale"]

        for kind, _rec in chosen:
            item = ds[0 if kind == "normal" else 1]
            with torch.no_grad():
                out = model(item["rgb"][None], item["depth"][None], item["valid"].float()[None])
            a_rgb = out["A_rgb"][0, 0].numpy().astype(np.float64)
            a_depth = out["A_depth"][0, 0].numpy().astype(np.float64)
            valid = item["valid"][0].numpy().astype(bool)
            zr = calibrate(a_rgb, med_r, sc_r)
            zd = calibrate(a_depth, med_d, sc_d)
            a_final = np.where(valid, 0.5 * zr + 0.5 * zd, zr).astype(np.float32)
            score = variant_scores(a_final, None)["max"]
            np.savez_compressed(
                out_dir / f"{cat}_{kind}.npz",
                rgb=item["rgb"].numpy(), depth=item["depth"].numpy(),
                valid=item["valid"].numpy().astype(np.float32),
                A_rgb=out["A_rgb"][0, 0].numpy().astype(np.float32),
                A_depth=out["A_depth"][0, 0].numpy().astype(np.float32),
                A_final=a_final, score=np.float32(score), sample_id=str(item["sample_id"]))
            rows.append({"category": cat, "kind": kind, "sample_id": item["sample_id"],
                         "score": round(float(score), 6),
                         "A_rgb_min": float(a_rgb.min()), "A_rgb_max": float(a_rgb.max()),
                         "A_depth_max": float(a_depth.max())})
            print(f"{cat} {kind}: id={item['sample_id']} score={score:.6f}", flush=True)
    print(f"wrote {len(rows)} reference fixtures -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())