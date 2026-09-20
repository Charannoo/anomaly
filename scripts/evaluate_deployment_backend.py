"""Evaluate deployment backends on the frozen E4 protocol (Phase 6, milestone 11).

Runs the SAME protocol as scripts/evaluate_concat.py (calibrated 50/50 mean
with depth-invalid -> RGB fallback, image score = max) on the full test set
for a chosen backend:

  pytorch      : DeploymentE4 on CPU
  onnx         : ONNX Runtime FP32 (CPU)
  openvino_fp32: OpenVINO compiled FP32
  openvino_int8: OpenVINO compiled INT8

Per-category metrics: image AUROC/APRC(max), pixel AUROC/APRC, AUPRO plus
deltas vs the frozen PyTorch FP32 reference (same script) and vs the frozen
native table (E4_concat_metrics_native.csv, for context).

Writes experiments/tables/deployment_{backend}_metrics.csv and
experiments/tables/deployment_int8_vs_pytorch.csv (etc. per backend).
Test anomalies are used ONLY for this final evaluation, never for calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xmvad.data.mvtec3d import EXPECTED_CATEGORIES, MVTec3DDataset, discover_split
from xmvad.data.stats import fit_train_depth_stats
from xmvad.data.transforms import SynchronizedTransform
from xmvad.deployment.model_wrapper import build_deployment_e4
from xmvad.fusion.score_fusion import calibrate
from xmvad.metrics.detection import image_auprc, image_auroc
from xmvad.metrics.localization import pixel_auprc, pixel_auroc
from xmvad.metrics.pro import aupro
from xmvad.metrics.scores import SCORE_VARIANTS, variant_scores
from xmvad.utils.config import load_config


class Backend:
    def __init__(self, name: str) -> None:
        self.name = name
        self._ort = None
        self._ov = None
        self._compiled = None
        self._model = None

    def load(self, cfg: dict, cat: str, onnx_dir: Path, ov_fp32_dir: Path, ov_int8_dir: Path, kind: str) -> None:
        if kind.startswith("openvino"):
            import openvino as ov
            core = self._ov if self._ov is not None else ov.Core()
            self._ov = core
            ir = ov_fp32_dir if kind == "openvino_fp32" else ov_int8_dir
            self._compiled = core.compile_model(str(ir / f"{cat}.xml"), "CPU")
        elif kind == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            so.inter_op_num_threads = 1
            self._ort = ort.InferenceSession(str(onnx_dir / f"{cat}.onnx"),
                                             sess_options=so, providers=["CPUExecutionProvider"])
        else:
            r_ckpt = sorted((REPO_ROOT / "checkpoints" / "E1_baseline_rgb").glob(f"{cat}_best.pt"))[0]
            d_ckpt = sorted((REPO_ROOT / "checkpoints" / "E2_baseline_depth").glob(f"{cat}_best.pt"))[0]
            f_ckpt = sorted((REPO_ROOT / "checkpoints" / "E4_concat_fusion").glob(f"{cat}_best.pt"))[0]
            self._model = build_deployment_e4(cfg, r_ckpt, d_ckpt, f_ckpt, "cpu").eval()

    @torch.no_grad()
    def predict(self, rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rgb_b, depth_b, valid_b = rgb[None], depth[None], valid[None]
        if self._compiled is not None:
            out = self._compiled({"rgb": rgb_b.astype(np.float32), "depth": depth_b.astype(np.float32),
                                  "valid": valid_b.astype(np.float32)})
            return np.asarray(out[0][0, 0]), np.asarray(out[1][0, 0])
        if self._ort is not None:
            out = self._ort.run(None, {"rgb": rgb_b.astype(np.float32), "depth": depth_b.astype(np.float32),
                                       "valid": valid_b.astype(np.float32)})
            return np.asarray(out[0][0, 0]), np.asarray(out[1][0, 0])
        out = self._model(torch.from_numpy(rgb)[None], torch.from_numpy(depth)[None],
                          torch.from_numpy(valid.astype(np.float32))[None])
        return out["A_rgb"][0, 0].numpy(), out["A_depth"][0, 0].numpy()

    def close(self) -> None:
        self._compiled = None
        self._ort = None
        self._model = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/concat_fusion.yaml")
    ap.add_argument("--backend", required=True,
                    choices=["pytorch", "onnx", "openvino_fp32", "openvino_int8"])
    ap.add_argument("--categories", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.get("data", {}).get("root", "data/raw/mvtec3d"))
    cats = args.categories or cfg.get("data", {}).get("categories") or list(EXPECTED_CATEGORIES)
    calib = json.loads((REPO_ROOT / "experiments" / "tables" / "E4_concat_calibration.json").read_text(encoding="utf-8"))
    onnx_dir = REPO_ROOT / "deployment" / "onnx"
    ov_fp = REPO_ROOT / "deployment" / "openvino_fp32"
    ov_i8 = REPO_ROOT / "deployment" / "openvino_int8"
    native = pd.read_csv(REPO_ROOT / "experiments" / "tables" / "E4_concat_metrics_native.csv")
    input_size = tuple(cfg.get("data", {}).get("input_size", [256, 256]))

    torch.set_num_threads(4)
    rows = []
    for cat in cats:
        backend = Backend(args.backend)
        backend.load(cfg, cat, onnx_dir, ov_fp, ov_i8, args.backend)
        depth_stats = fit_train_depth_stats(root, cat, cfg.get("data", {}).get("depth_norm", "robust_median"),
                                            verbose=False)
        tf = SynchronizedTransform(
            input_size=input_size,
            rgb_mean=tuple(cfg.get("data", {}).get("rgb_mean", [0.485, 0.456, 0.406])),
            rgb_std=tuple(cfg.get("data", {}).get("rgb_std", [0.229, 0.224, 0.225])),
            depth_stats=depth_stats,
            depth_clip=tuple(cfg["data"]["depth_clip"]) if cfg.get("data", {}).get("depth_clip") else None,
            train=False)
        recs = discover_split(root, cat, "test")
        ds = MVTec3DDataset(recs, input_size=input_size, transform=tf, depth_stats=depth_stats)
        med_r = calib["categories"][cat]["rgb"]["median"]
        sc_r = calib["categories"][cat]["rgb"]["scale"]
        med_d = calib["categories"][cat]["depth"]["median"]
        sc_d = calib["categories"][cat]["depth"]["scale"]

        labels, gts, fused, fused_max = [], [], [], []
        for i in tqdm(range(len(ds)), desc=cat, leave=False):
            item = ds[i]
            a_r, a_d = backend.predict(item["rgb"].numpy(), item["depth"].numpy(), item["valid"].float().numpy())
            valid = item["valid"][0].numpy().astype(bool)
            zr = calibrate(a_r, med_r, sc_r)
            zd = calibrate(a_d, med_d, sc_d)
            fm = np.where(valid, 0.5 * zr + 0.5 * zd, zr).astype(np.float32)
            fused.append(fm)
            fused_max.append(np.where(valid, np.maximum(zr, zd), zr).astype(np.float32))
            labels.append(item["label"])
            gts.append(item["gt"].numpy().astype(bool))
        backend.close()
        sc = np.array([variant_scores(m, None)["max"] for m in fused])
        sc_max = np.array([variant_scores(m, None)["max"] for m in fused_max])
        row = {"category": cat, "image_auroc_max": image_auroc(labels, sc),
               "image_auprc_max": image_auprc(labels, sc), "pixel_auroc": pixel_auroc(gts, fused),
               "pixel_auprc": pixel_auprc(gts, fused), "aupro": aupro(gts, fused),
               "max_image_auroc": image_auroc(labels, sc_max), "max_pixel_auroc": pixel_auroc(gts, fused_max),
               "num_test_images": len(labels), "num_anomalous": int(sum(labels))}
        rows.append(row)
        print(f"{cat}: I-AUROC={row['image_auroc_max']:.4f} P-AUROC={row['pixel_auroc']:.4f} "
              f"AUPRO={row['aupro']:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(REPO_ROOT / "experiments" / "tables" / f"deployment_{args.backend}_metrics.csv", index=False)

    if args.backend != "pytorch":
        ref = pd.read_csv(REPO_ROOT / "experiments" / "tables" / "deployment_pytorch_metrics.csv")
        merge = []
        for _, r in out.iterrows():
            cat = r["category"]
            fr = ref[ref.category == cat].iloc[0]
            nr = native[native.category == cat].iloc[0]
            merge.append({"category": cat, "backend": args.backend,
                          "d_I_AUROC": round(r["image_auroc_max"] - fr["image_auroc_max"], 5),
                          "d_I_AUPRC": round(r["image_auprc_max"] - fr["image_auprc_max"], 5),
                          "d_P_AUROC": round(r["pixel_auroc"] - fr["pixel_auroc"], 5),
                          "d_P_AUPRC": round(r["pixel_auprc"] - fr["pixel_auprc"], 5),
                          "d_AUPRO": round(r["aupro"] - fr["aupro"], 5),
                          "d_vs_native_I_AUROC": round(r["image_auroc_max"] - nr["image_auroc_max"], 5),
                          "d_vs_native_P_AUROC": round(r["pixel_auroc"] - nr["pixel_auroc"], 5),
                          "d_vs_native_AUPRO": round(r["aupro"] - nr["aupro"], 5)})
        pd.DataFrame(merge).to_csv(REPO_ROOT / "experiments" / "tables"
                                   / f"deployment_{args.backend}_vs_pytorch.csv", index=False)
        print(f"wrote {args.backend} deltas (vs pytorch + vs frozen native)")
    print(f"wrote deployment_{args.backend}_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())