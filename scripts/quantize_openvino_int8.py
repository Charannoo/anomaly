"""NNCF INT8 PTQ for frozen E4 OpenVINO models (Phase 6, milestone 10).

Calibration uses TRAIN-NORMAL images ONLY (deterministic subset of the
train-normal split; exact source sample IDs written to
deployment/int8_calibration_manifest.json). The PyTorch-reference test
fixtures are used ONLY for parity validation of the quantized model
(INT8 vs FP32 on identical inputs), never for calibration.

Per category:
  1. load deployment/openvino_fp32/<cat>.xml
  2. nncf.quantize(...) with the train-normal calibration set
  3. serialize to deployment/openvino_int8/<cat>.xml|.bin
  4. validation: compiled INT8 vs compiled FP32 output differences plus
     INT8-vs-PyTorch-reference max abs error on all 20 fixtures

Writes deployment/openvino_int8/openvino_int8_manifest.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import openvino as ov
import nncf

from xmvad.data.depth import normalize_depth
from xmvad.data.mvtec3d import MVTec3DDataset
from xmvad.data.stats import fit_train_depth_stats
from xmvad.data.transforms import SynchronizedTransform
from xmvad.utils.config import load_config

CALIB_PER_CATEGORY = 50  # deterministic subset; train-normal only (pre-declared)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/concat_fusion.yaml")
    ap.add_argument("--fp32-dir", default=str(REPO_ROOT / "deployment" / "openvino_fp32"))
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--preset", choices=["performance", "mixed"], default="performance",
                    help="NNCF PTQ preset (symmetric vs asymmetric activations).")
    ap.add_argument("--bias-correction", choices=["true", "false"], default="true",
                    help="NNCF fast bias correction on/off.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.get("data", {}).get("root", "data/raw/mvtec3d"))
    from xmvad.data.mvtec3d import EXPECTED_CATEGORIES
    cats = args.categories or cfg.get("data", {}).get("categories") or list(EXPECTED_CATEGORIES)
    fp32_dir = Path(args.fp32_dir)
    out_dir = REPO_ROOT / "deployment" / "openvino_int8"
    out_dir.mkdir(parents=True, exist_ok=True)
    fixtures = REPO_ROOT / "experiments" / "deployment" / "reference_fixtures"

    input_size = tuple(cfg.get("data", {}).get("input_size", [256, 256]))
    manifest_json: dict = {"fit_on": "train_normal_only", "samples_per_category": CALIB_PER_CATEGORY,
                           "categories": {}, "method": "static PTQ (NNCF, OpenVINO backend), symmetric INT8"}
    core = ov.Core()

    rows = []
    for cat in cats:
        depth_stats = fit_train_depth_stats(root, cat, cfg.get("data", {}).get("depth_norm", "robust_median"),
                                            verbose=False)
        tf = SynchronizedTransform(
            input_size=input_size,
            rgb_mean=tuple(cfg.get("data", {}).get("rgb_mean", [0.485, 0.456, 0.406])),
            rgb_std=tuple(cfg.get("data", {}).get("rgb_std", [0.229, 0.224, 0.225])),
            depth_stats=depth_stats,
            depth_clip=tuple(cfg["data"]["depth_clip"]) if cfg.get("data", {}).get("depth_clip") else None,
            train=False)
        # Deterministic calibration subset: first CALIB_PER_CATEGORY train-normal records.
        from xmvad.data.mvtec3d import discover_split
        normals = [r for r in discover_split(root, cat, "train") if r.label == 0][:CALIB_PER_CATEGORY]
        manifest_json["categories"][cat] = {"split": "train", "defect": "good",
                                            "sample_ids": [r.sample_id for r in normals]}
        ds = MVTec3DDataset(normals, input_size=input_size, transform=tf, depth_stats=depth_stats)

        def calib_gen():
            for i in range(len(ds)):
                item = ds[i]
                yield {"rgb": item["rgb"].numpy()[None], "depth": item["depth"].numpy()[None],
                       "valid": item["valid"].float().numpy()[None].astype(np.float32)}

        fp32_model = core.read_model(str(fp32_dir / f"{cat}.xml"))
        calibration_dataset = nncf.Dataset(calib_gen())
        preset_obj = nncf.QuantizationPreset.MIXED if args.preset == "mixed" else nncf.QuantizationPreset.PERFORMANCE
        quantized = nncf.quantize(
            fp32_model, calibration_dataset, subset_size=CALIB_PER_CATEGORY, preset=preset_obj,
            fast_bias_correction=args.bias_correction == "true")
        xml_path = out_dir / f"{cat}.xml"
        ov.save_model(quantized, str(xml_path), compress_to_fp16=False)

        # Validation: INT8 vs FP32 (compiled) parity on fixtures + vs PyTorch ref.
        q_model = core.read_model(str(xml_path))
        compiled_fp32 = core.compile_model(fp32_model, "CPU")
        compiled_int8 = core.compile_model(q_model, "CPU")
        errs_fp32 = []   # INT8 vs compiled-FP32
        errs_ref = []    # INT8 vs PyTorch reference fixture
        for kind in ("normal", "anomalous"):
            fx = np.load(fixtures / f"{cat}_{kind}.npz")
            inp = {"rgb": fx["rgb"][None], "depth": fx["depth"][None], "valid": fx["valid"][None]}
            o_f = compiled_fp32(inp)
            o_q = compiled_int8(inp)
            errs_fp32.append(float(np.max(np.abs(o_q[0] - o_f[0]))))
            errs_fp32.append(float(np.max(np.abs(o_q[1] - o_f[1]))))
            errs_ref.append(float(np.max(np.abs(o_q[0] - fx["A_rgb"][None]))))
            errs_ref.append(float(np.max(np.abs(o_q[1] - fx["A_depth"][None]))))
        rows.append({"category": cat, "int8_bytes": int((out_dir / f"{cat}.bin").stat().st_size),
                     "parity_int8_vs_fp32_maxerr": float(np.max(errs_fp32)),
                     "parity_int8_vs_pytorch_maxerr": float(np.max(errs_ref))})
        print(f"{cat}: INT8 vs FP32 max|err|={np.max(errs_fp32):.3e} "
              f"INT8 vs PyTorch max|err|={np.max(errs_ref):.3e}", flush=True)

    (REPO_ROOT / "deployment" / "int8_calibration_manifest.json").write_text(
        json.dumps(manifest_json, indent=2), encoding="utf-8")
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / "openvino_int8_manifest.csv", index=False)
    print(f"wrote {len(rows)} INT8 IRs -> {out_dir}")
    print("calibration manifest -> deployment/int8_calibration_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())