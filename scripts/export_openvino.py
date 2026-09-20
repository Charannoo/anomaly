"""Export frozen E4 ONNX models to OpenVINO FP32 IR (Phase 6, milestone 9).

Per category: ov.convert_model(deployment/onnx/<cat>.onnx), serialize to
deployment/openvino_fp32/<cat>.xml|.bin, then parity-check the compiled CPU
model against the PyTorch reference fixture (max abs error on A_rgb/A_depth).
Writes deployment/openvino_fp32/openvino_fp32_manifest.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import openvino as ov

from xmvad.utils.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default=str(REPO_ROOT / "deployment" / "onnx"))
    ap.add_argument("--categories", nargs="*", default=None)
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir)
    out_dir = REPO_ROOT / "deployment" / "openvino_fp32"
    out_dir.mkdir(parents=True, exist_ok=True)
    fixtures = REPO_ROOT / "experiments" / "deployment" / "reference_fixtures"

    cats = args.categories or [p.name[:-5] for p in sorted(onnx_dir.glob("*.onnx"))]
    core = ov.Core()
    manifest = []
    for cat in cats:
        src = onnx_dir / f"{cat}.onnx"
        model = ov.convert_model(str(src))
        xml_path = out_dir / f"{cat}.xml"
        ov.save_model(model, str(xml_path))
        compiled = core.compile_model(model, "CPU")

        fx = np.load(fixtures / f"{cat}_normal.npz")
        inp = {"rgb": fx["rgb"][None], "depth": fx["depth"][None], "valid": fx["valid"][None]}
        out = compiled(inp)
        a_rgb = out[compiled.output(0)]
        a_depth = out[compiled.output(1)]
        err_rgb = float(np.max(np.abs(a_rgb - fx["A_rgb"][None])))
        err_depth = float(np.max(np.abs(a_depth - fx["A_depth"][None])))
        manifest.append({"category": cat, "ir_bytes": int((out_dir / f"{cat}.bin").stat().st_size),
                         "err_max_A_rgb": err_rgb, "err_max_A_depth": err_depth})
        print(f"{cat}: parity err_A_rgb={err_rgb:.3e} err_A_depth={err_depth:.3e}", flush=True)

    import pandas as pd
    pd.DataFrame(manifest).to_csv(out_dir / "openvino_fp32_manifest.csv", index=False)
    print(f"wrote {len(manifest)} OpenVINO FP32 IRs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())