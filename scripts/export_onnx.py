"""Export frozen E4 deployment models to ONNX (Phase 6, milestone 6).

Per category: build DeploymentE4 on CPU, export with fixed batch-1 256x256
inputs (opset 17), run onnx.checker + shape inference, then immediately
verify against the PyTorch reference fixture (max abs error on A_rgb/A_depth)
via onnxruntime CPU. Writes deployment/onnx/<cat>.onnx plus a per-export
manifest CSV. No test-set tuning; conversion time is NOT part of latency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import onnx
import onnx.shape_inference
import torch

from xmvad.deployment.model_wrapper import build_deployment_e4
from xmvad.utils.config import load_config

OPSET = 17


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/concat_fusion.yaml")
    ap.add_argument("--rgb-checkpoint", default="checkpoints/E1_baseline_rgb")
    ap.add_argument("--depth-checkpoint", default="checkpoints/E2_baseline_depth")
    ap.add_argument("--fusion-checkpoint", default="checkpoints/E4_concat_fusion")
    ap.add_argument("--categories", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    from xmvad.data.mvtec3d import EXPECTED_CATEGORIES
    cats = args.categories or cfg.get("data", {}).get("categories") or list(EXPECTED_CATEGORIES)

    out_dir = REPO_ROOT / "deployment" / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    fixtures = REPO_ROOT / "experiments" / "deployment" / "reference_fixtures"

    manifest = []
    torch.set_num_threads(1)
    for cat in cats:
        r_ckpt = sorted(Path(args.rgb_checkpoint).glob(f"{cat}_best.pt"))[0]
        d_ckpt = sorted(Path(args.depth_checkpoint).glob(f"{cat}_best.pt"))[0]
        f_ckpt = sorted(Path(args.fusion_checkpoint).glob(f"{cat}_best.pt"))[0]
        model = build_deployment_e4(cfg, r_ckpt, d_ckpt, f_ckpt, "cpu").eval()
        model.eval()

        fx = np.load(fixtures / f"{cat}_normal.npz")
        rgb = torch.from_numpy(fx["rgb"])[None]
        depth = torch.from_numpy(fx["depth"])[None]
        valid = torch.from_numpy(fx["valid"])[None]

        dst = out_dir / f"{cat}.onnx"
        torch.onnx.export(
            model, (rgb, depth, valid), str(dst),
            input_names=["rgb", "depth", "valid"],
            output_names=["A_rgb", "A_depth"],
            opset_version=OPSET,
            do_constant_folding=True,
        )
        m = onnx.load(str(dst))
        onnx.checker.check_model(m)
        m2 = onnx.shape_inference.infer_shapes(m)
        onnx.checker.check_model(m2)
        onnx.save(m2, str(dst))

        # Parity pre-check with onnxruntime CPU against the reference fixture.
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(dst), sess_options=so, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"rgb": rgb.numpy(), "depth": depth.numpy(), "valid": valid.numpy()})
        a_rgb_ref = fx["A_rgb"][None]
        a_depth_ref = fx["A_depth"][None]
        err_rgb = float(np.max(np.abs(out[0] - a_rgb_ref)))
        err_depth = float(np.max(np.abs(out[1] - a_depth_ref)))
        manifest.append({"category": cat, "opset": int(m2.opset_import[0].version),
                         "onnx_bytes": int(dst.stat().st_size),
                         "err_max_A_rgb": err_rgb, "err_max_A_depth": err_depth})
        print(f"{cat}: onnx={dst.stat().st_size} B, max|err| A_rgb={err_rgb:.3e} A_depth={err_depth:.3e}",
              flush=True)

    import pandas as pd
    pd.DataFrame(manifest).to_csv(out_dir / "onnx_export_manifest.csv", index=False)
    print(f"wrote {len(manifest)} ONNX models -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())