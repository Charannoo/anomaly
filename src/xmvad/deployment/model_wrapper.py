"""Export-safe deployment wrapper for the frozen E4 model (Phase 6).

Wraps ``E4ConcatModel`` (RGB+Depth frozen Phase 2 branches + ConcatFusion)
so the exported artifacts only contain traceable tensor ops:

    inputs:  rgb   float32 [B,3,H,W]  ImageNet-normalized
             depth float32 [B,1,H,W]  z-normalized (robust), invalid = 0
             valid float32 [B,1,H,W]  1.0 = valid depth sample, else 0.0
    outputs: A_rgb   float32 [B,1,H,W]  raw RGB anomaly map (uncalibrated)
             A_depth float32 [B,1,H,W]  raw depth anomaly map (uncalibrated)

Batch 1 is the primary target; H=W=256 fixed. Calibration, reliability
fusion and image-score computation are POST-PROCESSING kept outside this
wrapper/graph (see docs/DEPLOYMENT_INPUT_CONTRACT.md).

The wrapper deliberately does not write files, create tensors whose shapes
depend on python values, or branch on data; it is exported with
``torch.onnx.export`` tracing in the ONNX export script.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from xmvad.models.anomaly.teacher_student import TeacherStudentAD
from xmvad.models.fusion.concat import E4ConcatModel
from xmvad.training.checkpoints import load_checkpoint


class DeploymentE4(nn.Module):
    """Call-compatible, export-safe forward mirror of E4ConcatModel.predict."""

    def __init__(self, model: E4ConcatModel) -> None:
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        v = valid > 0.5
        out = self.model.predict(rgb, depth, v, out_size=(rgb.shape[-2], rgb.shape[-1]))
        return {"A_rgb": out["rgb_map"], "A_depth": out["depth_map"]}


def build_deployment_e4(
    cfg: dict,
    rgb_ckpt: Path,
    depth_ckpt: Path,
    fusion_ckpt: Path | None = None,
    device: str | torch.device = "cpu",
) -> DeploymentE4:
    """Rebuild the frozen E4 deployment model from config + checkpoints.

    Mirrors scripts/evaluate_concat.py::build_e4 exactly (same construction
    order, same hyperparameters from the config). Checkpoints are loaded on
    CPU and the whole model moved to ``device`` in eval mode.
    """
    mc = cfg.get("model", {})

    def branch(modality: str, ckpt: Path) -> TeacherStudentAD:
        m = TeacherStudentAD(
            modality=modality,
            level_weights=tuple(mc.get("level_weights", [0.5, 0.5])),
            pretrained=False,
            score_mode="max",
            topk_frac=0.01,
            smooth_sigma=float(mc.get("smooth_sigma", 0.0)),
        )
        load_checkpoint(ckpt, m)
        return m.eval()

    e4 = E4ConcatModel(
        branch("rgb", rgb_ckpt),
        branch("depth", depth_ckpt),
        proj_dim=int(mc.get("projection_dim", 32)),
        gamma_init=float(mc.get("gamma_init", 0.0)),
        smooth_sigma=float(mc.get("smooth_sigma", 0.0)),
    )
    if fusion_ckpt is not None:
        fz = torch.load(str(fusion_ckpt), map_location="cpu", weights_only=False)
        ckpt_proj = int(fz.get("proj_dim", mc.get("projection_dim", 32)))
        cfg_proj = int(mc.get("projection_dim", 32))
        if ckpt_proj != cfg_proj:
            raise ValueError(
                f"checkpoint proj_dim ({ckpt_proj}) != config projection_dim ({cfg_proj}); "
                "frozen artifact mismatch, refusing to guess."
            )
        e4.fusion.load_state_dict(fz["fusion_state"])
    dev = torch.device(device)
    return DeploymentE4(e4.to(dev).eval())