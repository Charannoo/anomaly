"""Phase 6 tests: deployment wrapper contract + post-processing pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from xmvad.deployment.model_wrapper import DeploymentE4, build_deployment_e4
from xmvad.fusion.score_fusion import calibrate
from xmvad.metrics.scores import variant_scores
from xmvad.models.anomaly.teacher_student import TeacherStudentAD
from xmvad.models.fusion.concat import E4ConcatModel


def _model() -> E4ConcatModel:
    rgb = TeacherStudentAD(modality="rgb", pretrained=False).eval()
    depth = TeacherStudentAD(modality="depth", pretrained=False).eval()
    return E4ConcatModel(rgb, depth, proj_dim=32)


def test_deployment_forward_contract_shapes() -> None:
    m = DeploymentE4(_model())
    r = torch.randn(1, 3, 256, 256)
    d = torch.randn(1, 1, 256, 256)
    v = torch.ones(1, 1, 256, 256)
    out = m(r, d, v)
    assert out["A_rgb"].shape == (1, 1, 256, 256)
    assert out["A_depth"].shape == (1, 1, 256, 256)
    assert out["A_rgb"].dtype == torch.float32
    assert out["A_depth"].dtype == torch.float32


def test_deployment_accepts_all_invalid_depth() -> None:
    m = DeploymentE4(_model())
    out = m(torch.randn(1, 3, 128, 128),
            torch.randn(1, 1, 128, 128),
            torch.zeros(1, 1, 128, 128))
    assert out["A_rgb"].shape == (1, 1, 128, 128)


def test_proj_dim_mismatch_refuses_to_guess(tmp_path: pytest.TempPathFactory) -> None:
    cfg = {"model": {"level_weights": [0.5, 0.5], "projection_dim": 32,
                     "smooth_sigma": 0.0, "gamma_init": 0.0}}
    rgb = TeacherStudentAD(modality="rgb", pretrained=False)
    depth = TeacherStudentAD(modality="depth", pretrained=False)
    rgb_ckpt = tmp_path / "rgb.pt"
    depth_ckpt = tmp_path / "depth.pt"
    torch.save({"model_state": rgb.state_dict()}, rgb_ckpt)
    torch.save({"model_state": depth.state_dict()}, depth_ckpt)
    e4 = E4ConcatModel(rgb.eval(), depth.eval(), proj_dim=32)
    fusion_ckpt = tmp_path / "fusion.pt"
    torch.save({"proj_dim": 64, "fusion_state": e4.fusion.state_dict()}, fusion_ckpt)
    with pytest.raises(ValueError, match="proj_dim"):
        build_deployment_e4(cfg, rgb_ckpt, depth_ckpt, fusion_ckpt, "cpu")


def test_proj_dim_match_loads_fusion(tmp_path: pytest.TempPathFactory) -> None:
    cfg = {"model": {"level_weights": [0.5, 0.5], "projection_dim": 32,
                     "smooth_sigma": 0.0, "gamma_init": 0.0}}
    rgb = TeacherStudentAD(modality="rgb", pretrained=False)
    depth = TeacherStudentAD(modality="depth", pretrained=False)
    rgb_ckpt = tmp_path / "rgb.pt"
    depth_ckpt = tmp_path / "depth.pt"
    torch.save({"model_state": rgb.state_dict()}, rgb_ckpt)
    torch.save({"model_state": depth.state_dict()}, depth_ckpt)
    e4 = E4ConcatModel(rgb.eval(), depth.eval(), proj_dim=32)
    fusion_ckpt = tmp_path / "fusion.pt"
    torch.save({"proj_dim": 32, "fusion_state": e4.fusion.state_dict()}, fusion_ckpt)
    m = build_deployment_e4(cfg, rgb_ckpt, depth_ckpt, fusion_ckpt, "cpu")
    out = m(torch.randn(1, 3, 64, 64), torch.randn(1, 1, 64, 64), torch.ones(1, 1, 64, 64))
    assert out["A_rgb"].shape == (1, 1, 64, 64)


def test_postprocess_valid_fallback_keeps_rgb_only() -> None:
    """Invalid depth pixels must fall back to the RGB-only anomaly score."""
    a_r = np.full((1, 1, 64, 64), 2.0, dtype=np.float32)
    a_d = np.full((1, 1, 64, 64), 10.0, dtype=np.float32)
    valid = np.zeros((1, 1, 64, 64), dtype=np.float32)  # every depth sample invalid
    zr = calibrate(a_r, median=1.0, scale=2.0)
    zd = calibrate(a_d, median=1.0, scale=2.0)
    fused = np.where(valid > 0.5, 0.5 * zr + 0.5 * zd, zr)
    assert np.allclose(fused, zr, atol=1e-6)
    score = variant_scores(fused, None)["max"]
    assert score == pytest.approx(float(zr.max()), abs=1e-6)


def test_postprocess_mixed_validity_averages() -> None:
    a_r = np.zeros((1, 1, 4, 4), dtype=np.float32)
    a_d = np.full((1, 1, 4, 4), 4.0, dtype=np.float32)
    valid = np.zeros((1, 1, 4, 4), dtype=np.float32)
    valid[0, 0, :2, :2] = 1.0
    zr = calibrate(a_r, median=0.0, scale=1.0)
    zd = calibrate(a_d, median=0.0, scale=1.0)
    fused = np.where(valid > 0.5, 0.5 * zr + 0.5 * zd, zr)
    assert fused[0, 0, 0, 0] == pytest.approx(0.5 * zr[0, 0, 0, 0] + 0.5 * zd[0, 0, 0, 0])  # 50/50 on valid px
    assert fused[0, 0, 3, 3] == pytest.approx(zr[0, 0, 3, 3])  # pure RGB elsewhere