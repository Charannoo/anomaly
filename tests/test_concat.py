"""Phase 4A tests: frozen branches, identity control, fusion shapes."""

from __future__ import annotations

import torch

from xmvad.models.anomaly.teacher_student import TeacherStudentAD
from xmvad.models.fusion.concat import IN_CH, ConcatFusion, E4ConcatModel


def _branches():
    rgb = TeacherStudentAD(modality="rgb", pretrained=False)
    depth = TeacherStudentAD(modality="depth", pretrained=False)
    return rgb.eval(), depth.eval()


def test_only_fusion_params_trainable() -> None:
    rgb, depth = _branches()
    m = E4ConcatModel(rgb, depth)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable, "fusion must have trainable parameters"
    assert all(n.startswith("fusion.") for n in trainable), trainable
    # Branch parameters all frozen.
    assert all(not p.requires_grad for p in m.rgb.parameters())
    assert all(not p.requires_grad for p in m.depth.parameters())


def test_fusion_module_shapes() -> None:
    f = ConcatFusion(proj_dim=32, gamma_init=0.0)
    s8 = torch.randn(2, IN_CH, 32, 32)
    sd = torch.randn(2, IN_CH, 32, 32)
    fr, fd, dr, dd = f(s8, sd)
    assert fr.shape == s8.shape and fd.shape == sd.shape
    assert dr.shape == s8.shape and dd.shape == sd.shape
    assert f.count_parameters() < 25000
    assert float(f.gamma_rgb.detach()) == 0.0 and float(f.gamma_depth.detach()) == 0.0


def test_identity_gamma_zero_reproduces_phase2() -> None:
    """gamma=0 MUST reproduce frozen Phase 2 modality outputs (tight tol)."""
    torch.manual_seed(0)
    rgb, depth = _branches()
    m = E4ConcatModel(rgb, depth, gamma_init=0.0).eval()
    x_rgb = torch.randn(1, 3, 64, 64)
    x_depth = torch.randn(1, 1, 64, 64)
    valid = torch.ones(1, 1, 64, 64, dtype=torch.bool)
    valid[:, :, :16, :] = False
    out = m.predict(x_rgb, x_depth, valid)
    with torch.no_grad():
        pr = rgb.predict(x_rgb, None)
        pd = depth.predict(x_depth, valid)
    assert torch.equal(out["rgb_map"], pr["anomaly_map"])
    assert torch.equal(out["depth_map"], pd["anomaly_map"])


def test_fusion_receives_gradients_branches_do_not() -> None:
    torch.manual_seed(0)
    rgb, depth = _branches()
    m = E4ConcatModel(rgb, depth).train()
    x_rgb = torch.randn(2, 3, 64, 64)
    x_depth = torch.randn(2, 1, 64, 64)
    valid = torch.ones(2, 1, 64, 64, dtype=torch.bool)
    m.forward_train(x_rgb, x_depth, valid)["loss"].backward()
    assert all(p.grad is None for p in m.rgb.parameters())
    assert all(p.grad is None for p in m.depth.parameters())
    grads = [p.grad for p in m.fusion.parameters()]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_gamma_updates_and_residual_shapes() -> None:
    torch.manual_seed(0)
    rgb, depth = _branches()
    m = E4ConcatModel(rgb, depth, gamma_init=0.01)
    x_rgb = torch.randn(2, 3, 64, 64)
    x_depth = torch.randn(2, 1, 64, 64)
    out = m.forward_16(x_rgb, x_depth, None)
    assert out["d_rgb"].shape == (2, IN_CH, 4, 4)
    assert out["d_depth"].shape == (2, IN_CH, 4, 4)
    assert torch.isfinite(out["f_rgb16"]).all()


def test_e4_save_load_consistency(tmp_path) -> None:
    rgb, depth = _branches()
    m = E4ConcatModel(rgb, depth)
    p = tmp_path / "fusion.pt"
    torch.save({"fusion_state": m.fusion.state_dict()}, p)
    m2 = E4ConcatModel(*_branches())
    m2.fusion.load_state_dict(torch.load(p, weights_only=True)["fusion_state"])
    for a, b in zip(m.fusion.state_dict().values(), m2.fusion.state_dict().values()):
        assert torch.equal(a, b)


def test_no_test_split_in_concat_training_script() -> None:
    from pathlib import Path

    src = open(Path(__file__).resolve().parents[1] / "scripts" / "train_concat_fusion.py",
               encoding="utf-8").read()
    assert '"train"' in src and '"validation"' in src
    assert '"test"' not in src and "'test'" not in src
