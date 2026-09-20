"""Phase 4B XMV cross-attention tests (E5/E6/E7).

Covers the full mandatory list: shapes, two-head split/merge, the three mode
combinations, validity masking (keys redirection + queries), all-invalid
fallback, NaN safety, gamma=0 identity vs frozen Phase 2, branch freezing,
gradient routing (active vs direction-disabled params), deterministic init,
checkpoint save/load, map output sizes, and the no-test-in-training source
guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xmvad.models.anomaly.teacher_student import TeacherStudentAD
from xmvad.models.fusion.cross_attention import (
    HEAD_DIM,
    IN_CH,
    NUM_HEADS,
    XMVAttention,
    XMVModel,
)


def _branches():
    torch.manual_seed(0)
    rgb = TeacherStudentAD(modality="rgb", pretrained=False)
    depth = TeacherStudentAD(modality="depth", pretrained=False)
    return rgb.eval(), depth.eval()


def _feats(n: int, hw: tuple[int, int] = (16, 16)):
    torch.manual_seed(1)
    return torch.randn(n, IN_CH, *hw), torch.randn(n, IN_CH, *hw)


# -- shapes + head mechanics -------------------------------------------

def test_attention_shapes() -> None:
    a = XMVAttention(mode="bidirectional")
    s, _ = _feats(2)
    v = torch.ones(2, 1, 16, 16, dtype=torch.bool)
    fr, fd, dr, dd = a(s, s, v)
    assert fr.shape == s.shape and fd.shape == s.shape
    assert dr.shape == s.shape and dd.shape == s.shape
    assert a.last_attn_rgb.shape == (2, NUM_HEADS, 256, 256)
    assert a.last_attn_depth.shape == (2, NUM_HEADS, 256, 256)
    assert a.head_dim == HEAD_DIM == 16
    assert not torch.isnan(fr).any() and not torch.isnan(fd).any()


def test_two_head_split_merge_consistent() -> None:
    """Multi-head path: a single head uses head_dim; merged out dim = IN_CH."""
    a = XMVAttention(mode="bidirectional")
    s, _ = _feats(1)
    p_rgb = a.ln_rgb(a.proj_rgb(s).flatten(2).transpose(1, 2))
    q = a.wq_rgb(p_rgb)
    heads = a._heads(q)
    assert heads.shape == (1, NUM_HEADS, 256, HEAD_DIM)
    merged = a._merge(heads)
    assert merged.shape == (1, 256, a.attn_dim)
    assert torch.equal(merged, q)


# -- modes --------------------------------------------------------------

def test_rgb_from_depth_mode() -> None:
    s, d = _feats(2)
    a = XMVAttention(mode="rgb_from_depth")
    fr, fd, dr, dd = a(s, d, None)
    assert dr is not None and dd is None
    assert a.last_attn_rgb is not None and a.last_attn_depth is None
    assert torch.equal(fd, d)          # depth pathway unchanged
    assert float(a.gamma_depth_from_rgb.detach()) == 0.0


def test_depth_from_rgb_mode() -> None:
    s, d = _feats(2)
    a = XMVAttention(mode="depth_from_rgb")
    fr, fd, dr, dd = a(s, d, None)
    assert dr is None and dd is not None
    assert a.last_attn_depth is not None and a.last_attn_rgb is None
    assert torch.equal(fr, s)          # rgb pathway unchanged
    assert float(a.gamma_rgb_from_depth.detach()) == 0.0


def test_bidirectional_mode() -> None:
    s, d = _feats(2)
    a = XMVAttention(mode="bidirectional")
    fr, fd, dr, dd = a(s, d, None)
    assert dr is not None and dd is not None
    assert a.last_attn_rgb is not None and a.last_attn_depth is not None


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError):
        XMVAttention(mode="sideways")


# -- validity masking -----------------------------------------------------

def test_invalid_key_masking_disabled_key_gets_zero_attention() -> None:
    """RGB<-Depth: masked depth key logits -> exactly zero attention mass."""
    s, d = _feats(2)
    v = torch.ones(2, 1, 16, 16, dtype=torch.bool)
    v[:, :, :, 0:4] = False  # first 4 depth keys invalid
    a = XMVAttention(mode="rgb_from_depth")
    fr, fd, dr, _ = a(s, d, v)
    w = a.last_attn_rgb  # (2,H,256,256)
    invalid_cols = torch.zeros(256, dtype=torch.bool)
    invalid_cols[0:4] = True
    assert (w[:, :, :, invalid_cols] == 0).all()
    assert torch.isfinite(fr).all() and torch.isfinite(fd).all()
    # Rows with >=1 valid key must still sum to 1 per head.
    row_sums = w.sum(-1)  # (2,H,256)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_all_invalid_fallback_zero_residual() -> None:
    """Sample with zero valid depth tokens -> RGB<-Depth residual exactly 0."""
    s, d = _feats(2)
    v = torch.ones(2, 1, 16, 16, dtype=torch.bool)
    v[1] = False  # second sample fully invalid depth
    a = XMVAttention(mode="rgb_from_depth")
    fr, fd, dr, _ = a(s, d, v)
    assert (dr[1] == 0).all()
    assert torch.equal(fr[1], s[1])            # residual zero -> identity
    assert (a.last_attn_rgb[1] == 0).all()     # guarded, no uniform garbage
    assert torch.isfinite(fr).all()


def test_invalid_query_masking_zeroes_delta_depth() -> None:
    """Depth<-RGB: invalid depth query locations receive no update."""
    s, d = _feats(2)
    v = torch.ones(2, 1, 16, 16, dtype=torch.bool)
    v[:, :, 0:8, :] = False  # top half depth locations invalid
    a = XMVAttention(mode="depth_from_rgb")
    fr, fd, dr, dd = a(s, d, v)
    assert (dd[:, :, 0:8, :] == 0).all()
    assert (dd[:, :, 8:, :].abs().sum() > 0).all()  # valid half updates
    assert torch.isfinite(fd).all() and torch.isfinite(fr).all()


# -- NaN safety -----------------------------------------------------------

def test_no_nans_any_mode() -> None:
    s, d = _feats(4)
    v = torch.ones(4, 1, 16, 16, dtype=torch.bool)
    v[2, :, 0:6, :] = False
    v[3] = False
    for mode in ("rgb_from_depth", "depth_from_rgb", "bidirectional"):
        a = XMVAttention(mode=mode)
        fr, fd, dr, dd = a(s, d, v)
        for t in (fr, fd, dr, dd):
            if t is not None:
                assert torch.isfinite(t).all(), mode


# -- identity control -------------------------------------------------------

def test_gamma_zero_reproduces_phase2_predictions() -> None:
    """gamma=0 must give Phase 2 maps bit-exactly (strict tolerance)."""
    torch.manual_seed(0)
    rgb, depth = _branches()
    for mode in ("rgb_from_depth", "depth_from_rgb", "bidirectional"):
        m = XMVModel(rgb, depth, mode=mode, gamma_init=0.0).eval()
        x_rgb = torch.randn(1, 3, 64, 64)
        x_depth = torch.randn(1, 1, 64, 64)
        valid = torch.ones(1, 1, 64, 64, dtype=torch.bool)
        valid[:, :, :16, :] = False
        out = m.predict(x_rgb, x_depth, valid)
        with torch.no_grad():
            pr = rgb.predict(x_rgb, None)
            pd = depth.predict(x_depth, valid)
        assert torch.equal(out["rgb_map"], pr["anomaly_map"]), mode
        assert torch.equal(out["depth_map"], pd["anomaly_map"]), mode


def test_identity_also_at_attn_feature_level() -> None:
    torch.manual_seed(0)
    rgb, depth = _branches()
    m = XMVModel(rgb, depth, mode="bidirectional", gamma_init=0.0).eval()
    x_rgb = torch.randn(1, 3, 64, 64)
    x_depth = torch.randn(1, 1, 64, 64)
    fw = m.forward_16(x_rgb, x_depth, torch.ones(1, 1, 64, 64, dtype=torch.bool))
    # gamma=0: fused features equal the frozen student stage16 taps.
    s_rgb = m._student_taps(m.rgb.student, x_rgb)[1]
    s_depth = m._student_taps(m.depth.student, x_depth)[1]
    assert torch.equal(fw["f_rgb16"], s_rgb)
    assert torch.equal(fw["f_depth16"], s_depth)


# -- gradients and freezing -------------------------------------------------

def test_branches_frozen_attn_receives_gradients() -> None:
    torch.manual_seed(0)
    rgb, depth = _branches()
    m = XMVModel(rgb, depth, mode="bidirectional").train()
    x_rgb = torch.randn(2, 3, 64, 64)
    x_depth = torch.randn(2, 1, 64, 64)
    valid = torch.ones(2, 1, 64, 64, dtype=torch.bool)
    m.forward_train(x_rgb, x_depth, valid)["loss"].backward()
    assert all(not p.requires_grad for p in m.rgb.parameters())
    assert all(not p.requires_grad for p in m.depth.parameters())
    assert all(p.grad is None for p in m.rgb.parameters())
    assert all(p.grad is None for p in m.depth.parameters())
    grads = [p.grad for p in m.attn.parameters()]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_direction_disabled_params_get_no_grad() -> None:
    torch.manual_seed(0)
    for mode, inactive in (("rgb_from_depth", "gamma_depth_from_rgb"),
                           ("depth_from_rgb", "gamma_rgb_from_depth")):
        rgb, depth = _branches()
        m = XMVModel(rgb, depth, mode=mode).train()
        x_rgb = torch.randn(2, 3, 64, 64)
        x_depth = torch.randn(2, 1, 64, 64)
        m.forward_train(x_rgb, x_depth, torch.ones(2, 1, 64, 64, dtype=torch.bool))["loss"].backward()
        grads = {p for n, p in m.attn.named_parameters() if p.grad is not None}
        active = set(m.attn.active_parameters())
        inactive_params = set(m.attn.inactive_parameters())
        assert all(p not in grads for p in inactive_params), mode
        assert inactive not in {n.split(".")[0] for n, p in m.attn.named_parameters()
                                if p.grad is not None}, mode
        assert set(active) | inactive_params == set(m.attn.parameters())


def test_train_mode_keeps_branches_frozen() -> None:
    rgb, depth = _branches()
    m = XMVModel(rgb, depth, mode="bidirectional").train()
    assert not m.rgb.training and not m.depth.training
    assert m.attn.training


# -- determinism + checkpoint -------------------------------------------------

def test_deterministic_initialization() -> None:
    torch.manual_seed(0)
    a1 = XMVAttention(mode="bidirectional")
    torch.manual_seed(0)
    a2 = XMVAttention(mode="bidirectional")
    for (n1, p1), (n2, p2) in zip(a1.named_parameters(), a2.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)


def test_checkpoint_save_load() -> None:
    torch.manual_seed(3)
    rgb, depth = _branches()
    m = XMVModel(rgb, depth, mode="bidirectional", gamma_init=0.0)
    p = torch.save({"attn_state": m.attn.state_dict()}, "dummy")
    p = Path("dummy")
    try:
        m2 = XMVModel(*_branches(), mode="bidirectional")
        m2.attn.load_state_dict(torch.load(p, weights_only=True)["attn_state"])
        for a, b in zip(m.attn.state_dict().values(), m2.attn.state_dict().values()):
            assert torch.equal(a, b)
    finally:
        p.unlink(missing_ok=True)


# -- map output sizes ----------------------------------------------------------

def test_map_output_sizes() -> None:
    torch.manual_seed(0)
    rgb, depth = _branches()
    m = XMVModel(rgb, depth, mode="bidirectional").eval()
    x_rgb = torch.randn(1, 3, 64, 64)
    x_depth = torch.randn(1, 1, 64, 64)
    valid = torch.ones(1, 1, 64, 64, dtype=torch.bool)
    out = m.predict(x_rgb, x_depth, valid)
    assert out["rgb_map"].shape == (1, 1, 64, 64)
    assert out["depth_map"].shape == (1, 1, 64, 64)


def test_param_budget() -> None:
    a = XMVAttention(mode="bidirectional")
    assert a.count_parameters() < 20000  # E7 pre-registered target
    assert a.count_parameters() > 0


# -- no test split in training script (source guard) -------------------------

def test_no_test_split_in_training_script() -> None:
    src = open(Path(__file__).resolve().parents[1] / "scripts" / "train_cross_attention.py",
               encoding="utf-8").read()
    assert '"train"' in src and '"validation"' in src
    assert '"test"' not in src and "'test'" not in src
    assert "test_split" not in src and "test_set" not in src