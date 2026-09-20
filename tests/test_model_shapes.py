"""Model-shape / behavior tests for Phase 2 single-modal baselines.

Uses small random weights (pretrained=False) — no downloads, no dataset.
"""

from __future__ import annotations

import pytest
import torch

from xmvad.losses.distillation import (
    cosine_anomaly_map,
    cosine_discrepancy,
    downsample_valid_mask,
)
from xmvad.models.anomaly.teacher_student import TeacherStudentAD
from xmvad.models.encoders.depth_encoder import DepthTeacher, adapt_stem_to_1ch
from xmvad.models.encoders.rgb_encoder import OUT_CHANNELS, RGBTeacher


def test_rgb_teacher_taps_and_frozen() -> None:
    t = RGBTeacher(pretrained=False)
    assert t.is_frozen
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        feats = t(x)
    assert set(feats) == {"stage8", "stage16"}
    assert feats["stage8"].shape == (2, OUT_CHANNELS["stage8"], 8, 8)
    assert feats["stage16"].shape == (2, OUT_CHANNELS["stage16"], 4, 4)
    # Teacher ignores outer train mode.
    t.train(True)
    assert not t.training


def test_depth_stem_avg_init_and_frozen() -> None:
    w = torch.randn(16, 3, 3, 3)
    d1 = adapt_stem_to_1ch(w)
    assert d1.shape == (16, 1, 3, 3)
    assert torch.allclose(d1[:, 0], w.mean(dim=1))
    t = DepthTeacher(pretrained=False)
    assert t.is_frozen
    assert t.stem_in_channels == 1
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        feats = t(x)
    assert feats["stage8"].shape[1] == OUT_CHANNELS["stage8"]
    assert feats["stage16"].shape[1] == OUT_CHANNELS["stage16"]
    with pytest.raises(ValueError):
        t(torch.randn(1, 3, 64, 64))


@pytest.mark.parametrize("modality,ch", [("rgb", 3), ("depth", 1)])
def test_student_matches_teacher_shapes(modality: str, ch: int) -> None:
    m = TeacherStudentAD(modality=modality, pretrained=False)
    x = torch.randn(2, ch, 64, 64)
    valid = torch.ones(2, 1, 64, 64, dtype=torch.bool) if modality == "depth" else None
    out = m.forward_train(x, valid)
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() >= 0.0
    # Student heads predict frozen teacher channels directly (no projection).
    s = m.student(x)
    assert s["stage8"].shape[1] == OUT_CHANNELS["stage8"] == 24
    assert s["stage16"].shape[1] == OUT_CHANNELS["stage16"] == 48
    pred = m.predict(x, valid)
    assert pred["anomaly_map"].shape == (2, 1, 64, 64)
    assert pred["score"].shape == (2,)
    assert torch.isfinite(pred["anomaly_map"]).all()
    assert (pred["anomaly_map"] >= 0).all()


def test_gradients_reach_student_not_teacher() -> None:
    m = TeacherStudentAD(modality="rgb", pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    m.forward_train(x)["loss"].backward()
    teacher_grads = [p.grad for p in m.teacher.parameters()]
    assert all(g is None for g in teacher_grads)
    student_grads = [p.grad for p in m.student.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in student_grads)
    # Every trainable parameter must live inside the student (fixed targets).
    trainable_names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable_names and all(n.startswith("student.") for n in trainable_names)


def test_validity_mask_excludes_invalid_depth() -> None:
    m = TeacherStudentAD(modality="depth", pretrained=False)
    m.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 32)
    valid = torch.ones(1, 1, 32, 32, dtype=torch.bool)
    valid[:, :, :16, :] = False  # top half invalid
    pred = m.predict(x, valid)
    am = pred["anomaly_map"]
    assert (am[:, :, :16, :] == 0).all()
    # Score must ignore invalid region: all-invalidexcept check below uses max over valid.
    assert torch.isfinite(pred["score"]).all()


def test_downsample_valid_mask_majority() -> None:
    v = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    v[0, 0, 0, 0] = False  # single invalid pixel: 15/16 valid -> survives
    small = downsample_valid_mask(v, (2, 2))
    assert small.shape == (1, 1, 2, 2)
    assert small.all()
    v[:, :, :4, :4] = False  # whole cell invalid -> dropped
    small = downsample_valid_mask(v, (2, 2))
    assert not small[0, 0, 0, 0]
    assert small[0, 0, 1, 1]


def test_discrepancy_identical_is_zero() -> None:
    torch.manual_seed(1)
    t = torch.randn(2, 8, 4, 4)
    assert cosine_discrepancy(t, t.clone()).item() == pytest.approx(0.0, abs=1e-5)
    m = cosine_anomaly_map(t, t.clone())
    assert m.shape == (2, 1, 4, 4)
    assert (m.abs() < 1e-4).all()


def test_topk_score_mode() -> None:
    m = TeacherStudentAD(modality="rgb", pretrained=False, score_mode="topk", topk_frac=0.1)
    x = torch.randn(2, 3, 32, 32)
    s = m.predict(x)["score"]
    assert s.shape == (2,) and torch.isfinite(s).all()


def test_checkpoint_save_load_consistency(tmp_path) -> None:
    from xmvad.training.checkpoints import count_parameters, load_checkpoint, save_checkpoint

    m = TeacherStudentAD(modality="depth", pretrained=False)
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-3)
    p = tmp_path / "ckpt.pt"
    save_checkpoint(p, m, opt, None, epoch=2, val_loss=0.5,
                    config={"experiment": "t"}, seed=0)
    m2 = TeacherStudentAD(modality="depth", pretrained=False)
    ckpt = load_checkpoint(p, m2)
    assert ckpt["epoch"] == 2 and ckpt["modality"] == "depth"
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)
    info = count_parameters(m)
    assert info["trainable"] < info["total"] and info["frozen_teacher"] > 0


def test_predict_masks_invalid_before_upsampling() -> None:
    """Regression test: low-res invalid cells must be zeroed BEFORE bilinear
    upsampling, with the same conservative mask used in training.

    Without this, untrained invalid-cell values bleed into neighboring valid
    pixels and dominate max/top-k image scores (observed as chance-level
    depth metrics on real bagel data despite near-zero training loss).
    """
    import torch.nn.functional as F

    torch.manual_seed(0)
    m = TeacherStudentAD(modality="depth", pretrained=False)
    m.eval()
    x = torch.randn(1, 1, 64, 64)
    valid = torch.ones(1, 1, 64, 64, dtype=torch.bool)
    valid[:, :, :32, :] = False  # top half invalid
    pred = m.predict(x, valid)
    # Full-res invalid region is exactly zero.
    assert (pred["anomaly_map"][:, :, :32, :] == 0).all()
    # Per-level maps equal manual masked-then-upsampled computation.
    with torch.no_grad():
        t = m.teacher(x)  # frozen targets directly
        s = m.student(x)
        for lvl in ("stage8", "stage16"):
            raw = cosine_anomaly_map(t[lvl], s[lvl]).clamp_min(0.0)
            v_low = downsample_valid_mask(valid, raw.shape[-2:])
            expected = F.interpolate(torch.where(v_low, raw, torch.zeros_like(raw)),
                                     size=(64, 64), mode="bilinear", align_corners=False)
            assert torch.equal(pred["maps"][lvl], expected)


def test_predict_all_invalid_gives_zero() -> None:
    torch.manual_seed(0)
    m = TeacherStudentAD(modality="depth", pretrained=False)
    m.eval()
    x = torch.randn(1, 1, 32, 32)
    valid = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
    pred = m.predict(x, valid)
    assert (pred["anomaly_map"] == 0).all()
    assert pred["score"].item() == 0.0
