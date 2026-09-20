"""Phase 3C gate tests: split isolation, determinism, frozen branches, heads."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from xmvad.models.fusion.reliability_gate import CONFIDENCE_FLOOR_ALPHA, ReliabilityGate


def test_gate_param_budget() -> None:
    m = ReliabilityGate(in_dim=27)
    assert m.count_parameters() < 5000


def test_gate_weights_sum_to_one_no_nan() -> None:
    torch.manual_seed(0)
    m = ReliabilityGate(in_dim=27).eval()
    x = torch.randn(16, 27)
    with torch.no_grad():
        out = m(x)
    for k in ("w_img", "w_map"):
        assert out[k].shape == (16, 2)
        assert torch.isfinite(out[k]).all()
        assert torch.allclose(out[k].sum(-1), torch.ones(16), atol=1e-5)
        # confidence floor: no extreme 0/1 weights
        assert (out[k] >= CONFIDENCE_FLOOR_ALPHA * 0.5 - 1e-6).all()
        assert (out[k] <= 1 - CONFIDENCE_FLOOR_ALPHA * 0.5 + 1e-6).all()


def test_gate_heads_independent() -> None:
    m = ReliabilityGate(in_dim=8)
    assert m.det_head is not m.loc_head
    assert set(id(p) for p in m.det_head.parameters()).isdisjoint(
        set(id(p) for p in m.loc_head.parameters()))
    # Heads can express different weights for the same input: give each
    # head's two rows opposing signs so its logits differ.
    torch.manual_seed(1)
    with torch.no_grad():
        m.det_head.weight[0].fill_(1.0)
        m.det_head.weight[1].fill_(-1.0)
        m.det_head.bias.zero_()
        m.loc_head.weight[0].fill_(-1.0)
        m.loc_head.weight[1].fill_(1.0)
        m.loc_head.bias.zero_()
        out = m(torch.randn(4, 8))
    assert not torch.allclose(out["w_img"], out["w_map"])


def test_gate_receives_gradients_branches_frozen() -> None:
    from xmvad.models.anomaly.teacher_student import TeacherStudentAD

    gate = ReliabilityGate(in_dim=27)
    det = TeacherStudentAD(modality="rgb", pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    loss = det.forward_train(x)["loss"] + gate(torch.randn(2, 27))["w_img"].sum()
    loss.backward()
    assert all(p.grad is None for p in det.teacher.parameters())
    assert any(p.grad is not None for p in gate.parameters())


def test_gate_checkpoint_save_load(tmp_path) -> None:
    m = ReliabilityGate(in_dim=27)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    p = tmp_path / "gate.pt"
    torch.save({"model_state": m.state_dict(), "in_dim": 27}, p)
    m2 = ReliabilityGate(in_dim=27)
    m2.load_state_dict(torch.load(p, weights_only=True)["model_state"])
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_split_isolation_and_variant_co_location() -> None:
    from pathlib import Path

    man = json.load(open(Path(__file__).resolve().parents[1] / "experiments" / "calibration"
                         / "E3C_split_manifest.json", encoding="utf-8"))["categories"]
    for cat, splits in man.items():
        tr, va = set(splits["gate_train"]), set(splits["gate_val"])
        assert tr.isdisjoint(va) and len(tr) > 0 and len(va) > 0
    # Gate data variants inherit their source split (checked per category file).
    import glob as gb

    for meta_path in sorted(gb.glob(str(Path(__file__).resolve().parents[1]
                                       / "experiments" / "gate_data" / "*_meta.json"))):
        meta = json.load(open(meta_path, encoding="utf-8"))
        cat = meta["category"]
        tr = set(man[cat]["gate_train"])
        for sid, split in zip(meta["source_ids"], meta["splits"]):
            stem = sid.split("|")[0].split("/")[-1]
            assert (stem in tr) == (split == "gate_train"), sid


def test_synthetic_determinism() -> None:
    from xmvad.data import synthetic_anomalies as SA

    rng = np.random.default_rng(0)
    rgb = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
    depth = rng.random((48, 64)).astype(np.float32)
    valid = np.ones((48, 64), bool)
    a1 = SA.perturb_rgb_color_patch(rgb, seed=11, strength=0.6)
    a2 = SA.perturb_rgb_color_patch(rgb, seed=11, strength=0.6)
    assert np.array_equal(a1[0], a2[0]) and np.array_equal(a1[1], a2[1])
    b1 = SA.perturb_joint_scratch_dent(rgb, depth, valid, seed=5)
    b2 = SA.perturb_joint_scratch_dent(rgb, depth, valid, seed=5)
    assert all(np.array_equal(x, y) for x, y in zip(b1, b2))


def test_no_test_records_in_gate_data() -> None:
    """Gate .npz source_ids must reference train split only (never test)."""
    import glob as gb
    from pathlib import Path

    files = sorted(gb.glob(str(Path(__file__).resolve().parents[1]
                               / "experiments" / "gate_data" / "*_meta.json")))
    assert files, "no gate data generated yet"
    for meta_path in files:
        meta = json.load(open(meta_path, encoding="utf-8"))
        for sid in meta["source_ids"]:
            assert "/train/" in sid.split("|")[0], sid
            assert "/test/" not in sid, sid
