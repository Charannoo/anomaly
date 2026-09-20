"""Phase 3B tests: reliability rules, validity behavior, oracle isolation.

All rules are deterministic pure-numpy functions of validation-normal
statistics and calibrated evidence. No test labels enter any rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from xmvad.fusion.score_fusion import (
    EPS,
    TEMPERATURE,
    calibrate,
    evidence_weights,
    fuse_maps,
    static_weights,
    validity_adjusted_weights,
)


def test_static_weights_sum_to_one() -> None:
    w_r, w_d = static_weights(0.01, 0.05)
    assert w_r + w_d == pytest.approx(1.0)
    # Noisier depth gets lower weight.
    assert w_r > w_d
    # Degenerate zero noise is finite.
    a, b = static_weights(0.0, 0.0)
    assert np.isfinite([a, b]).all() and a + b == pytest.approx(1.0)


def test_validity_zero_kills_depth() -> None:
    w_r, w_d = validity_adjusted_weights(0.5, 0.5, valid_frac=0.0)
    assert w_d == pytest.approx(0.0, abs=1e-4)
    assert w_r + w_d == pytest.approx(1.0)
    w_r, w_d = validity_adjusted_weights(0.5, 0.5, valid_frac=1.0)
    assert w_r == pytest.approx(0.5) and w_d == pytest.approx(0.5)


def test_evidence_weights_fixed_temperature() -> None:
    assert TEMPERATURE == 1.0
    w_r, w_d = evidence_weights(3.0, 1.0)
    assert w_r + w_d == pytest.approx(1.0)
    assert w_r > w_d  # stronger evidence wins
    # Negative evidence clipped to zero -> uniform.
    a, b = evidence_weights(-2.0, -5.0)
    assert a == pytest.approx(0.5) and b == pytest.approx(0.5)
    # Deterministic.
    assert evidence_weights(3.0, 1.0) == (w_r, w_d)


def test_fuse_maps_invalid_fallback_and_finite() -> None:
    rng = np.random.default_rng(0)
    zr = rng.normal(size=(16, 16))
    zd = rng.normal(size=(16, 16)) + 10.0
    valid = np.ones((16, 16), bool)
    valid[:, :8] = False
    out = fuse_maps(zr, zd, valid, 0.3, 0.7)
    assert np.array_equal(out[:, :8], zr[:, :8])
    assert np.allclose(out[:, 8:], 0.3 * zr[:, 8:] + 0.7 * zd[:, 8:])
    assert np.isfinite(out).all()
    # Non-finite inputs cannot leak through.
    bad = zr.copy()
    bad[0, 0] = np.inf
    assert np.isfinite(fuse_maps(bad, zd, valid, 0.5, 0.5)).all()


def test_calibrate_no_clamp_documented() -> None:
    z = calibrate(np.array([[100.0]]), median=0.0, scale=1.0)
    assert z[0, 0] == pytest.approx(100.0)  # extreme values preserved


def test_oracle_isolated_from_inference() -> None:
    import xmvad.fusion.score_fusion as sf

    src = open(sf.__file__, encoding="utf-8").read()
    assert "oracle" not in src.lower()
    # Inference modules must not import the oracle.
    for mod in ("xmvad.models.anomaly.teacher_student", "xmvad.training.trainer"):
        m = __import__(mod, fromlist=["x"])
        path = (m.__file__ or "")
        if path:
            assert "oracle" not in open(path, encoding="utf-8").read().lower()


def test_oracle_uses_labels_correctly() -> None:
    from xmvad.analysis.oracle import oracle_image_scores, oracle_map_set

    rgb = np.array([0.9, 0.1, 0.8, 0.2])
    depth = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([1, 1, 0, 0])
    o = oracle_image_scores(rgb, depth, labels)
    # Anomalous take max, normal take min -> perfectly separable.
    assert list(o) == [0.9, 0.9, 0.2, 0.2]
    maps_r = [np.full((4, 4), v) for v in rgb]
    maps_d = [np.full((4, 4), v) for v in depth]
    gts = [np.zeros((4, 4), bool) for _ in range(4)]
    gts[0][1:3, 1:3] = True
    gts[1][1:3, 1:3] = True
    picked = oracle_map_set(maps_r, maps_d, gts, labels)
    assert float(picked[0].mean()) == pytest.approx(0.9)  # anomalous: higher GT response
    assert float(picked[2].mean()) == pytest.approx(0.2)  # normal: lower max


def test_reliability_stats_are_validation_only_by_construction() -> None:
    """The stats script builds datasets from the validation split only.

    The only discover_split call targets "validation"; no "test" split
    string is used as a split argument anywhere in the script.
    """
    from pathlib import Path

    src = open(Path(__file__).resolve().parents[1] / "scripts" / "compute_reliability_stats.py",
               encoding="utf-8").read()
    assert src.count("discover_split(") == 1
    assert 'category, "validation"' in src
    assert '"test"' not in src and "'test'" not in src
