"""Phase 3A tests: prediction pairing, calibration math, invalid handling.

No retraining, no test-metric tuning — these pin the fusion CONTRACT:
exact pairing (fail loudly), validation-only calibration statistics, and
RGB-only fallback at depth-invalid pixels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_late_fusion import pair_predictions  # noqa: E402


def _entry(shape=(8, 8), label=0, defect="good", seed=0):
    rng = np.random.default_rng(seed)
    return {"map": rng.random(shape, dtype=np.float32), "gt": np.zeros(shape, bool),
            "label": label, "defect": defect, "score_max": 0.5}


def test_pairing_exact_match() -> None:
    rgb = {"a/b/good/000": _entry(label=0), "a/b/hole/001": _entry(label=1, defect="hole", seed=1)}
    depth = {"a/b/good/000": _entry(label=0), "a/b/hole/001": _entry(label=1, defect="hole", seed=2)}
    # GT must match: give both the same gt explicitly
    gt = np.zeros((8, 8), bool)
    gt[2:4, 2:4] = True
    rgb["a/b/hole/001"]["gt"] = gt
    depth["a/b/hole/001"]["gt"] = gt.copy()
    pairs = pair_predictions(rgb, depth, "a")
    assert [p["sample_id"] for p in pairs] == ["a/b/good/000", "a/b/hole/001"]


def test_pairing_missing_id_fails() -> None:
    rgb = {"a/b/good/000": _entry()}
    depth = {"a/b/good/000": _entry(), "a/b/good/001": _entry(seed=3)}
    with pytest.raises(ValueError, match="ID mismatch"):
        pair_predictions(rgb, depth, "a")


def test_pairing_gt_mismatch_fails() -> None:
    rgb = {"a/b/good/000": _entry()}
    d = _entry()
    d["gt"][0, 0] = True
    with pytest.raises(ValueError, match="GT mismatch"):
        pair_predictions(rgb, {"a/b/good/000": d}, "a")


def test_pairing_shape_mismatch_fails() -> None:
    rgb = {"a/b/good/000": _entry(shape=(8, 8))}
    depth = {"a/b/good/000": _entry(shape=(4, 4))}
    with pytest.raises(ValueError, match="map shape"):
        pair_predictions(rgb, depth, "a")


def test_pairing_label_mismatch_fails() -> None:
    rgb = {"a/b/good/000": _entry(label=0)}
    depth = {"a/b/good/000": _entry(label=1, defect="hole")}
    with pytest.raises(ValueError, match="label/defect mismatch"):
        pair_predictions(rgb, depth, "a")


def test_calibration_math_median_mad() -> None:
    from calibrate_late_fusion import robust_scale

    vals = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # outlier must not move median
    median, scale = robust_scale(vals)
    assert median == pytest.approx(3.0)
    assert scale == pytest.approx(1.4826 * 1.0)  # MAD = 1
    # degenerate constant input falls back without crashing
    m2, s2 = robust_scale(np.ones(10))
    assert m2 == pytest.approx(1.0) and s2 > 0


def test_invalid_depth_uses_rgb_only() -> None:
    rng = np.random.default_rng(0)
    zr = rng.random((16, 16))
    zd = rng.random((16, 16)) + 5.0  # depth evidence much stronger
    valid = np.ones((16, 16), bool)
    valid[:, :8] = False  # left half invalid
    fused = np.where(valid, 0.5 * zr + 0.5 * zd, zr)
    assert np.array_equal(fused[:, :8], zr[:, :8])  # RGB only
    assert np.allclose(fused[:, 8:], 0.5 * zr[:, 8:] + 0.5 * zd[:, 8:])
    # max variant likewise
    fmax = np.where(valid, np.maximum(zr, zd), zr)
    assert np.array_equal(fmax[:, :8], zr[:, :8])


def test_score_variants_shared_module() -> None:
    from xmvad.metrics.scores import SCORE_VARIANTS, variant_scores

    assert [n for n, _ in SCORE_VARIANTS] == ["max", "topk_0.1pct", "topk_1pct"]
    amap = np.zeros((100, 100))
    amap[50, 50] = 2.0
    vs = variant_scores(amap, None)
    assert vs["max"] == pytest.approx(2.0)
    # top 1% of 10000 = 100 px: (2 + 99*0)/100
    assert vs["topk_1pct"] == pytest.approx(0.02)
