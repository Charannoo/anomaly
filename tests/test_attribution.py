"""Attribution / evidence module tests with known fixtures (Phase 5).

Evidencer semantics under test:
- region evidence (topk / mean, empty-region and non-finite safety),
- positive-clamped relative evidence and [0,1] normalization,
- evidence classes (RGB/DEPTH-DOMINANT, JOINT, UNCERTAIN) with margins,
- decision ablation deltas and AttributionConsistency sign agreement,
- spatial explanation metrics on a synthetic perfect / inverted map,
- zero-evidence fallback safety (no NaN, no division blow-up).

These only check the metrics module: relative anomaly evidence, not a causal
claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from xmvad.metrics.attribution import (CLASS_DEPTH_DOMINANT,
                                       CLASS_JOINT, CLASS_RGB_DOMINANT,
                                       CLASS_UNCERTAIN,
                                       attribution_consistency,
                                       decision_ablation, evidence_class,
                                       evidence_margin, normalize_evidence,
                                       region_evidence, relative_evidence,
                                       spatial_explanation_metrics)

EPS = 1e-6


def test_region_evidence_empty_and_values() -> None:
    amap = np.arange(16, dtype=np.float64).reshape(4, 4)
    empty = np.zeros((4, 4), dtype=bool)
    full = np.ones((4, 4), dtype=bool)
    assert region_evidence(amap, empty) == 0.0
    # Top-1% of 16 px = top 1 px (15.0); mean-mode = 7.5.
    assert region_evidence(amap, full, mode="topk", frac=0.01) == pytest.approx(15.0)
    assert region_evidence(amap, full, mode="mean") == pytest.approx(7.5)
    with pytest.raises(ValueError):
        region_evidence(amap, full, mode="bogus")


def test_region_evidence_nonfinite_guarded() -> None:
    amap = np.array([[1.0, np.nan], [np.inf, 3.0]])
    amap = amap / 3.0  # keep scale
    full = np.ones((2, 2), dtype=bool)
    val = region_evidence(amap, full, mode="mean")
    assert np.isfinite(val)
    assert val == pytest.approx(2.0 / 3.0)  # finite 1/3 and 3/3 normalized


def test_relative_evidence_properties() -> None:
    p_rgb, p_depth = relative_evidence(3.0, 1.0)
    assert p_rgb == pytest.approx(0.75)
    assert p_depth == pytest.approx(0.25)
    assert p_rgb + p_depth == pytest.approx(1.0)
    # Negative evidence is clamped to zero before normalization.
    p_rgb, p_depth = relative_evidence(-2.0, 4.0)
    assert p_rgb == pytest.approx(0.0)
    assert p_depth == pytest.approx(1.0)
    # Both zero -> EPS-split, never NaN.
    p_rgb, p_depth = relative_evidence(0.0, 0.0)
    assert np.isfinite(p_rgb) and np.isfinite(p_depth)
    assert p_rgb == pytest.approx(0.5)


def test_normalize_evidence_bounds() -> None:
    u_r, u_d = normalize_evidence(3.0, 1.0)
    assert u_r == pytest.approx(1.0)
    assert u_d == pytest.approx(1.0 / 3.0)
    u_r, u_d = normalize_evidence(0.0, 0.0)
    assert u_r == 0.0 and u_d == 0.0
    for a, b in [(5.0, -1.0), (-1.0, 5.0), (0.0, 2.0), (2.0, 0.0)]:
        u_a, u_b = normalize_evidence(a, b)
        assert (u_a >= 0.0) and (u_b >= 0.0)
        assert (u_a <= 1.0 + EPS) and (u_b <= 1.0 + EPS)


def test_evidence_class_fixtures() -> None:
    assert evidence_class(5.0, 0.5) == CLASS_RGB_DOMINANT        # p ~ .91
    assert evidence_class(0.5, 5.0) == CLASS_DEPTH_DOMINANT      # p ~ .09
    assert evidence_class(2.0, 2.0) == CLASS_JOINT               # p ~ .5
    assert evidence_class(0.02, 0.01) == CLASS_UNCERTAIN         # below floor
    assert evidence_class(0.0, 0.0) == CLASS_UNCERTAIN
    # Margin boundary: no-evidence returns UNCERTAIN; clear >0.7 dominance.
    assert evidence_class(8.0, 2.0) == CLASS_RGB_DOMINANT        # p ~ .8
    assert evidence_class(2.0, 8.0) == CLASS_DEPTH_DOMINANT      # p ~ .2


def test_evidence_margin_fields() -> None:
    m = evidence_margin(9.0, 1.0)
    assert set(m) == {"dominance_margin", "p_rgb", "evidence_gap"}
    assert m["p_rgb"] == pytest.approx(0.9)


def test_decision_ablation_arithmetic() -> None:
    d = decision_ablation(s_full=10.0, s_rgb_only=6.0, s_depth_only=4.0)
    assert d["Delta_RGB"] == pytest.approx(6.0)
    assert d["Delta_DEPTH"] == pytest.approx(4.0)
    assert d["S_full"] == pytest.approx(10.0)


def test_attribution_consistency_sign_agreement() -> None:
    # RGB-leaning evidence with RGB-removal hurting more -> consistent.
    assert attribution_consistency(p_rgb=0.9, delta_rgb=5.0, delta_depth=1.0)
    assert attribution_consistency(p_rgb=0.1, delta_rgb=1.0, delta_depth=5.0)
    # Contradiction.
    assert not attribution_consistency(p_rgb=0.9, delta_rgb=1.0, delta_depth=5.0)
    # Flat evidence cannot contradict either direction.
    assert attribution_consistency(p_rgb=0.5 + 1e-12, delta_rgb=-2.0, delta_depth=-1.0)
    assert attribution_consistency(p_rgb=0.5 - 1e-12, delta_rgb=3.0, delta_depth=0.0)


def test_spatial_explanation_metrics_perfect_and_inverted() -> None:
    # All positives strictly above all negatives.
    pred = np.array([[0.2, 0.95], [0.90, 0.1]], dtype=float)
    gt = np.array([[0, 1], [1, 0]], dtype=bool)
    m = spatial_explanation_metrics(pred, gt)
    assert m["pixel_auroc"] == pytest.approx(1.0)
    assert m["pixel_auprc"] == pytest.approx(1.0)
    assert m["dice_best"] == pytest.approx(1.0)      # threshold isolates the 2 positives
    assert m["iou_best"] == pytest.approx(1.0)
    inv = 1.0 - pred                                  # positives now rank below negatives
    m2 = spatial_explanation_metrics(inv, gt)
    assert m2["pixel_auroc"] == pytest.approx(0.0)
    # Empty GT -> degenerate AUROC = NaN; no threshold is ever evaluated, so
    # dice_best stays 0.0 (never claimed as "both empty").
    m3 = spatial_explanation_metrics(pred, np.zeros((2, 2), bool))
    assert np.isnan(m3["pixel_auroc"])
    assert m3["dice_best"] == pytest.approx(0.0)