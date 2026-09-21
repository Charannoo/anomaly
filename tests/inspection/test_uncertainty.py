"""Tests for Uncertainty quantification, sensor quality, and Manual Review Guard."""

import pytest
import numpy as np

from xmvad.inspection.quality import evaluate_quality
from xmvad.inspection.uncertainty import evaluate_review_guard


def test_near_threshold_triggers_manual_review():
    """Verify that samples near the decision boundary trigger manual review recommendations."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    rgb = np.ones((100, 100, 3), dtype=np.uint8) * 150
    xyz = np.ones((100, 100, 3), dtype=np.float32) * 50.0
    valid = np.ones((100, 100), dtype=bool)

    quality = evaluate_quality(mask, rgb, xyz, valid, surface_fit_confidence=0.90)

    # Score = 0.51, Threshold = 0.50 -> Margin = 0.01 (Borderline)
    guard = evaluate_review_guard(
        image_score=0.51,
        threshold=0.50,
        quality=quality,
        is_physical_calibrated=True
    )

    assert guard.pntc_decision == "anomalous"  # Canonical decision strictly preserved
    assert guard.decision_certainty == "Borderline"
    assert guard.inspection_status == "MANUAL_REVIEW_RECOMMENDED"
    assert any("NEAR_DECISION_THRESHOLD" in r for r in guard.reasons)


def test_low_xyz_coverage_triggers_warning():
    """Verify that poor XYZ sensor coverage (<50%) triggers manual review recommendation."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True  # 1600 px
    rgb = np.ones((100, 100, 3), dtype=np.uint8) * 150
    xyz = np.ones((100, 100, 3), dtype=np.float32) * 50.0
    valid = np.zeros((100, 100), dtype=bool)
    valid[30:50, 30:50] = True  # Only 400 valid px out of 1600 (25% coverage)

    quality = evaluate_quality(mask, rgb, xyz, valid, surface_fit_confidence=0.85)
    assert quality.xyz_valid_fraction < 0.30

    guard = evaluate_review_guard(
        image_score=0.85,
        threshold=0.50,
        quality=quality,
        is_physical_calibrated=True
    )

    assert guard.pntc_decision == "anomalous"
    assert guard.inspection_status == "MANUAL_REVIEW_RECOMMENDED"
    assert any("LOW_XYZ_COVERAGE" in r for r in guard.reasons)


def test_strong_clear_defect_accepted():
    """Verify that a clear, unambiguous defect is marked DEFECT_DETECTED without unnecessary flags."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    rgb = np.ones((100, 100, 3), dtype=np.uint8) * 150
    xyz = np.ones((100, 100, 3), dtype=np.float32) * 50.0
    valid = np.ones((100, 100), dtype=bool)

    quality = evaluate_quality(mask, rgb, xyz, valid, surface_fit_confidence=0.92)

    # Score = 0.92, Threshold = 0.50 -> Strong anomaly
    guard = evaluate_review_guard(
        image_score=0.92,
        threshold=0.50,
        quality=quality,
        is_physical_calibrated=True
    )

    assert guard.pntc_decision == "anomalous"
    assert guard.inspection_status == "DEFECT_DETECTED"
    assert guard.decision_certainty == "High"
    assert guard.measurement_reliability == "High"


def test_missing_calibration_does_not_alter_detection():
    """Verify uncalibrated coordinates add a warning but preserve detection decision."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    rgb = np.ones((100, 100, 3), dtype=np.uint8) * 150
    xyz = np.ones((100, 100, 3), dtype=np.float32) * 50.0
    valid = np.ones((100, 100), dtype=bool)

    quality = evaluate_quality(mask, rgb, xyz, valid, surface_fit_confidence=0.90)

    guard = evaluate_review_guard(
        image_score=0.88,
        threshold=0.50,
        quality=quality,
        is_physical_calibrated=False  # Uncalibrated!
    )

    assert guard.pntc_decision == "anomalous"
    assert any("MISSING_PHYSICAL_CALIBRATION" in r for r in guard.reasons)
