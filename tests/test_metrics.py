"""Metric tests with known fixtures (Phase 2).

AUROC/AUPRC fixtures have analytically known values; PRO/AUPRO fixtures test
perfect detection (~1.0), empty masks (NaN), and multi-component averaging.
"""

from __future__ import annotations

import numpy as np
import pytest

from xmvad.metrics.detection import image_auprc, image_auroc
from xmvad.metrics.localization import dice_at_threshold, iou_at_threshold, pixel_auprc, pixel_auroc
from xmvad.metrics.pro import aupro, aupro_stream, pro_curve


def test_image_auroc_known() -> None:
    # Perfect ranking -> 1.0; worst -> 0.0; chance ordering -> 0.5.
    y = [0, 0, 1, 1]
    assert image_auroc(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert image_auroc(y, [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)
    assert image_auroc(y, [0.1, 0.9, 0.2, 0.8]) == pytest.approx(0.5)


def test_image_auprc_known() -> None:
    y = [0, 0, 1, 1]
    assert image_auprc(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    # Single-class input is degenerate -> NaN (never silently averaged).
    assert np.isnan(image_auroc([0, 0], [0.1, 0.2]))
    assert np.isnan(image_auprc([1, 1], [0.5, 0.6]))


def test_pixel_metrics_known() -> None:
    gt = [np.array([[0, 1], [1, 0]], dtype=bool)]
    perfect = [np.array([[0.0, 1.0], [1.0, 0.0]])]
    assert pixel_auroc(gt, perfect) == pytest.approx(1.0)
    assert pixel_auprc(gt, perfect) == pytest.approx(1.0)
    assert np.isnan(pixel_auroc([np.zeros((2, 2), bool)], perfect))


def test_iou_dice() -> None:
    gt = np.array([[1, 1], [0, 0]], dtype=bool)
    pr = np.array([[0.9, 0.4], [0.6, 0.1]])
    assert iou_at_threshold(gt, pr, 0.5) == pytest.approx(1 / 3)
    assert dice_at_threshold(gt, pr, 0.5) == pytest.approx(0.5)
    empty = np.zeros((2, 2), bool)
    assert iou_at_threshold(empty, np.zeros((2, 2)), 0.5) == 1.0
    assert dice_at_threshold(empty, np.zeros((2, 2)), 0.5) == 1.0


def test_aupro_perfect_is_one() -> None:
    gt = [np.zeros((16, 16), bool) for _ in range(3)]
    pred = [np.zeros((16, 16)) for _ in range(3)]
    gt[1][4:8, 4:8] = True
    pred[1][4:8, 4:8] = 1.0
    gt[2][10:12, 10:12] = True
    pred[2][10:12, 10:12] = 1.0
    assert aupro(gt, pred) == pytest.approx(1.0, abs=0.02)


def test_aupro_matches_official_golden() -> None:
    """Golden values produced by the OFFICIAL MVTec evaluation_code v2.0
    (pro_curve_util.compute_pro + generic_util.trapezoid, FPR limit 0.3).

    The diagonally-touching fixture pins 8-connectivity: the two pixels form
    ONE region officially (0.205387205), but would form TWO under 4-connectivity.
    """
    rng = np.random.default_rng(7)
    gt = np.zeros((10, 10), bool)
    gt[2, 2] = True
    gt[3, 3] = True
    pr = rng.random((10, 10))
    assert aupro([gt, np.zeros((10, 10), bool)], [pr, np.zeros((10, 10))]) == pytest.approx(
        0.205387205, abs=1e-9
    )


def test_aupro_empty_is_nan() -> None:
    gt = [np.zeros((8, 8), bool)]
    pred = [np.zeros((8, 8))]
    assert np.isnan(aupro(gt, pred))


def test_pro_curve_anchored_and_monotone() -> None:
    rng = np.random.default_rng(0)
    gt = [(rng.random((16, 16)) > 0.9)]
    pred = [rng.random((16, 16))]
    fprs, pros = pro_curve(gt, pred)
    assert fprs[0] == 0.0 and pros[0] == 0.0
    assert fprs[-1] == 1.0 and pros[-1] == 1.0
    assert (np.diff(fprs) >= -1e-9).all()  # non-decreasing as threshold loosens
    assert ((pros >= 0) & (pros <= 1)).all()


def test_aupro_stream_matches_exact() -> None:
    """Memory-bounded AUPRO must equal the exact implementation bit-for-bit."""
    rng = np.random.default_rng(11)
    gts, preds = [], []
    for _ in range(4):
        g = rng.random((24, 24)) > 0.8
        gts.append(g)
        p = rng.random((24, 24)) + (0.5 if g.any() else 0.0)
        preds.append(p)
    assert aupro_stream(gts, preds) == pytest.approx(aupro(gts, preds), abs=1e-12)
    assert np.isnan(aupro_stream([np.zeros((8, 8), bool)], [np.zeros((8, 8))]))
