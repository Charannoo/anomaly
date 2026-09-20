"""PRO / AUPRO metric implementing the official MVTec 3D-AD evaluation protocol.

Algorithm (matches official ``evaluation_code`` v2.0 ``pro_curve_util.py`` /
``generic_util.trapezoid`` exactly):
- 8-connectivity for ground-truth components (``np.ones((3, 3))`` structuring
  element); every component contributes equally regardless of size.
- Exact curve over ALL unique anomaly scores (sort descending + cumsum, ties
  merged keeping the final point), anchored at (0, 0) and (1, 1).
- Trapezoidal integration up to ``fpr_limit`` (default 0.3) with linear
  interpolation when the limit falls between sampled FPRs; normalized by the
  limit so a perfect detector scores 1.0.

Verification: ``tests/test_metrics.py::test_aupro_matches_official_golden``
pins golden values produced by the official script on fixed fixtures, and a
standalone cross-check (official script vs this module) on synthetic + real
256px prediction maps agreed to < 1e-9 (see docs/METRICS.md).

Remaining documented difference vs official end-to-end evaluation: the
official script compares FULL-resolution maps; Phase 2 evaluates at config
``input_size`` (256x256). The curve/integration code path is identical.

References: Bergmann et al., "The MVTec 3D-AD Dataset for Unsupervised 3D
Anomaly Detection and Localization" (VISAPP 2022); MVTec evaluation_code v2.0.
"""

from __future__ import annotations

from bisect import bisect

import numpy as np


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy.ndimage import label

    structure = np.ones((3, 3), dtype=int)  # 8-connectivity per official code
    return label(np.asarray(mask, dtype=bool), structure=structure)


def pro_curve(
    gt_masks: list[np.ndarray],
    pred_maps: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Official PRO curve: (fprs, pros) anchored at (0,0) and (1,1).

    All test images (normal + anomalous) are included; normal images
    contribute ok-pixels to the FPR denominator, matching the official
    script (whose ``test/good/gt`` all-black masks are part of the set).
    """
    gts = [np.asarray(m, dtype=bool) for m in gt_masks]
    preds = [np.asarray(a, dtype=float) for a in pred_maps]
    if len(gts) != len(preds):
        raise ValueError("gt_masks and pred_maps must have equal length.")
    if any(g.shape != p.shape for g, p in zip(gts, preds)):
        raise ValueError("Each GT mask must match its anomaly map shape.")
    if any(not np.all(np.isfinite(p)) for p in preds):
        return np.array([np.nan]), np.array([np.nan])

    n = len(gts)
    h, w = gts[0].shape
    fp_changes = np.zeros((n, h, w), dtype=np.uint64)
    pro_changes = np.zeros((n, h, w), dtype=np.float64)
    num_ok_pixels = 0
    num_gt_regions = 0
    for i, (gt, _pred) in enumerate(zip(gts, preds)):
        labeled, n_components = _label_components(gt)
        num_gt_regions += n_components
        ok_mask = labeled == 0
        num_ok_pixels += int(ok_mask.sum())
        fp_change = np.zeros_like(gt, dtype=np.uint64)
        fp_change[ok_mask] = 1
        pro_change = np.zeros_like(gt, dtype=np.float64)
        for k in range(n_components):
            region = labeled == (k + 1)
            pro_change[region] = 1.0 / max(int(region.sum()), 1)
        fp_changes[i] = fp_change
        pro_changes[i] = pro_change

    if num_gt_regions == 0 or num_ok_pixels == 0:
        return np.array([np.nan]), np.array([np.nan])

    scores = np.array(preds).ravel()
    fp_flat = fp_changes.ravel()
    pro_flat = pro_changes.ravel()
    order = np.argsort(scores, kind="stable")[::-1]
    scores_sorted = scores[order]
    fp_sorted = np.cumsum(fp_flat[order]).astype(np.float64) / num_ok_pixels
    pro_sorted = np.cumsum(pro_flat[order]) / num_gt_regions

    # Merge ties: keep the final point of each equal-score group.
    keep = np.append(np.diff(scores_sorted) != 0, True)
    fprs = np.clip(fp_sorted[keep], None, 1.0)
    pros = np.clip(pro_sorted[keep], None, 1.0)
    return np.concatenate(([0.0], fprs, [1.0])), np.concatenate(([0.0], pros, [1.0]))


def trapezoid(x, y, x_max: float | None = None) -> float:
    """Definite integral by trapezoidal rule with upper-limit interpolation.

    Mirrors official ``generic_util.trapezoid`` (bisect insertion +
    linear interpolation of y at x_max when x_max is not sampled).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            ins = bisect(list(x), x_max)
            assert 0 < ins < len(x), "x_max outside curve range"
            y_interp = y[ins - 1] + (y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1])
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])
        m = x <= x_max
        x, y = x[m], y[m]
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction)


def pro_curve_stream(
    gt_masks: list[np.ndarray],
    pred_maps: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """PRO curve with the EXACT official algorithm but bounded memory.

    Mathematically identical to :func:`pro_curve`: same component weighting
    (8-connectivity, equal weight per component), same cumulative FPR/PRO,
    same tie merging. The only differences are storage types that do not
    change the result: the false-positive indicator is ``uint8`` instead of
    ``uint64`` (its values are 0/1) and per-image arrays are released eagerly.

    Intended for FULL-resolution (native-GT) evaluation, where the dense
    ``(n, H, W)`` float64 arrays of :func:`pro_curve` exceed memory. Verified
    to equal :func:`pro_curve` to < 1e-12 on 256px fixtures
    (see ``tests/test_pro.py::test_stream_matches_exact``).
    """
    gts = [np.asarray(m, dtype=bool) for m in gt_masks]
    preds = [np.asarray(a, dtype=float) for a in pred_maps]
    if len(gts) != len(preds):
        raise ValueError("gt_masks and pred_maps must have equal length.")
    if any(g.shape != p.shape for g, p in zip(gts, preds)):
        raise ValueError("Each GT mask must match its anomaly map shape.")
    if any(not np.all(np.isfinite(p)) for p in preds):
        return np.array([np.nan]), np.array([np.nan])

    from scipy.ndimage import label

    structure = np.ones((3, 3), dtype=int)  # 8-connectivity per official code
    s_parts: list[np.ndarray] = []
    fp_parts: list[np.ndarray] = []
    pro_parts: list[np.ndarray] = []
    num_ok_pixels = 0
    num_gt_regions = 0
    for i, (gt, pred) in enumerate(zip(gts, preds)):
        labeled, n_components = label(gt, structure=structure)
        num_gt_regions += n_components
        ok_mask = labeled == 0
        num_ok_pixels += int(ok_mask.sum())
        pro = np.zeros_like(gt, dtype=np.float64)
        for k in range(n_components):
            region = labeled == (k + 1)
            pro[region] = 1.0 / max(int(region.sum()), 1)
        fp = np.zeros(gt.size, dtype=np.uint8)
        fp[ok_mask.ravel()] = 1
        s_parts.append(np.ascontiguousarray(pred, dtype=np.float64).ravel())
        fp_parts.append(fp)
        pro_parts.append(np.ascontiguousarray(pro, dtype=np.float64).ravel())

    if num_gt_regions == 0 or num_ok_pixels == 0:
        return np.array([np.nan]), np.array([np.nan])

    scores = np.concatenate(s_parts)
    fp_flat = np.concatenate(fp_parts).astype(np.uint64, copy=False)
    pro_flat = np.concatenate(pro_parts)
    del s_parts, fp_parts, pro_parts, pro  # type: ignore[name-defined]

    order = np.argsort(scores, kind="stable")[::-1]
    fp_sorted = np.cumsum(fp_flat[order]).astype(np.float64) / num_ok_pixels
    pro_sorted = np.cumsum(pro_flat[order]) / num_gt_regions
    fp_flat = pro_flat = None  # type: ignore[assignment]

    scores_sorted = scores[order]
    keep = np.append(np.diff(scores_sorted) != 0, True)
    fprs = np.clip(fp_sorted[keep], None, 1.0)
    pros = np.clip(pro_sorted[keep], None, 1.0)
    return np.concatenate(([0.0], fprs, [1.0])), np.concatenate(([0.0], pros, [1.0]))


def aupro(
    gt_masks: list[np.ndarray],
    pred_maps: list[np.ndarray],
    fpr_limit: float = 0.3,
) -> float:
    """AUPRO normalized to [0, 1]; NaN when undefined (no anomaly pixels)."""
    fprs, pros = pro_curve(gt_masks, pred_maps)
    if np.isnan(fprs).any() or len(fprs) < 2:
        return float("nan")
    return trapezoid(fprs, pros, x_max=fpr_limit) / fpr_limit


def aupro_stream(
    gt_masks: list[np.ndarray],
    pred_maps: list[np.ndarray],
    fpr_limit: float = 0.3,
) -> float:
    """Memory-bounded AUPRO; identical value to :func:`aupro` (see above)."""
    fprs, pros = pro_curve_stream(gt_masks, pred_maps)
    if np.isnan(fprs).any() or len(fprs) < 2:
        return float("nan")
    return trapezoid(fprs, pros, x_max=fpr_limit) / fpr_limit
