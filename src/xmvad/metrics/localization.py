"""Pixel-level localization metrics (Phase 2).

Operates on flattened anomaly maps vs boolean ground-truth masks.
"""

from __future__ import annotations

import numpy as np


def _flatten(masks, maps) -> tuple[np.ndarray, np.ndarray]:
    y = np.concatenate([np.asarray(m, dtype=bool).ravel() for m in masks])
    s = np.concatenate([np.asarray(a, dtype=float).ravel() for a in maps])
    if y.shape != s.shape:
        raise ValueError(f"Shape mismatch: {y.shape} vs {s.shape}")
    return y.astype(int), s


def pixel_auroc(gt_masks, pred_maps) -> float:
    """Pixel-level AUROC over all test pixels; NaN if degenerate."""
    from sklearn.metrics import roc_auc_score

    y, s = _flatten(gt_masks, pred_maps)
    if np.unique(y).size < 2 or not np.all(np.isfinite(s)):
        return float("nan")
    return float(roc_auc_score(y, s))


def pixel_auprc(gt_masks, pred_maps) -> float:
    """Pixel-level AUPRC; NaN if degenerate."""
    from sklearn.metrics import average_precision_score

    y, s = _flatten(gt_masks, pred_maps)
    if np.unique(y).size < 2 or not np.all(np.isfinite(s)):
        return float("nan")
    return float(average_precision_score(y, s))


def iou_at_threshold(gt_mask: np.ndarray, pred_map: np.ndarray, threshold: float) -> float:
    """IoU of ``pred_map >= threshold`` vs GT; 1.0 when both empty."""
    gt = np.asarray(gt_mask, dtype=bool)
    pr = np.asarray(pred_map, dtype=float) >= threshold
    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def dice_at_threshold(gt_mask: np.ndarray, pred_map: np.ndarray, threshold: float) -> float:
    """Dice/F1 of ``pred_map >= threshold`` vs GT; 1.0 when both empty."""
    gt = np.asarray(gt_mask, dtype=bool)
    pr = np.asarray(pred_map, dtype=float) >= threshold
    denom = gt.sum() + pr.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(gt, pr).sum()) / float(denom)
