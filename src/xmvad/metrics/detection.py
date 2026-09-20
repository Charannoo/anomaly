"""Image-level detection metrics (Phase 2).

AUROC / AUPRC via scikit-learn. Degenerate inputs (e.g. a category split
with a single class — should not happen on MVTec 3D-AD test, but guarded)
return NaN instead of crashing; callers must not silently average NaNs.
"""

from __future__ import annotations

import numpy as np


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_score.shape}")
    return y_true, y_score


def image_auroc(y_true, y_score) -> float:
    """Image-level AUROC; NaN if only one class present."""
    from sklearn.metrics import roc_auc_score

    y_true, y_score = _as_arrays(y_true, y_score)
    if np.unique(y_true).size < 2:
        return float("nan")
    if not np.all(np.isfinite(y_score)):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def image_auprc(y_true, y_score) -> float:
    """Image-level area under precision-recall curve; NaN if degenerate."""
    from sklearn.metrics import average_precision_score

    y_true, y_score = _as_arrays(y_true, y_score)
    if np.unique(y_true).size < 2:
        return float("nan")
    if not np.all(np.isfinite(y_score)):
        return float("nan")
    return float(average_precision_score(y_true, y_score))
