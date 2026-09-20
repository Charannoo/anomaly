"""TEST-ORACLE DIAGNOSTICS — ANALYSIS ONLY. NOT A DEPLOYABLE METHOD.

This module uses TEST LABELS (and GT masks) to compute upper-bound
diagnostics for modality selection. It must NEVER appear in the main model
comparison as a real method and must NEVER be imported by any inference
path (enforced by tests/test_reliability_fusion.py).

Purpose: measure headroom — if oracle >> best single branch, reliability
selection has significant potential; if oracle ~= best branch, gating is
not worth pursuing.
"""

from __future__ import annotations

import numpy as np


def oracle_image_scores(rgb_scores: np.ndarray, depth_scores: np.ndarray,
                        labels: np.ndarray) -> np.ndarray:
    """Per-sample branch selection using test labels.

    Anomalous samples take the HIGHER calibrated score, normal samples the
    LOWER one. This is the selection upper bound, not a method.
    """
    rgb_scores = np.asarray(rgb_scores, dtype=float)
    depth_scores = np.asarray(depth_scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    return np.where(labels == 1,
                    np.maximum(rgb_scores, depth_scores),
                    np.minimum(rgb_scores, depth_scores))


def oracle_map_set(rgb_maps: list[np.ndarray], depth_maps: list[np.ndarray],
                   gts: list[np.ndarray], labels: np.ndarray) -> list[np.ndarray]:
    """Per-image map selection using test labels + GT (localization oracle).

    Anomalous images take the branch map with the HIGHER mean GT-region
    response; normal images take the branch map with the LOWER max score
    (fewer false positives). Analysis only.
    """
    out = []
    for rm, dm, gt, lab in zip(rgb_maps, depth_maps, gts, labels):
        gt = np.asarray(gt, dtype=bool)
        if int(lab) == 1 and gt.any():
            r_resp = float(np.asarray(rm, dtype=float)[gt].mean())
            d_resp = float(np.asarray(dm, dtype=float)[gt].mean())
            out.append(rm if r_resp >= d_resp else dm)
        else:
            r_max = float(np.max(np.asarray(rm, dtype=float)))
            d_max = float(np.max(np.asarray(dm, dtype=float)))
            out.append(rm if r_max <= d_max else dm)
    return out
