#!/usr/bin/env python3
"""Metrics for XMV-AD-H: I-AUROC, P-AUROC and AUPRO@0.3.

AUPRO is computed with the exact routine shipped in the local repo
(third_party/m3dm/utils/au_pro_util.py, itself the official MVTec 3D-AD
evaluation code), so numbers are directly comparable to the published
M3DM protocol (integration limit 0.3, 100 thresholds, 224x224 maps).
"""

import csv
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from bisect import bisect

try:
    from scipy.ndimage import label
except Exception:  # very old scipy
    from scipy.ndimage.measurements import label


# ----------------------------------------------------------------------
# AUPRO (official MVTec 3D-AD evaluation code, via repo au_pro_util.py)
# ----------------------------------------------------------------------

class GroundTruthComponent(object):
    """Sorted anomaly scores of one ground-truth connected component."""

    def __init__(self, anomaly_scores):
        self.anomaly_scores = anomaly_scores.copy()
        self.anomaly_scores.sort()
        self.index = 0
        self.last_threshold = None

    def compute_overlap(self, threshold):
        if self.last_threshold is not None:
            assert self.last_threshold <= threshold
        while (self.index < len(self.anomaly_scores)
               and self.anomaly_scores[self.index] <= threshold):
            self.index += 1
        return 1.0 - self.index / len(self.anomaly_scores)


def trapezoid(x, y, x_max=None):
    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print("WARNING: non-finite values in PRO curve dropped.")
    x = x[finite_mask]
    y = y[finite_mask]
    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            ins = bisect(x, x_max)
            assert 0 < ins < len(x)
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])
        mask = x <= x_max
        x = x[mask]
        y = y[mask]
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def collect_anomaly_scores(anomaly_maps, ground_truth_maps):
    assert len(anomaly_maps) == len(ground_truth_maps)
    ground_truth_components = []
    anomaly_scores_ok_pixels = np.zeros(len(ground_truth_maps) * ground_truth_maps[0].size)
    structure = np.ones((3, 3), dtype=int)
    ok_index = 0
    for gt_map, prediction in zip(ground_truth_maps, anomaly_maps):
        labeled, n_components = label(gt_map, structure)
        num_ok_pixels = len(prediction[labeled == 0])
        anomaly_scores_ok_pixels[ok_index:ok_index + num_ok_pixels] = prediction[labeled == 0].copy()
        ok_index += num_ok_pixels
        for k in range(n_components):
            component_scores = prediction[labeled == (k + 1)]
            ground_truth_components.append(GroundTruthComponent(component_scores))
    anomaly_scores_ok_pixels = np.resize(anomaly_scores_ok_pixels, ok_index)
    anomaly_scores_ok_pixels.sort()
    return ground_truth_components, anomaly_scores_ok_pixels


def compute_pro(anomaly_maps, ground_truth_maps, num_thresholds):
    ground_truth_components, anomaly_scores_ok_pixels = collect_anomaly_scores(
        anomaly_maps, ground_truth_maps)
    threshold_positions = np.linspace(0, len(anomaly_scores_ok_pixels) - 1,
                                      num=num_thresholds, dtype=int)
    fprs = [1.0]
    pros = [1.0]
    for pos in threshold_positions:
        threshold = anomaly_scores_ok_pixels[pos]
        fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)
        pro = 0.0
        for component in ground_truth_components:
            pro += component.compute_overlap(threshold)
        pro /= len(ground_truth_components)
        fprs.append(fpr)
        pros.append(pro)
    fprs = fprs[::-1]
    pros = pros[::-1]
    return fprs, pros


def calculate_au_pro(gts, predictions, integration_limit=0.3, num_thresholds=100):
    """Area under the PRO curve up to `integration_limit` (default 0.3)."""
    pro_curve = compute_pro(anomaly_maps=predictions, ground_truth_maps=gts,
                            num_thresholds=num_thresholds)
    au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max=integration_limit)
    au_pro /= integration_limit
    return au_pro, pro_curve


# ----------------------------------------------------------------------
# High-level metric computation
# ----------------------------------------------------------------------

def safe_auc(y_true, y_score):
    if y_true is None or y_score is None or len(y_true) < 2:
        return None
    if len(set(list(y_true))) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def compute_metrics(image_labels, image_scores, pixel_labels, pixel_scores,
                    gts, maps, integration_limit=0.3, num_thresholds=100):
    """Return dict with I-AUROC, P-AUROC, AUPRO (or None when undefined)."""
    i_auroc = safe_auc(image_labels, image_scores)

    if pixel_labels is not None and len(pixel_labels) > 0:
        p_auroc = safe_auc(pixel_labels, pixel_scores)
    else:
        p_auroc = None

    au_pro = None
    if gts is not None and maps is not None and len(gts) > 0:
        has_anomaly = any(np.any(g > 0.5) for g in gts)
        if has_anomaly:
            au_pro, _ = calculate_au_pro(gts, maps,
                                         integration_limit=integration_limit,
                                         num_thresholds=num_thresholds)
            au_pro = float(au_pro)
    return {"I": i_auroc, "P": p_auroc, "AUPRO": au_pro}


# ----------------------------------------------------------------------
# CSV output helpers
# ----------------------------------------------------------------------

RESULT_FIELDS = ["category", "I", "P", "AUPRO"]


def write_results_csv(path, rows, provenance="measured_by_us", model="", extra_fields=None):
    fieldnames = ["model", "category", "provenance", "I", "P", "AUPRO"]
    if extra_fields:
        fieldnames = fieldnames + [c for c in extra_fields if c not in fieldnames]
    ensure = os.path.dirname(path)
    if ensure:
        os.makedirs(ensure, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = dict(r)
            row.setdefault("provenance", provenance)
            if model:
                row.setdefault("model", model)
            writer.writerow(row)
    return path


def mean_row(rows, key="MEAN"):
    def _mean(k):
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            return None
        return round(float(np.mean(vals)), 4)
    m = {"I": _mean("I"), "P": _mean("P"), "AUPRO": _mean("AUPRO")}
    m["category"] = key
    return m


def round_dict(d, fields=("I", "P", "AUPRO")):
    out = {}
    for k, v in d.items():
        out[k] = round(v, 4) if (k in fields and v is not None) else v
    return out
