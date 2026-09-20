"""Modality-aware evidence + attribution metrics (Phase 5).

Evidence is computed on E4's branch maps and express RELATIVE anomaly
evidence, NOT causal probability. The operator-facing language is:
"the RGB evidence is stronger than the depth evidence in region R", never
"RGB caused the anomaly".

Definitions
-----------
- Region evidence: ``E_rgb(R)`` / ``E_depth(R)`` = mean or robust top-k of the
  CALIBRATED anomaly values (robust z-scores from validation-normal
  median/MAD, see ``xmvad.fusion.score_fusion.calibrate``) inside region R.
- Positive-clamped relative evidence (softmax-free, sums to one):

    p_rgb = pos(E_rgb) / (pos(E_rgb) + pos(E_depth) + EPS)
    p_depth = 1 - p_rgb        (pos(x) = max(x, 0))

- Output classes: RGB-DOMINANT, DEPTH-DOMINANT, JOINT, UNCERTAIN (see
  :func:`evidence_class`). UNCERTAIN means the region carries no decisive
  evidence (total calibrated evidence at or below ``floor``).

Attribution metrics (controlled benchmark, TRAIN-NORMAL synthetic data):
- dominant-modality accuracy and macro-F1 vs injected modality,
- confusion matrix (predicted modality vs injected modality),
- mean evidence ratio per injected modality/severity,
- AttributionConsistency: agreement between the decision-ablation direction
  (Delta_RGB vs Delta_DEPTH) and the evidence direction (p_rgb vs 0.5).
- Spatial explanation quality: pixel AUROC / AUPRC / Dice@best against
  synthetic masks (and, on real data, GT).

All functions are deterministic and pure; no test data enters any constant.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6

# Evidence-class thresholds (fixed, declared before Phase 5 evaluation).
DOMINANCE_MARGIN = 0.2   # |p_rgb - 0.5| >= margin -> dominant class
EVIDENCE_FLOOR = 0.10    # pos(E_rgb)+pos(E_depth) <= floor -> UNCERTAIN

CLASS_RGB_DOMINANT = "RGB-DOMINANT"
CLASS_DEPTH_DOMINANT = "DEPTH-DOMINANT"
CLASS_JOINT = "JOINT"
CLASS_UNCERTAIN = "UNCERTAIN"
EVIDENCE_CLASSES = (CLASS_RGB_DOMINANT, CLASS_DEPTH_DOMINANT, CLASS_JOINT, CLASS_UNCERTAIN)


def robust_topk(values: np.ndarray, frac: float = 0.01, min_pixels: int = 1) -> float:
    """Mean of the top ``frac`` of *total pixels*; ``min_pixels`` lower bound."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    k = int(np.clip(int(round(frac * v.size)), min_pixels, v.size))
    return float(np.mean(np.sort(v)[-k:]))


def region_evidence(amap: np.ndarray, region: np.ndarray,
                    mode: str = "topk", frac: float = 0.01) -> float:
    """Evidence of one modality inside boolean ``region``.

    ``amap`` must be the CALIBRATED anomaly map (z-scores). ``mode`` is
    ``'topk'`` (robust top-``frac`` mean, default) or ``'mean'``.
    Empty region -> 0.0 (no evidence).
    """
    region = np.asarray(region, dtype=bool)
    vals = np.asarray(amap, dtype=np.float64)[region]
    if vals.size == 0:
        return 0.0
    if mode == "mean":
        return float(np.mean(vals[np.isfinite(vals)])) if np.any(np.isfinite(vals)) else 0.0
    if mode == "topk":
        return robust_topk(vals, frac=frac)
    raise ValueError(f"unknown mode {mode!r}")


def relative_evidence(e_rgb: float, e_depth: float) -> tuple[float, float]:
    """Positive-clamped relative evidence (sums to one).

    When both modalities carry no positive evidence (both raw values <= 0),
    returns (0.5, 0.5): equally uninformative.
    """
    pos_r = float(max(0.0, float(e_rgb)))
    pos_d = float(max(0.0, float(e_depth)))
    if pos_r == 0.0 and pos_d == 0.0:
        return 0.5, 0.5
    s = pos_r + pos_d + EPS
    return pos_r / s, pos_d / s


def normalize_evidence(e_rgb: float, e_depth: float) -> tuple[float, float]:
    """Elementwise [0, 1] dominance normalization by the stronger modality."""
    pos_r = float(max(0.0, float(e_rgb)))
    pos_d = float(max(0.0, float(e_depth)))
    m = max(pos_r, pos_d)
    if m <= 0.0:
        return 0.0, 0.0
    return pos_r / m, pos_d / m


def evidence_class(e_rgb: float, e_depth: float,
                   margin: float = DOMINANCE_MARGIN,
                   floor: float = EVIDENCE_FLOOR) -> str:
    """Label one region as RGB/DEPTH-DOMINANT, JOINT or UNCERTAIN.

    "Relative anomaly evidence", not a causal claim. UNCERTAIN when the
    region carries no decisive evidence (total positive evidence < floor).
    """
    p_rgb, _ = relative_evidence(e_rgb, e_depth)
    total = float(max(0.0, float(e_rgb))) + float(max(0.0, float(e_depth)))
    if total <= floor:
        return CLASS_UNCERTAIN
    if p_rgb >= 0.5 + margin:
        return CLASS_RGB_DOMINANT
    if p_rgb <= 0.5 - margin:
        return CLASS_DEPTH_DOMINANT
    return CLASS_JOINT


def evidence_margin(e_rgb: float, e_depth: float) -> dict[str, float]:
    """Margins for reporting: dominance distance and evidence gap."""
    p_rgb, _ = relative_evidence(e_rgb, e_depth)
    return {"dominance_margin": float(abs(p_rgb - 0.5)),
            "p_rgb": float(p_rgb),
            "evidence_gap": float(abs(max(0.0, float(e_rgb)) - max(0.0, float(e_depth))))}


def decision_ablation(s_full: float, s_rgb_only: float, s_depth_only: float) -> dict[str, float]:
    """Delta from dropping each modality.

    S_full uses the E4 fused calibrated map; S_rgb_only / S_depth_only use the
    calibrated RGB / depth branch maps only. Deltas on image-level scores:

        Delta_RGB   = S_full - S_depth_only   (value lost if RGB is removed)
        Delta_DEPTH = S_full - S_rgb_only     (value lost if depth is removed)

    Documented limitation: E4 fuses features before scoring, so the deltas are
    NOT perfectly additive; they measure removal sensitivity, not pure
    causation.
    """
    s_full = float(s_full)
    d_r = s_full - float(s_depth_only)
    d_d = s_full - float(s_rgb_only)
    return {"S_full": s_full, "S_rgb_only": float(s_rgb_only), "S_depth_only": float(s_depth_only),
            "Delta_RGB": d_r, "Delta_DEPTH": d_d}


def attribution_consistency(p_rgb: float, delta_rgb: float, delta_depth: float,
                            eps: float = 1e-9) -> bool:
    """Evidence direction agrees with decision-ablation direction.

    True when (p_rgb - 0.5) and (Delta_RGB - Delta_DEPTH) share a sign
    (either both positive RGB-leaning or both negative depth-leaning); near-tie
    evidence (|p_rgb-0.5| <= eps) counts as consistent with a near-tie decision
    (|Delta_RGB-Delta_DEPTH| <= eps).
    """
    ev = float(p_rgb) - 0.5
    s = float(delta_rgb) - float(delta_depth)
    if abs(ev) <= eps:
        return True  # flat evidence cannot contradict either direction
    return (ev > 0) == (s > 0)


def confusion_metrics(y_true, y_pred, labels=EVIDENCE_CLASSES) -> dict:
    """Dominant-modality accuracy, macro-F1, and confusion matrix."""
    from sklearn.metrics import confusion_matrix, f1_score

    y_true = [str(t) for t in y_true]
    y_pred = [str(p) for p in y_pred]
    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    present = [l for l in labels if l in y_true]
    macro_f1 = float(f1_score(y_true, y_pred, labels=present, average="macro",
                              zero_division=0.0))
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return {"dominant_accuracy": acc, "macro_f1": macro_f1,
            "confusion": cm, "labels": list(labels)}


def mean_evidence_ratio(e_rgb_list, e_depth_list) -> dict[str, float]:
    """Per-group mean evidence ratio pos(E_rgb)/pos(E_depth)."""
    r = np.asarray([max(0.0, float(a)) for a in e_rgb_list], dtype=np.float64)
    d = np.asarray([max(0.0, float(b)) for b in e_depth_list], dtype=np.float64)
    return {"mean_ratio": float(np.mean(r / (d + EPS))) if len(r) else 0.0,
            "mean_rgb": float(np.mean(r)) if len(r) else 0.0,
            "mean_depth": float(np.mean(d)) if len(d) else 0.0}


def spatial_explanation_metrics(pred_map: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Pixel AUROC/AUPRC + Dice@best-IoU-threshold vs a mask (synthetic/GT)."""
    from xmvad.metrics.localization import dice_at_threshold, iou_at_threshold
    from xmvad.metrics.localization import pixel_auprc, pixel_auroc

    pa = pixel_auroc([np.asarray(gt_mask, dtype=bool)], [np.asarray(pred_map, dtype=float)])
    pp = pixel_auprc([np.asarray(gt_mask, dtype=bool)], [np.asarray(pred_map, dtype=float)])
    pr = np.asarray(pred_map, dtype=float)
    best_t, best_dice, best_iou = 0.0, 0.0, 0.0
    if np.asarray(gt_mask, dtype=bool).any() and pr.size:
        lo, hi = float(pr.min()), float(pr.max())
        ts = np.linspace(lo, hi, 51) if hi > lo else np.array([lo])
        for t in ts:
            d = dice_at_threshold(gt_mask, pr, float(t))
            if d >= best_dice:
                best_dice, best_t = float(d), float(t)
    best_iou = iou_at_threshold(gt_mask, pr, best_t)
    return {"pixel_auroc": pa, "pixel_auprc": pp,
            "dice_best": best_dice, "iou_best": best_iou, "best_threshold": best_t}