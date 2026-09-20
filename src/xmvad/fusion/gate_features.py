"""Gate input features (Phase 3C): compact statistics of frozen-branch outputs.

27 features: 10 RGB branch stats + 10 depth branch stats (over evidence
pixels) + 4 cross-modal stats + 3 depth-coverage stats. Normalization uses
gate_train statistics only (persisted separately).
"""

from __future__ import annotations

import numpy as np
import torch

from xmvad.fusion.score_fusion import EPS
from xmvad.losses.distillation import downsample_valid_mask

FEATURE_NAMES = (
    [f"rgb_{n}" for n in ("max", "top01", "top1", "p95", "p99", "mean", "std",
                          "frac_gt2", "frac_gt3", "frac_gt5")]
    + [f"depth_{n}" for n in ("max", "top01", "top1", "p95", "p99", "mean", "std",
                              "frac_gt2", "frac_gt3", "frac_gt5")]
    + ["abs_max_diff", "map_corr", "high_iou", "log_area_ratio",
       "valid_frac", "stage8_valid", "stage16_valid"]
)


def branch_stats(z: np.ndarray, ev_mask: np.ndarray | None) -> list[float]:
    """10 summary stats of a calibrated map over evidence pixels."""
    v = np.asarray(z, dtype=np.float64).ravel() if ev_mask is None else np.asarray(
        z, dtype=np.float64)[np.asarray(ev_mask, dtype=bool)]
    if v.size == 0:
        return [0.0] * 10
    k01 = max(1, int(round(0.001 * v.size)))
    k1 = max(1, int(round(0.01 * v.size)))
    s = np.sort(v)
    return [float(v.max()), float(s[-k01:].mean()), float(s[-k1:].mean()),
            float(np.percentile(v, 95)), float(np.percentile(v, 99)),
            float(v.mean()), float(v.std()),
            float((v > 2).mean()), float((v > 3).mean()), float((v > 5).mean())]


def gate_features(zr: np.ndarray, zd: np.ndarray, valid: np.ndarray) -> list[float]:
    """27-dim feature vector from calibrated maps + validity mask."""
    feats = branch_stats(zr, None) + branch_stats(zd, valid if valid.any() else None)
    fr = feats[0]
    fd = feats[10]
    a = zr.ravel().astype(np.float64)
    b = zd.ravel().astype(np.float64)
    vm = valid.ravel().astype(bool)
    if vm.sum() >= 10 and np.std(a[vm]) > 0 and np.std(b[vm]) > 0:
        corr = float(np.corrcoef(a[vm], b[vm])[0, 1])
    else:
        corr = 0.0
    hr = (zr > 3)
    hd = (zd > 3) & valid
    union = (hr | hd).sum()
    iou = float((hr & hd).sum() / union) if union else 0.0
    frr = float(hr.mean())
    frd = float(hd[valid].mean()) if valid.any() else 0.0
    log_ratio = float(np.log((frr + EPS) / (frd + EPS)))
    t = torch.from_numpy(valid).unsqueeze(0).unsqueeze(0)
    s8 = float(downsample_valid_mask(t, (zr.shape[0] // 8, zr.shape[1] // 8)).float().mean())
    s16 = float(downsample_valid_mask(t, (zr.shape[0] // 16, zr.shape[1] // 16)).float().mean())
    feats += [abs(fr - fd), corr, iou, log_ratio, float(valid.mean()), s8, s16]
    return [float(np.nan_to_num(x)) for x in feats]
