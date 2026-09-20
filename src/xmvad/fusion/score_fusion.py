"""Score-level reliability fusion rules (Phase 3B, pre-registered).

All rules fuse CALIBRATED maps (Z-scores from validation-normal median/MAD).
Depth-invalid pixels always fall back to RGB evidence only. Every formula is
explicit; no test data enters any parameter. Weights always sum to one and
are finite (guarded by eps).

Rules:
- E3B-1 static: per-category base weights from normal-state noise.
- E3B-2 validity-adjusted: E3B-1 base x per-image depth-valid fraction.
- E3B-3 evidence-adaptive: per-image softmax over calibrated evidence,
  fixed T=1. Labeled "evidence-adaptive", NOT learned.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6
TEMPERATURE = 1.0  # fixed, declared before any test evaluation


def calibrate(amap: np.ndarray, median: float, scale: float) -> np.ndarray:
    """Robust z-calibration (no clamping; finite guarded)."""
    z = (np.asarray(amap, dtype=np.float64) - float(median)) / (float(scale) + EPS)
    return np.where(np.isfinite(z), z, 0.0)


def static_weights(noise_rgb: float, noise_depth: float) -> tuple[float, float]:
    """E3B-1: base reliability weights from normal-state noise levels.

    base_m = 1 / (noise_m + eps), normalized to sum to one.
    """
    b_r = 1.0 / (float(noise_rgb) + EPS)
    b_d = 1.0 / (float(noise_depth) + EPS)
    s = b_r + b_d
    if s <= 0.0 or not np.isfinite(s):
        return 0.5, 0.5
    return b_r / s, b_d / s


def validity_adjusted_weights(base_rgb: float, base_depth: float,
                              valid_frac: float) -> tuple[float, float]:
    """E3B-2: scale depth reliability by per-image valid-depth fraction."""
    q = float(np.clip(valid_frac, 0.0, 1.0))
    r_r = float(base_rgb)
    r_d = float(base_depth) * q
    s = r_r + r_d
    if s <= 0.0 or not np.isfinite(s):
        return 0.5, 0.5
    return r_r / s, r_d / s


def evidence_weights(e_rgb: float, e_depth: float,
                     temperature: float = TEMPERATURE) -> tuple[float, float]:
    """E3B-3: softmax over non-negative calibrated evidence, fixed T."""
    e = np.array([max(0.0, float(e_rgb)), max(0.0, float(e_depth))]) / float(temperature)
    e = e - e.max()
    w = np.exp(e)
    s = w.sum()
    if s <= 0.0 or not np.isfinite(s):
        return 0.5, 0.5
    w = w / s
    return float(w[0]), float(w[1])


def fuse_maps(z_rgb: np.ndarray, z_depth: np.ndarray, valid: np.ndarray,
              w_rgb: float, w_depth: float) -> np.ndarray:
    """Weighted fusion with RGB-only fallback at depth-invalid pixels."""
    z_rgb = np.asarray(z_rgb, dtype=np.float64)
    z_depth = np.asarray(z_depth, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    fused = w_rgb * z_rgb + w_depth * z_depth
    out = np.where(v, fused, z_rgb)
    return np.where(np.isfinite(out), out, 0.0)
