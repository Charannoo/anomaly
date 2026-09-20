"""Image-score readouts from anomaly maps (shared by evaluation + fusion).

Variants (reported side-by-side; never silently pick the winner):
- ``max``: maximum map value (primary protocol; official MVTec code uses max).
- ``topk_0.1pct`` / ``topk_1pct``: mean of the top 0.1% / 1% of TOTAL map
  pixels. Invalid depth pixels are excluded via -inf masking (same semantics
  as the model's own score); non-finite top-k entries fall back to 0.
"""

from __future__ import annotations

import numpy as np

SCORE_VARIANTS: tuple[tuple[str, float], ...] = (
    ("max", 0.0),
    ("topk_0.1pct", 0.001),
    ("topk_1pct", 0.01),
)


def variant_scores(amap: np.ndarray, valid: np.ndarray | None) -> dict[str, float]:
    """Compute all score variants for one anomaly map."""
    flat = np.asarray(amap, dtype=float).ravel()
    if valid is not None:
        flat = np.where(np.asarray(valid, dtype=bool).ravel(), flat, -np.inf)
    out: dict[str, float] = {}
    finite = flat[np.isfinite(flat)]
    out["max"] = float(finite.max()) if finite.size else 0.0
    for name, frac in SCORE_VARIANTS[1:]:
        k = max(1, int(round(frac * flat.size)))
        topk = np.sort(flat)[-k:]
        topk = np.where(np.isfinite(topk), topk, 0.0)
        out[name] = float(topk.mean())
    return out
