"""Depth-unit tests (pure synthetic arrays; no dataset needed)."""

from __future__ import annotations

import numpy as np
import pytest

from xmvad.data.depth import (
    compute_robust_stats,
    depth_range,
    depth_to_uint8,
    extract_depth,
    fit_depth_stats,
    invalid_fraction,
    normalize_depth,
    valid_mask_from_xyz,
)


def _xyz(h=8, w=10, z_val=0.7) -> np.ndarray:
    xyz = np.ones((h, w, 3), dtype=np.float32)
    xyz[..., 2] = z_val
    xyz[0, :, :] = 0.0  # invalid row
    return xyz


def test_valid_mask_marks_zero_sentinel() -> None:
    xyz = _xyz()
    valid = valid_mask_from_xyz(xyz)
    assert valid.shape == (8, 10)
    assert not valid[0].any()
    assert valid[1:].all()
    assert invalid_fraction(valid) == pytest.approx(1.0 / 8.0)


def test_valid_mask_marks_nonfinite() -> None:
    xyz = _xyz()
    xyz[3, 3, 1] = np.nan
    xyz[4, 4, 0] = np.inf
    valid = valid_mask_from_xyz(xyz)
    assert not valid[3, 3]
    assert not valid[4, 4]


def test_extract_depth_is_z() -> None:
    xyz = _xyz(z_val=1.25)
    d = extract_depth(xyz)
    assert d.shape == (8, 10)
    assert np.allclose(d[1:, :], 1.25)


def test_robust_stats_and_normalize() -> None:
    vals = np.array([0.5, 0.5, 0.5, 0.6, 10.0], dtype=np.float64)  # outlier-robust
    stats = compute_robust_stats(vals, "robust_median")
    assert abs(stats["median"] - 0.5) < 1e-9
    depth = np.full((4, 4), 0.5, dtype=np.float32)
    valid = np.ones((4, 4), dtype=bool)
    out = normalize_depth(depth, stats, valid_mask=valid, clip=(-3, 3))
    assert np.allclose(out, 0.0, atol=1e-5)
    # invalid pixels forced to 0, not to a "meaningful" value
    valid[0, 0] = False
    depth[0, 0] = 999.0
    out = normalize_depth(depth, stats, valid_mask=valid, clip=(-3, 3))
    assert out[0, 0] == 0.0


def test_percentile_stats_and_normalize() -> None:
    vals = np.linspace(0, 1, 101)
    stats = compute_robust_stats(vals, "percentile")
    assert stats["p5"] < stats["p50"] < stats["p95"]
    out = normalize_depth(np.array([[0.5]], dtype=np.float32), stats, valid_mask=None, clip=None)
    assert np.isfinite(out).all()


def test_fit_depth_stats_caps_pixels() -> None:
    pools = [np.linspace(0, 1, 10000) for _ in range(5)]
    stats = fit_depth_stats(pools, max_pixels=1000, seed=0)
    assert "median" in stats or "p50" in stats


def test_depth_range_and_uint8() -> None:
    xyz = _xyz(z_val=2.0)
    d = extract_depth(xyz)
    v = valid_mask_from_xyz(xyz)
    dmin, dmax = depth_range(d, v)
    assert dmin == 2.0 and dmax == 2.0
    u8 = depth_to_uint8(d, v)
    assert u8.dtype == np.uint8 and u8[0, 0] == 0  # invalid rendered black
    # empty-valid edge case
    dmin2, dmax2 = depth_range(d, np.zeros_like(v))
    assert np.isnan(dmin2) and np.isnan(dmax2)
    u8b = depth_to_uint8(d, np.zeros_like(v))
    assert (u8b == 0).all()
