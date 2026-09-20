"""Tests for coordinate unit detection and physical scaling."""

import numpy as np
import pytest
from xmvad.explain.units import CoordinateUnitManager


def test_explicit_mm_unit():
    mgr = CoordinateUnitManager(declared_unit="mm")
    xyz = np.ones((50, 50, 3), dtype=np.float32) * 50.0
    scaled, unit, factor, calibrated = mgr.validate_and_scale(xyz)
    assert unit == "mm"
    assert factor == 1.0
    assert calibrated is True
    assert np.allclose(scaled, xyz)


def test_explicit_m_unit_scaling():
    mgr = CoordinateUnitManager(declared_unit="m")
    # Coordinates in meters: 0.05m = 50mm
    xyz = np.ones((50, 50, 3), dtype=np.float32) * 0.05
    scaled, unit, factor, calibrated = mgr.validate_and_scale(xyz)
    assert unit == "mm"
    assert factor == 1000.0
    assert calibrated is True
    assert np.allclose(scaled, 50.0)


def test_unknown_auto_detection_meters():
    mgr = CoordinateUnitManager(declared_unit="unknown")
    # Industrial part extent ~0.15m
    grid_y, grid_x = np.mgrid[0:100, 0:100]
    x = grid_x * 0.0015
    y = grid_y * 0.0015
    z = np.ones_like(x) * 0.5
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    
    scaled, unit, factor, calibrated = mgr.validate_and_scale(xyz)
    assert unit == "mm"
    assert factor == 1000.0
    assert calibrated is True
    assert np.isclose(scaled[99, 99, 0], 148.5, atol=1e-2)


def test_unknown_auto_detection_uncalibrated():
    mgr = CoordinateUnitManager(declared_unit="unknown")
    # Normalized coordinates in [0, 1] with arbitrary non-metric frame
    xyz = np.random.uniform(-5000.0, 5000.0, size=(50, 50, 3)).astype(np.float32)
    scaled, unit, factor, calibrated = mgr.validate_and_scale(xyz)
    assert calibrated is False
    assert unit == "unknown"
