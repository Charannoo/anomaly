"""Tests for 2D morphology computation and shape descriptors."""

import numpy as np
import pytest
from xmvad.explain.morphology2d import compute_2d_morphology


def test_circular_disk_morphology():
    mask = np.zeros((100, 100), dtype=bool)
    grid_y, grid_x = np.mgrid[0:100, 0:100]
    # Circle with radius 15
    mask[(grid_x - 50)**2 + (grid_y - 50)**2 <= 15**2] = True

    metrics, label = compute_2d_morphology(mask)
    assert metrics["area_px"] > 600
    assert 0.90 <= metrics["aspect_ratio"] <= 1.25
    assert metrics["circularity"] >= 0.70
    assert metrics["solidity"] >= 0.90
    assert "circular" in label or "rounded" in label


def test_elongated_stripe_morphology():
    mask = np.zeros((100, 100), dtype=bool)
    # Thin rectangle 50x6 (aspect ratio ~8.3)
    mask[20:70, 45:51] = True

    metrics, label = compute_2d_morphology(mask)
    assert metrics["aspect_ratio"] >= 4.0
    assert metrics["circularity"] <= 0.45
    assert "elongated" in label or "stripe" in label


def test_irregular_shape_solidity():
    mask = np.zeros((100, 100), dtype=bool)
    # L-shaped polygon
    mask[20:80, 20:35] = True
    mask[65:80, 20:80] = True

    metrics, label = compute_2d_morphology(mask)
    # Convex hull area is significantly larger than actual L area -> lower solidity
    assert metrics["solidity"] <= 0.75
    assert "irregular" in label
