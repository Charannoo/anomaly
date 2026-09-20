"""Tests for local surface analysis and signed deviation classification."""

import numpy as np
import pytest
from xmvad.explain.surface import analyze_local_surface


def test_localized_depression_surface():
    h, w = 80, 80
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    # Flat horizontal plane at z = 50.0 mm
    x_mm = grid_x * 0.5
    y_mm = grid_y * 0.5
    z_mm = np.ones((h, w), dtype=np.float32) * 50.0

    # Defect region in center: recessed depression of depth 2.0 mm (further away: z = 52.0)
    defect_mask = np.zeros((h, w), dtype=bool)
    defect_mask[30:50, 30:50] = True
    z_mm[defect_mask] += 2.0

    anomaly_map = np.zeros((h, w), dtype=np.float32)
    anomaly_map[defect_mask] = 0.85
    valid_mask = np.ones((h, w), dtype=bool)
    xyz = np.stack([x_mm, y_mm, z_mm], axis=-1)

    geo = analyze_local_surface(
        component_mask=defect_mask,
        anomaly_map=anomaly_map,
        scaled_xyz=xyz,
        valid_xyz_mask=valid_mask,
        is_physical_calibrated=True,
        threshold=0.5,
        unit="mm"
    )

    assert "depression" in geo.label
    assert geo.max_depression is not None
    assert 1.8 <= geo.max_depression <= 2.2
    assert geo.mean_deviation < -1.5


def test_raised_protrusion_surface():
    h, w = 80, 80
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    x_mm = grid_x * 0.5
    y_mm = grid_y * 0.5
    z_mm = np.ones((h, w), dtype=np.float32) * 50.0

    # Defect region protruding upwards towards camera by 3.5 mm (closer: z = 46.5)
    defect_mask = np.zeros((h, w), dtype=bool)
    defect_mask[30:50, 30:50] = True
    z_mm[defect_mask] -= 3.5

    anomaly_map = np.zeros((h, w), dtype=np.float32)
    anomaly_map[defect_mask] = 0.90
    valid_mask = np.ones((h, w), dtype=bool)
    xyz = np.stack([x_mm, y_mm, z_mm], axis=-1)

    geo = analyze_local_surface(
        component_mask=defect_mask,
        anomaly_map=anomaly_map,
        scaled_xyz=xyz,
        valid_xyz_mask=valid_mask,
        is_physical_calibrated=True,
        threshold=0.5,
        unit="mm"
    )

    assert "protrusion" in geo.label
    assert geo.max_protrusion is not None
    assert 3.2 <= geo.max_protrusion <= 3.8
    assert geo.mean_deviation > 2.5


def test_appearance_only_flat_surface():
    h, w = 80, 80
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    x_mm = grid_x * 0.5
    y_mm = grid_y * 0.5
    z_mm = np.ones((h, w), dtype=np.float32) * 50.0  # Perfect flat surface, zero depth change

    defect_mask = np.zeros((h, w), dtype=bool)
    defect_mask[30:50, 30:50] = True

    anomaly_map = np.zeros((h, w), dtype=np.float32)
    anomaly_map[defect_mask] = 0.75
    valid_mask = np.ones((h, w), dtype=bool)
    xyz = np.stack([x_mm, y_mm, z_mm], axis=-1)

    geo = analyze_local_surface(
        component_mask=defect_mask,
        anomaly_map=anomaly_map,
        scaled_xyz=xyz,
        valid_xyz_mask=valid_mask,
        is_physical_calibrated=True,
        threshold=0.5,
        unit="mm"
    )

    assert geo.label == "primarily appearance-based anomaly"
    assert abs(geo.mean_deviation) < 0.1
