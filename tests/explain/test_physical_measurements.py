"""Rigorous quantitative physical measurement verification on calibrated synthetic grids."""

import math
import numpy as np
import pytest
from xmvad.explain import explain_anomaly, ExplainInput


def test_calibrated_circular_depression_dimensions():
    """Verify diameter, area, and depth of a known circular depression within 5% error."""
    # Pixel spacing = 0.5 mm/pixel
    h, w = 100, 100
    spacing = 0.5
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    x_mm = (grid_x - w / 2.0) * spacing
    y_mm = (grid_y - h / 2.0) * spacing
    z_mm = np.ones((h, w), dtype=np.float32) * 60.0

    # Known true values:
    # Radius = 10.0 mm -> Diameter = 20.0 mm
    # True projected Area = pi * r^2 = 314.16 mm^2
    # True depth depression = 2.5 mm
    true_radius_mm = 10.0
    true_depth_mm = 2.5
    r_sq = x_mm**2 + y_mm**2
    defect_mask = r_sq <= true_radius_mm**2

    z_mm[defect_mask] += true_depth_mm

    anomaly_map = np.zeros((h, w), dtype=np.float32)
    anomaly_map[defect_mask] = 0.88

    rgb = np.ones((h, w, 3), dtype=np.uint8) * 200
    valid_mask = np.ones((h, w), dtype=bool)
    xyz = np.stack([x_mm, y_mm, z_mm], axis=-1)

    inp = ExplainInput(
        rgb=rgb,
        xyz=xyz,
        anomaly_map=anomaly_map,
        image_score=0.88,
        category="test_circle",
        sample_id="synth_001",
        valid_xyz_mask=valid_mask,
        coordinate_unit="mm",
        threshold=0.5
    )

    report = explain_anomaly(inp)
    assert report.num_defects == 1
    d = report.defects[0]

    # Verification of major dimension: 20 mm (+/- 5% with grid discretization)
    est_diam = d.size.major_length_mm
    assert abs(est_diam - 20.0) / 20.0 <= 0.051, f"Diameter error > 5%: got {est_diam}"

    # Verification of depth depression: 2.5 mm (+/- 5%)
    est_depth = d.geometry.max_depression
    assert abs(est_depth - 2.5) / 2.5 < 0.05, f"Depth error > 5%: got {est_depth}"

    # Verification of projected area: 314.16 mm^2 (+/- 10% from (19/20)^2 grid effect)
    est_area = d.size.projected_area_mm2
    assert abs(est_area - 314.16) / 314.16 < 0.10, f"Area error > 10%: got {est_area}"


def test_calibrated_rectangular_protrusion():
    """Verify width, height, and protrusion of a known rectangular protrusion within 5% error."""
    h, w = 120, 120
    spacing = 0.4
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    x_mm = grid_x * spacing
    y_mm = grid_y * spacing
    z_mm = np.ones((h, w), dtype=np.float32) * 50.0

    # Defect from pixel (40 to 80 in y) -> height = 40 * 0.4 = 16.0 mm
    # Defect from pixel (50 to 75 in x) -> width = 25 * 0.4 = 10.0 mm
    # True protrusion = 3.2 mm (closer to camera: z = 46.8)
    defect_mask = np.zeros((h, w), dtype=bool)
    defect_mask[40:80, 50:75] = True
    z_mm[defect_mask] -= 3.2

    anomaly_map = np.zeros((h, w), dtype=np.float32)
    anomaly_map[defect_mask] = 0.92

    rgb = np.ones((h, w, 3), dtype=np.uint8) * 180
    valid_mask = np.ones((h, w), dtype=bool)
    xyz = np.stack([x_mm, y_mm, z_mm], axis=-1)

    inp = ExplainInput(
        rgb=rgb,
        xyz=xyz,
        anomaly_map=anomaly_map,
        image_score=0.92,
        category="test_rect",
        sample_id="synth_002",
        valid_xyz_mask=valid_mask,
        coordinate_unit="mm",
        threshold=0.5
    )

    report = explain_anomaly(inp)
    assert report.num_defects == 1
    d = report.defects[0]

    # Major length ~ 16 mm, minor length ~ 10 mm
    assert abs(d.size.major_length_mm - 16.0) / 16.0 < 0.05
    assert abs(d.size.minor_length_mm - 10.0) / 10.0 < 0.05
    assert abs(d.geometry.max_protrusion - 3.2) / 3.2 < 0.05


def test_missing_xyz_graceful_fallback():
    """Verify graceful fallback to 2D-only explanation when XYZ coordinates are NaN or uncalibrated."""
    h, w = 60, 60
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 150
    xyz = np.full((h, w, 3), np.nan, dtype=np.float32)  # Completely missing geometry
    anomaly_map = np.zeros((h, w), dtype=np.float32)
    anomaly_map[20:40, 20:40] = 0.85

    inp = ExplainInput(
        rgb=rgb,
        xyz=xyz,
        anomaly_map=anomaly_map,
        image_score=0.85,
        category="test_fallback",
        sample_id="synth_fallback",
        valid_xyz_mask=np.zeros((h, w), dtype=bool),
        coordinate_unit="unknown",
        threshold=0.5
    )

    report = explain_anomaly(inp)
    assert report.num_defects == 1
    d = report.defects[0]
    assert d.size.physical_units_available is False
    assert d.size.major_length_mm is None
    assert d.size.area_px == 400
    assert "pixels" in d.explanation
