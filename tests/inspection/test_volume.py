"""Synthetic analytical validation and edge case tests for Material Loss / Excess Volume Estimation."""

import pytest
import numpy as np

from xmvad.inspection.volume import quantify_defect_volume


def test_synthetic_circular_depression_volume():
    """Validate estimated missing volume of a calibrated circular cylinder depression (r=10mm, d=2mm).
    
    Analytical Volume:
        V = pi * r^2 * d = pi * 10^2 * 2.0 = 628.3185 mm^3.
    Target:
        Volume error < 5% under 0.5 mm spatial grid.
    """
    h, w = 120, 120
    spacing = 0.5  # mm / pixel
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    x_mm = (grid_x - w / 2.0) * spacing
    y_mm = (grid_y - h / 2.0) * spacing
    z_mm = np.ones((h, w), dtype=np.float32) * 50.0

    radius_mm = 10.0
    true_depth_mm = 2.0
    r_sq = x_mm**2 + y_mm**2
    defect_mask = r_sq <= radius_mm**2

    # Since camera looks from z=0, receding into the object increases z
    # Outward normal points towards camera, so recessed points have negative displacement
    z_mm[defect_mask] += true_depth_mm

    xyz = np.stack([x_mm, y_mm, z_mm], axis=-1)
    valid_mask = np.ones((h, w), dtype=bool)
    anomaly_map = np.zeros((h, w), dtype=np.float32)
    anomaly_map[defect_mask] = 0.85

    vol = quantify_defect_volume(
        component_mask=defect_mask,
        anomaly_map=anomaly_map,
        scaled_xyz=xyz,
        valid_xyz_mask=valid_mask,
        is_physical_calibrated=True,
        threshold=0.5,
        unit="mm"
    )

    true_volume = np.pi * (radius_mm**2) * true_depth_mm  # 628.3185 mm^3
    rel_error = abs(vol.missing_material_volume - true_volume) / true_volume

    print(f"Circular Depression: Estimated = {vol.missing_material_volume:.2f} mm3, True = {true_volume:.2f} mm3, Error = {rel_error*100:.2f}%")
    assert rel_error < 0.05, f"Volume error exceeded 5%: got {rel_error*100:.2f}%"
    assert vol.physical_volume_available is True
    assert vol.volume_unit == "mm3"
    assert vol.excess_material_volume < 5.0


def test_synthetic_rectangular_protrusion_volume():
    """Validate estimated excess volume of a rectangular protrusion (10mm x 10mm x 2mm).
    
    Analytical Volume:
        V = 10.0 * 10.0 * 2.0 = 200.0 mm^3.
    Target:
        Volume error < 5%.
    """
    h, w = 100, 100
    spacing = 0.5  # mm / pixel
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    # Pixel-centered coordinates
    x_mm = (grid_x - w / 2.0 + 0.5) * spacing
    y_mm = (grid_y - h / 2.0 + 0.5) * spacing
    z_mm = np.ones((h, w), dtype=np.float32) * 50.0

    # Exactly 10 mm x 10 mm rectangle (20 x 20 pixels) centered at (0, 0)
    defect_mask = (np.abs(x_mm) <= 5.0) & (np.abs(y_mm) <= 5.0)
    true_height_mm = 2.0
    
    # Protruding towards camera decreases z
    z_mm[defect_mask] -= true_height_mm

    xyz = np.stack([x_mm, y_mm, z_mm], axis=-1)
    valid_mask = np.ones((h, w), dtype=bool)
    anomaly_map = np.zeros((h, w), dtype=np.float32)
    anomaly_map[defect_mask] = 0.90

    vol = quantify_defect_volume(
        component_mask=defect_mask,
        anomaly_map=anomaly_map,
        scaled_xyz=xyz,
        valid_xyz_mask=valid_mask,
        is_physical_calibrated=True,
        threshold=0.5,
        unit="mm"
    )

    true_volume = 10.0 * 10.0 * true_height_mm  # 200.0 mm^3
    rel_error = abs(vol.excess_material_volume - true_volume) / true_volume

    print(f"Rectangular Protrusion: Estimated = {vol.excess_material_volume:.2f} mm3, True = {true_volume:.2f} mm3, Error = {rel_error*100:.2f}%")
    assert rel_error < 0.05, f"Volume error exceeded 5%: got {rel_error*100:.2f}%"
    assert vol.missing_material_volume < 5.0
    assert vol.net_signed_volume > 0


def test_mixed_deformation_volume():
    """Verify that mixed depression and protrusion regions report separate volumes rather than a misleading single number."""
    h, w = 100, 100
    spacing = 0.5
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    x_mm = (grid_x - w / 2.0) * spacing
    y_mm = (grid_y - h / 2.0) * spacing
    z_mm = np.ones((h, w), dtype=np.float32) * 50.0

    # Left half: depression (depth 2 mm), Right half: protrusion (height 1.5 mm)
    dep_mask = (x_mm >= -8.0) & (x_mm < 0.0) & (np.abs(y_mm) <= 6.0)
    prot_mask = (x_mm >= 0.0) & (x_mm <= 8.0) & (np.abs(y_mm) <= 6.0)
    combined_mask = dep_mask | prot_mask

    z_mm[dep_mask] += 2.0
    z_mm[prot_mask] -= 1.5

    xyz = np.stack([x_mm, y_mm, z_mm], axis=-1)
    valid_mask = np.ones((h, w), dtype=bool)
    anomaly_map = np.zeros((h, w), dtype=np.float32)
    anomaly_map[combined_mask] = 0.88

    vol = quantify_defect_volume(
        component_mask=combined_mask,
        anomaly_map=anomaly_map,
        scaled_xyz=xyz,
        valid_xyz_mask=valid_mask,
        is_physical_calibrated=True,
        threshold=0.5,
        unit="mm"
    )

    assert vol.missing_material_volume > 20.0, "Missing volume should be detected"
    assert vol.excess_material_volume > 15.0, "Excess volume should be detected"
    # Net signed volume = excess - missing
    assert np.isclose(vol.net_signed_volume, vol.excess_material_volume - vol.missing_material_volume, atol=0.01)
    assert "Mixed geometric deformation" in vol.explanation


def test_uncalibrated_units_fallback():
    """Strict unit rule: verify mm3 is suppressed when physical calibration is absent."""
    h, w = 60, 60
    mask = np.zeros((h, w), dtype=bool)
    mask[20:40, 20:40] = True
    xyz = np.ones((h, w, 3), dtype=np.float32) * 20.0
    valid = np.ones((h, w), dtype=bool)
    anomaly = np.zeros((h, w), dtype=np.float32)
    anomaly[mask] = 0.8

    vol = quantify_defect_volume(
        component_mask=mask,
        anomaly_map=anomaly,
        scaled_xyz=xyz,
        valid_xyz_mask=valid,
        is_physical_calibrated=False,  # Uncalibrated!
        threshold=0.5,
        unit="unknown"
    )

    assert vol.physical_volume_available is False
    assert vol.volume_unit == "native_units3"
    assert "mm³" not in vol.volume_unit
    assert "withheld because coordinate units are uncalibrated" in vol.explanation


def test_missing_xyz_fallback():
    """Verify graceful handling when defect has sparse or missing 3D points."""
    h, w = 50, 50
    mask = np.zeros((h, w), dtype=bool)
    mask[10:20, 10:20] = True
    xyz = np.zeros((h, w, 3), dtype=np.float32)
    valid = np.zeros((h, w), dtype=bool)  # No valid XYZ points
    anomaly = np.zeros((h, w), dtype=np.float32)
    anomaly[mask] = 0.9

    vol = quantify_defect_volume(
        component_mask=mask,
        anomaly_map=anomaly,
        scaled_xyz=xyz,
        valid_xyz_mask=valid,
        is_physical_calibrated=True,
        threshold=0.5,
        unit="mm"
    )

    assert vol.valid_point_count == 0
    assert vol.measurement_confidence == 0.0
    assert vol.missing_material_volume == 0.0


def test_synthetic_gaussian_depression_volume():
    """Validate estimated missing volume of a Gaussian depression.
    
    Analytical Volume:
        V = 2 * pi * A * sigma^2 = 2 * pi * 2.0 * 16.0 = 201.06 mm^3.
    """
    h, w = 120, 120
    spacing = 0.5
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    x_mm = (grid_x - w / 2.0 + 0.5) * spacing
    y_mm = (grid_y - h / 2.0 + 0.5) * spacing
    z_mm = np.ones((h, w), dtype=np.float32) * 50.0

    amp = 2.0
    sigma = 4.0
    r_sq = x_mm**2 + y_mm**2
    # Truncate at 3*sigma = 12 mm
    defect_mask = r_sq <= (3.0 * sigma)**2
    
    # Depression shape: z increases away from camera
    depression = amp * np.exp(-r_sq / (2.0 * sigma**2))
    z_mm[defect_mask] += depression[defect_mask]

    xyz = np.stack([x_mm, y_mm, z_mm], axis=-1)
    valid_mask = np.ones((h, w), dtype=bool)
    anomaly_map = np.zeros((h, w), dtype=np.float32)
    anomaly_map[defect_mask] = 0.85

    vol = quantify_defect_volume(
        component_mask=defect_mask,
        anomaly_map=anomaly_map,
        scaled_xyz=xyz,
        valid_xyz_mask=valid_mask,
        is_physical_calibrated=True,
        threshold=0.5,
        unit="mm"
    )

    # Numerical ground truth over discrete grid:
    true_numerical = float(np.sum(depression[defect_mask] * (spacing ** 2)))
    analytical_infinite = 2.0 * np.pi * amp * (sigma ** 2)  # 201.06 mm3
    rel_error = abs(vol.missing_material_volume - true_numerical) / true_numerical

    print(f"Gaussian Depression: Estimated = {vol.missing_material_volume:.2f} mm3, True Numerical = {true_numerical:.2f} mm3 (Inf: {analytical_infinite:.2f}), Error = {rel_error*100:.2f}%")
    assert rel_error < 0.05, f"Gaussian error exceeded 5%: got {rel_error*100:.2f}%"


def test_synthetic_spherical_cap_indentation_volume():
    """Validate estimated volume of a spherical-cap indentation.
    
    Spherical cap: Sphere R=30mm, indentation depth h=2mm.
    Analytical Volume: V = pi * h^2 / 3 * (3R - h) = pi * 4 / 3 * 88 = 368.61 mm^3.
    """
    h_dim, w_dim = 120, 120
    spacing = 0.5
    grid_y, grid_x = np.mgrid[0:h_dim, 0:w_dim]
    x_mm = (grid_x - w_dim / 2.0 + 0.5) * spacing
    y_mm = (grid_y - h_dim / 2.0 + 0.5) * spacing
    z_mm = np.ones((h_dim, w_dim), dtype=np.float32) * 50.0

    R = 30.0
    h = 2.0
    # Base radius a = sqrt(2Rh - h^2) = sqrt(116) = 10.77 mm
    a = np.sqrt(2.0 * R * h - h**2)
    r_sq = x_mm**2 + y_mm**2
    defect_mask = r_sq <= a**2

    # Spherical indentation: z = z0 + (R - sqrt(R^2 - r^2))
    cap_depth = (R - np.sqrt(np.maximum(0.0, R**2 - r_sq)))
    z_mm[defect_mask] += cap_depth[defect_mask]

    xyz = np.stack([x_mm, y_mm, z_mm], axis=-1)
    valid_mask = np.ones((h_dim, w_dim), dtype=bool)
    anomaly_map = np.zeros((h_dim, w_dim), dtype=np.float32)
    anomaly_map[defect_mask] = 0.88

    vol = quantify_defect_volume(
        component_mask=defect_mask,
        anomaly_map=anomaly_map,
        scaled_xyz=xyz,
        valid_xyz_mask=valid_mask,
        is_physical_calibrated=True,
        threshold=0.5,
        unit="mm"
    )

    true_cap_vol = (np.pi * (h**2) / 3.0) * (3.0 * R - h)  # 368.61 mm3
    rel_error = abs(vol.missing_material_volume - true_cap_vol) / true_cap_vol

    print(f"Spherical Cap: Estimated = {vol.missing_material_volume:.2f} mm3, True = {true_cap_vol:.2f} mm3, Error = {rel_error*100:.2f}%")
    assert rel_error < 0.05, f"Spherical cap error exceeded 5%: got {rel_error*100:.2f}%"


def test_boundary_defect_partial_xyz():
    """Validate boundary defect with partial/missing XYZ coverage."""
    h, w = 100, 100
    spacing = 0.5
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    x_mm = (grid_x - w / 2.0 + 0.5) * spacing
    y_mm = (grid_y - h / 2.0 + 0.5) * spacing
    z_mm = np.ones((h, w), dtype=np.float32) * 50.0

    defect_mask = (np.abs(x_mm) <= 5.0) & (np.abs(y_mm) <= 5.0)
    z_mm[defect_mask] += 2.0

    xyz = np.stack([x_mm, y_mm, z_mm], axis=-1)
    # 40% missing XYZ along one side
    valid_mask = np.ones((h, w), dtype=bool)
    valid_mask[x_mm > 2.0] = False

    anomaly_map = np.zeros((h, w), dtype=np.float32)
    anomaly_map[defect_mask] = 0.85

    vol = quantify_defect_volume(
        component_mask=defect_mask,
        anomaly_map=anomaly_map,
        scaled_xyz=xyz,
        valid_xyz_mask=valid_mask,
        is_physical_calibrated=True,
        threshold=0.5,
        unit="mm"
    )

    # Partial volume is integrated over remaining valid region without crash
    assert vol.missing_material_volume > 0.0
    assert vol.valid_point_count < 400
    # Measurement confidence reflects the partial coverage
    assert vol.measurement_confidence < 0.90

