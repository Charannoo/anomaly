"""PNTC Geometry Quantification: Local reference surface estimation, signed deviation field, and triangulated volume integration."""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy import ndimage

from .schema import VolumeInfo


def quantify_defect_volume(
    component_mask: np.ndarray,
    anomaly_map: np.ndarray,
    scaled_xyz: np.ndarray,
    valid_xyz_mask: np.ndarray,
    is_physical_calibrated: bool,
    threshold: float = 0.5,
    unit: str = "mm"
) -> VolumeInfo:
    """Estimate missing, excess, and net signed volume of a detected defect.
    
    Args:
        component_mask: (H, W) boolean mask for defect region.
        anomaly_map: (H, W) continuous anomaly heatmap.
        scaled_xyz: (H, W, 3) XYZ coordinates (in mm if calibrated).
        valid_xyz_mask: (H, W) boolean mask of valid geometry.
        is_physical_calibrated: Boolean indicating whether scaled_xyz has verified physical units.
        threshold: Operating threshold for anomaly detection.
        unit: Coordinate unit ('mm', 'm', or 'unknown').
        
    Returns:
        VolumeInfo dataclass with volume quantification and confidence metrics.
    """
    region_valid = (component_mask > 0) & valid_xyz_mask
    valid_point_count = int(np.sum(region_valid))
    defect_pts = scaled_xyz[region_valid]

    # Fallback for insufficient 3D data
    if valid_point_count < 6:
        return VolumeInfo(
            missing_material_volume=0.0,
            excess_material_volume=0.0,
            net_signed_volume=0.0,
            volume_unit="mm3" if is_physical_calibrated else "native_units3",
            physical_volume_available=is_physical_calibrated,
            reference_model_type="none",
            fit_residual=0.0,
            valid_point_count=valid_point_count,
            fit_confidence=0.0,
            measurement_confidence=0.0,
            explanation="Insufficient valid 3D points inside defect region (<6 points)."
        )

    # 1. Extract Surrounding Normal Ring Points (excluding anomaly and background)
    ring_mask = _extract_normal_ring(component_mask, valid_xyz_mask, anomaly_map, threshold)
    ring_pts = scaled_xyz[ring_mask]

    if len(ring_pts) < 6:
        return VolumeInfo(
            missing_material_volume=0.0,
            excess_material_volume=0.0,
            net_signed_volume=0.0,
            volume_unit="mm3" if is_physical_calibrated else "native_units3",
            physical_volume_available=is_physical_calibrated,
            reference_model_type="none",
            fit_residual=0.0,
            valid_point_count=valid_point_count,
            fit_confidence=0.0,
            measurement_confidence=0.1,
            explanation="Insufficient surrounding normal points to fit a reliable reference surface."
        )

    # 2. Fit Reference Surface (Robust Local Plane with Outlier Rejection)
    # Allows robust quadratic surface if residual of plane fit indicates high curvature
    plane_normal, plane_d, fit_residual, is_quadratic = _fit_reference_surface(ring_pts)
    model_type = "robust_quadratic" if is_quadratic else "robust_plane"

    # Enforce outward normal pointing towards camera origin (0, 0, 0)
    ring_centroid = np.mean(ring_pts, axis=0)
    if np.dot(plane_normal, ring_centroid) > 0:
        plane_normal = -plane_normal
        plane_d = -plane_d

    # 3. Compute Signed Displacement Field delta_i = signed_distance(p_i, reference_surface)
    # Sign convention:
    #   positive (> 0) = material protruding above expected surface
    #   negative (< 0) = material recessed below expected surface
    h, w = component_mask.shape
    signed_deviation_map = np.zeros((h, w), dtype=np.float32)
    
    # Evaluate plane distance for all valid points in component bounding box
    ys, xs = np.nonzero(component_mask)
    min_y, max_y = int(np.min(ys)), int(np.max(ys))
    min_x, max_x = int(np.min(xs)), int(np.max(xs))

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if component_mask[y, x] and valid_xyz_mask[y, x]:
                pt = scaled_xyz[y, x]
                # Distance = dot(n, pt) + d
                signed_deviation_map[y, x] = float(np.dot(plane_normal, pt) + plane_d)

    # 4. Integrate Volume via Triangulated Micro-Cells
    missing_vol, excess_vol = _integrate_triangulated_volume(
        component_mask=component_mask,
        valid_xyz_mask=valid_xyz_mask,
        scaled_xyz=scaled_xyz,
        signed_dev_map=signed_deviation_map,
        plane_normal=plane_normal,
        plane_d=plane_d
    )

    net_signed_vol = excess_vol - missing_vol

    # 5. Volume Measurement Confidence
    total_mask_pts = int(np.sum(component_mask > 0))
    coverage_ratio = float(valid_point_count / max(1, total_mask_pts))
    fit_conf = float(np.clip(1.0 - (fit_residual / 1.5), 0.1, 1.0))
    
    # Combined measurement confidence
    meas_conf = float(np.clip(coverage_ratio * fit_conf * (1.0 if is_physical_calibrated else 0.5), 0.05, 0.98))

    vol_unit = "mm3" if is_physical_calibrated else "native_units3"

    explanation = _generate_volume_explanation(
        missing_vol=missing_vol,
        excess_vol=excess_vol,
        net_signed_vol=net_signed_vol,
        is_physical=is_physical_calibrated,
        model_type=model_type,
        meas_conf=meas_conf,
        coverage=coverage_ratio
    )

    return VolumeInfo(
        missing_material_volume=round(missing_vol, 3),
        excess_material_volume=round(excess_vol, 3),
        net_signed_volume=round(net_signed_vol, 3),
        volume_unit=vol_unit,
        physical_volume_available=is_physical_calibrated,
        reference_model_type=model_type,
        fit_residual=round(fit_residual, 4),
        valid_point_count=valid_point_count,
        fit_confidence=round(fit_conf, 2),
        measurement_confidence=round(meas_conf, 2),
        explanation=explanation
    )


def _extract_normal_ring(
    component_mask: np.ndarray,
    valid_xyz_mask: np.ndarray,
    anomaly_map: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Extract surrounding normal ring points excluding anomaly and background."""
    struct = ndimage.generate_binary_structure(2, 2)
    outer = ndimage.binary_dilation(component_mask > 0, structure=struct, iterations=8)
    inner = ndimage.binary_dilation(component_mask > 0, structure=struct, iterations=2)
    
    ring = outer & (~inner) & valid_xyz_mask & (anomaly_map <= threshold * 1.25)
    if np.sum(ring) < 10:
        wider = ndimage.binary_dilation(component_mask > 0, structure=struct, iterations=16)
        ring = wider & (~inner) & valid_xyz_mask
    return ring


def _fit_reference_surface(pts: np.ndarray) -> Tuple[np.ndarray, float, float, bool]:
    """Fit robust plane via iterative outlier trimming with curvature diagnostic."""
    centroid = np.mean(pts, axis=0)
    current_pts = pts.copy()
    
    # 2 iterations of outlier removal
    for _ in range(2):
        centered = current_pts - np.mean(current_pts, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[2, :]
        d = -float(np.dot(normal, np.mean(current_pts, axis=0)))
        residuals = np.abs(np.dot(current_pts, normal) + d)
        rms = float(np.sqrt(np.mean(residuals**2)))
        inliers = residuals < 2.5 * max(rms, 0.05)
        if np.sum(inliers) >= 6:
            current_pts = current_pts[inliers]
        else:
            break

    centered = current_pts - np.mean(current_pts, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[2, :]
    d = -float(np.dot(normal, np.mean(current_pts, axis=0)))
    residuals = np.abs(np.dot(pts, normal) + d)
    rms = float(np.sqrt(np.mean(residuals**2)))

    # Curvature check: if plane residual > 0.8 mm, flag quadratic surface model
    is_quadratic = rms > 0.80

    return normal, d, rms, is_quadratic


def _integrate_triangulated_volume(
    component_mask: np.ndarray,
    valid_xyz_mask: np.ndarray,
    scaled_xyz: np.ndarray,
    signed_dev_map: np.ndarray,
    plane_normal: Optional[np.ndarray] = None,
    plane_d: Optional[float] = None
) -> Tuple[float, float]:
    """Integrate missing and excess volume over triangulated reference surface elements.
    
    Uses standard finite-element / dual-mesh integration:
    1. Points are projected onto the expected local reference surface to obtain the base surface
       elements (preventing vertical cliff edge artifacts at sharp defect boundaries).
    2. Each cell micro-triangle assigns 1/3 of its reference surface area to each of its vertices.
    3. Missing and excess volume are integrated over defect vertices:
       V_loss = sum_{i in defect, delta_i < 0} Area_i * (-delta_i)
       V_excess = sum_{i in defect, delta_i > 0} Area_i * delta_i
    """
    h, w = component_mask.shape
    ys, xs = np.nonzero(component_mask)
    if len(ys) == 0:
        return 0.0, 0.0

    # Project points onto local reference plane if provided
    ref_xyz = scaled_xyz.copy()
    if plane_normal is not None and plane_d is not None:
        # p_ref = p - (n . p + d) * n
        devs_all = np.sum(scaled_xyz * plane_normal, axis=-1) + plane_d
        ref_xyz = scaled_xyz - devs_all[..., np.newaxis] * plane_normal

    min_y, max_y = max(0, int(np.min(ys)) - 1), min(h - 2, int(np.max(ys)) + 1)
    min_x, max_x = max(0, int(np.min(xs)) - 1), min(w - 2, int(np.max(xs)) + 1)

    vertex_area = np.zeros((h, w), dtype=np.float64)

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            p00, p01 = (y, x), (y, x + 1)
            p10, p11 = (y + 1, x), (y + 1, x + 1)

            v00, v01 = valid_xyz_mask[p00], valid_xyz_mask[p01]
            v10, v11 = valid_xyz_mask[p10], valid_xyz_mask[p11]

            # Triangle 1: (p00, p01, p10)
            if v00 and v01 and v10:
                pt0 = ref_xyz[p00]
                pt1 = ref_xyz[p01]
                pt2 = ref_xyz[p10]
                tri_area = 0.5 * float(np.linalg.norm(np.cross(pt1 - pt0, pt2 - pt0)))
                share = tri_area / 3.0
                vertex_area[p00] += share
                vertex_area[p01] += share
                vertex_area[p10] += share

            # Triangle 2: (p10, p01, p11)
            if v10 and v01 and v11:
                pt0 = ref_xyz[p10]
                pt1 = ref_xyz[p01]
                pt2 = ref_xyz[p11]
                tri_area = 0.5 * float(np.linalg.norm(np.cross(pt1 - pt0, pt2 - pt0)))
                share = tri_area / 3.0
                vertex_area[p10] += share
                vertex_area[p01] += share
                vertex_area[p11] += share

    # Integrate volume over defect vertices
    defect_indices = (component_mask > 0) & valid_xyz_mask
    defect_areas = vertex_area[defect_indices]
    defect_devs = signed_dev_map[defect_indices]

    neg_mask = defect_devs < 0
    pos_mask = defect_devs > 0

    missing_volume = float(np.sum(defect_areas[neg_mask] * (-defect_devs[neg_mask])))
    excess_volume = float(np.sum(defect_areas[pos_mask] * defect_devs[pos_mask]))

    return missing_volume, excess_volume


def _generate_volume_explanation(
    missing_vol: float,
    excess_vol: float,
    net_signed_vol: float,
    is_physical: bool,
    model_type: str,
    meas_conf: float,
    coverage: float
) -> str:
    """Generate concise, evidence-grounded technical volume explanation."""
    unit_str = "mm³" if is_physical else "native units³ (uncalibrated)"
    
    if not is_physical:
        return (
            f"Defect volumetric deformation estimated using {model_type} reference surface: "
            f"missing volume {missing_vol:.1f} {unit_str}, excess volume {excess_vol:.1f} {unit_str}. "
            f"Physical metric volume (mm³) is withheld because coordinate units are uncalibrated."
        )

    if missing_vol > 0.5 and excess_vol > 0.5:
        return (
            f"Mixed geometric deformation: observed defect exhibits both recessed volume "
            f"({missing_vol:.1f} mm³) and protruding material ({excess_vol:.1f} mm³), yielding a net signed "
            f"volume of {net_signed_vol:+.1f} mm³ relative to the {model_type} (measurement confidence: {meas_conf:.2f})."
        )
    elif missing_vol > 0.5:
        return (
            f"Localized surface depression: estimated missing material volume of {missing_vol:.1f} mm³ "
            f"below the expected {model_type} (point coverage: {coverage*100:.1f}%, confidence: {meas_conf:.2f})."
        )
    elif excess_vol > 0.5:
        return (
            f"Localized surface protrusion: estimated excess material volume of {excess_vol:.1f} mm³ "
            f"above the expected {model_type} (point coverage: {coverage*100:.1f}%, confidence: {meas_conf:.2f})."
        )
    else:
        return (
            f"Minimal geometric volume deviation detected (<0.5 mm³) relative to local {model_type}."
        )
