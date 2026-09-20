"""Local reference surface estimation, plane fitting, and signed geometric deviation analysis."""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy import ndimage
from .schema import GeometryInfo


def analyze_local_surface(
    component_mask: np.ndarray,
    anomaly_map: np.ndarray,
    scaled_xyz: np.ndarray,
    valid_xyz_mask: np.ndarray,
    is_physical_calibrated: bool,
    threshold: float = 0.5,
    unit: str = "mm"
) -> GeometryInfo:
    """Estimate local reference plane from surrounding normal ring and compute signed surface deviations.
    
    Args:
        component_mask: (H, W) boolean mask for anomaly region.
        anomaly_map: (H, W) continuous anomaly heatmap.
        scaled_xyz: (H, W, 3) point cloud map.
        valid_xyz_mask: (H, W) boolean mask of valid geometry.
        is_physical_calibrated: Boolean indicating whether scaled_xyz is in true mm.
        threshold: Operating threshold above which pixels are considered anomalous.
        unit: Coordinate unit ('mm' or 'unknown').
        
    Returns:
        GeometryInfo instance.
    """
    region_valid = (component_mask > 0) & valid_xyz_mask
    defect_pts = scaled_xyz[region_valid]

    if len(defect_pts) < 4:
        return GeometryInfo(
            label="insufficient 3D data",
            mean_deviation=0.0,
            max_depression=None,
            max_protrusion=None,
            rms_deviation=0.0,
            percentile_95_abs_deviation=0.0,
            surface_z_score=0.0,
            roughness_ratio=1.0,
            planarity=1.0,
            surface_variation=0.0,
            measurement_confidence=0.0,
            unit=unit
        )

    # 1. Extract Surrounding Normal Ring
    # Outer dilation radius = 10 px, Inner dilation radius = 3 px
    struct_outer = ndimage.generate_binary_structure(2, 2)
    outer_dilated = ndimage.binary_dilation(component_mask > 0, structure=struct_outer, iterations=8)
    inner_dilated = ndimage.binary_dilation(component_mask > 0, structure=struct_outer, iterations=2)
    
    # Ring must be outside anomaly, have valid XYZ, and have low anomaly score (< threshold)
    ring_mask = outer_dilated & (~inner_dilated) & valid_xyz_mask & (anomaly_map <= threshold * 1.2)
    ring_pts = scaled_xyz[ring_mask]

    # Fallback to wider ring if insufficient points
    if len(ring_pts) < 10:
        wider_dilated = ndimage.binary_dilation(component_mask > 0, structure=struct_outer, iterations=16)
        ring_mask = wider_dilated & (~inner_dilated) & valid_xyz_mask
        ring_pts = scaled_xyz[ring_mask]

    if len(ring_pts) < 6:
        # Cannot reliably fit reference plane
        return GeometryInfo(
            label="unsupported local boundary",
            mean_deviation=0.0,
            max_depression=None,
            max_protrusion=None,
            rms_deviation=0.0,
            percentile_95_abs_deviation=0.0,
            surface_z_score=0.0,
            roughness_ratio=1.0,
            planarity=1.0,
            surface_variation=0.0,
            measurement_confidence=0.1,
            unit=unit
        )

    # 2. Robust Plane Fitting (SVD with iterative outlier trimming)
    plane_normal, plane_d, ring_rms = _fit_robust_plane(ring_pts)

    # Enforce outward plane normal pointing towards the camera / viewing origin (0, 0, 0)
    # The centroid of the ring points has viewing vector from origin: centroid
    ring_centroid = np.mean(ring_pts, axis=0)
    # If dot(plane_normal, ring_centroid) > 0, normal points away from camera; flip to point towards camera
    if np.dot(plane_normal, ring_centroid) > 0:
        plane_normal = -plane_normal
        plane_d = -plane_d

    # 3. Compute Signed Point-to-Plane Deviations for Defect Points
    # Distance = dot(normal, p) + d
    # Since normal points towards the camera (outward from surface):
    # - Point closer to camera (raised/protruding) has dot(normal, p) + d > 0 (positive)
    # - Point further from camera (depressed/recessed) has dot(normal, p) + d < 0 (negative)
    signed_devs = np.dot(defect_pts, plane_normal) + plane_d

    mean_dev = float(np.mean(signed_devs))
    rms_dev = float(np.sqrt(np.mean(signed_devs ** 2)))
    p95_abs = float(np.percentile(np.abs(signed_devs), 95))

    # Negative deviations indicate depth recession; positive indicate height protrusion
    min_dev = float(np.min(signed_devs))
    max_dev = float(np.max(signed_devs))

    max_depression = float(abs(min_dev)) if min_dev < -1e-4 else 0.0
    max_protrusion = float(max_dev) if max_dev > 1e-4 else 0.0

    # Surface noise calibration
    sigma_noise = max(ring_rms, 0.05 if is_physical_calibrated else 0.01)
    surface_z = float(abs(mean_dev) / sigma_noise)

    # Roughness estimation (RMS deviation from local plane in defect vs ring)
    defect_rms = float(np.std(signed_devs))
    roughness_ratio = float(defect_rms / max(1e-4, ring_rms))

    # Eigenvalues of defect covariance for planarity and variation
    defect_centered = defect_pts - np.mean(defect_pts, axis=0)
    cov = np.cov(defect_centered, rowvar=False)
    try:
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        l1, l2, l3 = max(1e-6, eigvals[0]), max(1e-6, eigvals[1]), max(1e-6, eigvals[2])
        planarity = float((l2 - l3) / l1)
        surface_variation = float(l3 / (l1 + l2 + l3))
    except Exception:
        planarity = 1.0
        surface_variation = 0.0

    # Confidence score: combination of ring support, residual quality, and defect point density
    ring_support_conf = min(1.0, len(ring_pts) / 100.0)
    fit_conf = max(0.0, min(1.0, 1.0 - (ring_rms / (p95_abs + 1e-4))))
    density_conf = min(1.0, len(defect_pts) / max(1.0, np.sum(component_mask)))
    measurement_confidence = float(np.clip(0.4 * ring_support_conf + 0.4 * fit_conf + 0.2 * density_conf, 0.0, 1.0))

    # 4. Classify Geometric Structure
    label = _classify_geometric_structure(
        mean_dev=mean_dev,
        max_depression=max_depression,
        max_protrusion=max_protrusion,
        rms_dev=rms_dev,
        surface_z=surface_z,
        roughness_ratio=roughness_ratio,
        sigma_noise=sigma_noise
    )

    return GeometryInfo(
        label=label,
        mean_deviation=round(mean_dev, 3),
        max_depression=round(max_depression, 2) if max_depression > 0.05 else None,
        max_protrusion=round(max_protrusion, 2) if max_protrusion > 0.05 else None,
        rms_deviation=round(rms_dev, 3),
        percentile_95_abs_deviation=round(p95_abs, 3),
        surface_z_score=round(surface_z, 2),
        roughness_ratio=round(roughness_ratio, 2),
        planarity=round(planarity, 3),
        surface_variation=round(surface_variation, 3),
        measurement_confidence=round(measurement_confidence, 2),
        unit=unit if is_physical_calibrated else "normalized_units"
    )


def _fit_robust_plane(pts: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Fit a plane ax + by + cz + d = 0 using SVD with iterative outlier rejection."""
    # Iteration 1: standard PCA / SVD plane fit
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2, :]
    normal = normal / np.linalg.norm(normal)
    d = -float(np.dot(normal, centroid))

    # Residuals
    dists = np.abs(np.dot(pts, normal) + d)
    inliers = dists < np.percentile(dists, 85)

    if np.sum(inliers) >= 4:
        centroid = np.mean(pts[inliers], axis=0)
        centered = pts[inliers] - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2, :]
        normal = normal / np.linalg.norm(normal)
        d = -float(np.dot(normal, centroid))
        rms = float(np.sqrt(np.mean((np.dot(pts[inliers], normal) + d) ** 2)))
    else:
        rms = float(np.sqrt(np.mean(dists ** 2)))

    return normal, d, rms


def _classify_geometric_structure(
    mean_dev: float,
    max_depression: float,
    max_protrusion: float,
    rms_dev: float,
    surface_z: float,
    roughness_ratio: float,
    sigma_noise: float
) -> str:
    """Classify 3D structural deviation using train-normal standardized noise floor."""
    if surface_z < 1.5 and roughness_ratio < 1.6:
        return "primarily appearance-based anomaly"

    # Predominantly recessed
    if mean_dev < -1.8 * sigma_noise and max_depression > 1.8 * max_protrusion:
        if roughness_ratio > 2.5:
            return "rough localized depression"
        return "localized depression"

    # Predominantly protruding
    if mean_dev > 1.8 * sigma_noise and max_protrusion > 1.8 * max_depression:
        if roughness_ratio > 2.5:
            return "rough raised protrusion"
        return "raised protrusion"

    # Mixed positive and negative deviations
    if rms_dev > 2.0 * sigma_noise:
        if roughness_ratio > 2.2:
            return "irregular warped surface"
        return "irregular local deformation"

    return "subtle surface deformation"
