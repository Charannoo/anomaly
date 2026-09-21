"""Measurement, sensor, and data quality diagnostics for the inspection suite."""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy import ndimage

from .schema import QualityInfo


def evaluate_quality(
    component_mask: np.ndarray,
    observed_rgb: np.ndarray,
    observed_xyz: np.ndarray,
    valid_xyz_mask: np.ndarray,
    surface_fit_confidence: float = 0.85,
    topk_distances: Optional[list] = None
) -> QualityInfo:
    """Evaluate sensor quality, point coverage, boundary proximity, and retrieval stability.
    
    Args:
        component_mask: (H, W) boolean mask for defect region.
        observed_rgb: (H, W, 3) RGB image.
        observed_xyz: (H, W, 3) 3D coordinate map.
        valid_xyz_mask: (H, W) boolean valid geometry mask.
        surface_fit_confidence: Confidence score from local surface fitting.
        topk_distances: Optional list of retrieved prototype distances [d1, ..., dk, d_{k+1}].
        
    Returns:
        QualityInfo dataclass.
    """
    total_region_px = int(np.sum(component_mask > 0))
    if total_region_px == 0:
        return QualityInfo(
            xyz_valid_fraction=1.0,
            missing_xyz_fraction=0.0,
            point_density=1.0,
            rgb_contrast=0.5,
            surface_fit_confidence=surface_fit_confidence,
            region_size_px=0,
            object_boundary_proximity=0.0,
            alignment_confidence=1.0,
            retrieval_stability="high",
            retrieval_margin_gap=0.10
        )

    # 1. 3D Geometric Coverage inside defect region
    valid_in_region = component_mask & valid_xyz_mask
    valid_count = int(np.sum(valid_in_region))
    valid_fraction = float(valid_count / total_region_px)
    missing_fraction = 1.0 - valid_fraction

    # 2. Point Density (valid points per pixel area)
    point_density = float(np.clip(valid_fraction, 0.0, 1.0))

    # 3. RGB Contrast & Quality
    if observed_rgb.ndim == 3 and observed_rgb.shape[-1] >= 3:
        region_rgb = observed_rgb[component_mask]
        rgb_mean = float(np.mean(region_rgb))
        rgb_std = float(np.std(region_rgb))
        rgb_contrast = float(np.clip(rgb_std / max(1.0, rgb_mean), 0.0, 1.0))
    else:
        rgb_contrast = 0.5

    # 4. Object Boundary Proximity
    # Distance of defect centroid to the edge of the valid foreground object mask
    dist_to_boundary = _compute_boundary_proximity(component_mask, valid_xyz_mask)

    # 5. Retrieval Stability (Neighbor Margin Gap analysis: d_{k+1} - d_k)
    margin_gap = 0.08
    if topk_distances and len(topk_distances) >= 2:
        margin_gap = float(topk_distances[-1] - topk_distances[-2])
    
    if margin_gap > 0.04:
        stability = "high"
    elif margin_gap > 0.015:
        stability = "moderate"
    else:
        stability = "unstable"

    # 6. Alignment Confidence
    alignment_conf = float(np.clip(valid_fraction * (1.0 - 0.4 * dist_to_boundary), 0.1, 1.0))

    return QualityInfo(
        xyz_valid_fraction=round(valid_fraction, 4),
        missing_xyz_fraction=round(missing_fraction, 4),
        point_density=round(point_density, 3),
        rgb_contrast=round(rgb_contrast, 3),
        surface_fit_confidence=round(surface_fit_confidence, 2),
        region_size_px=total_region_px,
        object_boundary_proximity=round(dist_to_boundary, 2),
        alignment_confidence=round(alignment_conf, 2),
        retrieval_stability=stability,
        retrieval_margin_gap=round(margin_gap, 4)
    )


def _compute_boundary_proximity(component_mask: np.ndarray, foreground_mask: np.ndarray) -> float:
    """Compute normalized proximity to foreground object boundary [0 = center, 1 = touching boundary]."""
    ys, xs = np.nonzero(component_mask)
    if len(ys) == 0:
        return 0.0

    cy, cx = int(np.mean(ys)), int(np.mean(xs))
    h, w = component_mask.shape

    # Invert foreground mask to compute distance transform to background
    bg_mask = ~foreground_mask
    if not np.any(bg_mask):
        return 0.0

    dist_transform = ndimage.distance_transform_edt(~bg_mask)
    dist_at_centroid = float(dist_transform[min(h - 1, max(0, cy)), min(w - 1, max(0, cx))])
    max_dist = float(np.max(dist_transform))

    if max_dist < 1e-3:
        return 1.0

    # Closer to edge -> lower distance -> higher proximity
    norm_dist = dist_at_centroid / max_dist
    boundary_proximity = float(np.clip(1.0 - norm_dist, 0.0, 1.0))
    return boundary_proximity
