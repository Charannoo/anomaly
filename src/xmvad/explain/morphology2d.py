"""2D morphological feature extraction and shape characterization."""

from typing import Dict, Any, Tuple
import math
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull


def compute_2d_morphology(component_mask: np.ndarray) -> Tuple[Dict[str, Any], str]:
    """Compute comprehensive 2D morphological metrics and shape classification for a binary mask.
    
    Args:
        component_mask: (H, W) boolean or binary uint8 mask for a single connected component.
        
    Returns:
        metrics: Dictionary containing area, aspect_ratio, circularity, solidity, extent, etc.
        label: Qualitative morphological shape description (e.g. 'elongated irregular').
    """
    area = int(np.sum(component_mask > 0))
    if area == 0:
        return {
            "area_px": 0, "width_px": 0, "height_px": 0, "perimeter": 0.0,
            "centroid_px": (0, 0), "major_axis": 0.0, "minor_axis": 0.0,
            "aspect_ratio": 1.0, "circularity": 0.0, "solidity": 0.0,
            "extent": 0.0, "eccentricity": 0.0, "compactness": 0.0,
            "orientation_deg": 0.0, "bbox": (0, 0, 0, 0)
        }, "empty"

    # Coordinates
    ys, xs = np.nonzero(component_mask > 0)
    min_y, max_y = int(np.min(ys)), int(np.max(ys))
    min_x, max_x = int(np.min(xs)), int(np.max(xs))
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    bbox_area = width * height
    extent = float(area / max(1, bbox_area))

    # Centroid
    cy = float(np.mean(ys))
    cx = float(np.mean(xs))

    # Perimeter via morphological boundary
    eroded = ndimage.binary_erosion(component_mask > 0)
    boundary = (component_mask > 0) & (~eroded)
    perimeter = float(np.sum(boundary))
    if perimeter == 0.0:
        perimeter = 4.0  # single pixel minimum perimeter

    # Circularity: 4 * pi * Area / Perimeter^2 (1.0 for perfect circle)
    circularity = min(1.0, float((4.0 * math.pi * area) / (perimeter * perimeter)))

    # Convex hull & solidity
    points = np.column_stack((xs, ys))
    if len(points) >= 3:
        try:
            hull = ConvexHull(points)
            hull_area = float(hull.volume)  # For 2D, volume is area
            solidity = min(1.0, float(area / max(area, hull_area)))
        except Exception:
            solidity = extent
    else:
        solidity = 1.0

    # Moments and ellipse fitting for axis lengths and orientation
    # Central moments: mu20, mu02, mu11
    x_centered = xs - cx
    y_centered = ys - cy
    mu20 = np.mean(x_centered ** 2)
    mu02 = np.mean(y_centered ** 2)
    mu11 = np.mean(x_centered * y_centered)

    # Common orientation & inertia axis formula
    delta = math.sqrt(max(0.0, (mu20 - mu02) ** 2 + 4 * (mu11 ** 2)))
    lambda1 = 0.5 * (mu20 + mu02 + delta)
    lambda2 = 0.5 * max(0.0, (mu20 + mu02 - delta))

    # Major and minor axis lengths (4 * sqrt(eigenvalue))
    major_axis = 4.0 * math.sqrt(max(0.0, lambda1))
    minor_axis = 4.0 * math.sqrt(max(0.0, lambda2))
    if minor_axis < 1e-4:
        minor_axis = 1.0
    if major_axis < minor_axis:
        major_axis, minor_axis = minor_axis, major_axis

    aspect_ratio = float(major_axis / max(1.0, minor_axis))
    eccentricity = math.sqrt(max(0.0, 1.0 - (minor_axis / max(1e-4, major_axis)) ** 2))
    compactness = float((perimeter ** 2) / max(1.0, area))

    # Orientation in degrees [-90, 90]
    if abs(mu20 - mu02) > 1e-6 or abs(mu11) > 1e-6:
        theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
        orientation_deg = float(math.degrees(theta))
    else:
        orientation_deg = 0.0

    # Conservative rule-based morphological labeling
    label = _derive_shape_label(circularity, aspect_ratio, solidity, extent, area)

    metrics = {
        "area_px": area,
        "width_px": width,
        "height_px": height,
        "perimeter": round(perimeter, 2),
        "centroid_px": (int(round(cx)), int(round(cy))),
        "centroid_float": (cx, cy),
        "major_axis": round(major_axis, 2),
        "minor_axis": round(minor_axis, 2),
        "aspect_ratio": round(aspect_ratio, 2),
        "circularity": round(circularity, 3),
        "solidity": round(solidity, 3),
        "extent": round(extent, 3),
        "eccentricity": round(eccentricity, 3),
        "compactness": round(compactness, 2),
        "orientation_deg": round(orientation_deg, 1),
        "bbox": (min_x, min_y, max_x, max_y)
    }

    return metrics, label


def _derive_shape_label(
    circularity: float,
    aspect_ratio: float,
    solidity: float,
    extent: float,
    area: int
) -> str:
    """Generate conservative qualitative shape descriptions from morphological metrics."""
    if circularity >= 0.70 and aspect_ratio <= 1.4:
        return "compact circular region"
    elif circularity >= 0.50 and aspect_ratio <= 1.8 and solidity >= 0.85:
        return "compact rounded region"
    elif aspect_ratio >= 3.5:
        if solidity >= 0.75:
            return "thin elongated stripe"
        else:
            return "elongated irregular track"
    elif aspect_ratio >= 2.0:
        if solidity >= 0.80:
            return "elongated oval region"
        else:
            return "elongated irregular region"
    else:
        if solidity >= 0.80 and extent >= 0.55:
            return "compact irregular region"
        else:
            return "diffuse irregular region"
