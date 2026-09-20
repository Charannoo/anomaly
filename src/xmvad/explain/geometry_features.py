"""Geometric feature extraction: curvature, roughness, planarity, and surface variation."""

from typing import Dict, Any, Tuple
import numpy as np


def compute_geometry_descriptors(
    defect_pts: np.ndarray,
    ring_pts: np.ndarray
) -> Dict[str, float]:
    """Calculate geometric descriptors and relative shifts between anomaly region and normal reference ring.
    
    Args:
        defect_pts: (N, 3) XYZ coordinates in anomaly region.
        ring_pts: (M, 3) XYZ coordinates in surrounding normal ring.
        
    Returns:
        Dictionary with planarity, surface_variation, roughness_ratio, etc.
    """
    if len(defect_pts) < 4:
        return {
            "planarity": 1.0,
            "surface_variation": 0.0,
            "roughness_ratio": 1.0,
            "surface_variation_delta": 0.0
        }

    # Defect eigenvalues
    d_centered = defect_pts - np.mean(defect_pts, axis=0)
    d_cov = np.cov(d_centered, rowvar=False)
    try:
        d_eig = np.sort(np.linalg.eigvalsh(d_cov))[::-1]
        l1, l2, l3 = max(1e-6, d_eig[0]), max(1e-6, d_eig[1]), max(1e-6, d_eig[2])
        planarity = float((l2 - l3) / l1)
        surface_var = float(l3 / (l1 + l2 + l3))
    except Exception:
        planarity = 1.0
        surface_var = 0.0

    # Ring eigenvalues if available
    ring_surface_var = 0.0
    if len(ring_pts) >= 4:
        r_centered = ring_pts - np.mean(ring_pts, axis=0)
        r_cov = np.cov(r_centered, rowvar=False)
        try:
            r_eig = np.sort(np.linalg.eigvalsh(r_cov))[::-1]
            rl1, rl2, rl3 = max(1e-6, r_eig[0]), max(1e-6, r_eig[1]), max(1e-6, r_eig[2])
            ring_surface_var = float(rl3 / (rl1 + rl2 + rl3))
        except Exception:
            ring_surface_var = 0.0

    var_delta = float(surface_var - ring_surface_var)

    return {
        "planarity": round(planarity, 3),
        "surface_variation": round(surface_var, 3),
        "surface_variation_delta": round(var_delta, 3)
    }
