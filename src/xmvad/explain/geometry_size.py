"""3D physical size, bounding extents, PCA dimensions, and surface area computation."""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from .schema import SizeInfo


def compute_3d_geometry_size(
    component_mask: np.ndarray,
    scaled_xyz: np.ndarray,
    valid_xyz_mask: np.ndarray,
    is_physical_calibrated: bool,
    object_mask: Optional[np.ndarray] = None
) -> SizeInfo:
    """Compute 2D pixel size and calibrated 3D physical dimensions for an anomaly region.
    
    Args:
        component_mask: (H, W) boolean mask for the anomaly region.
        scaled_xyz: (H, W, 3) point cloud map (in mm if calibrated).
        valid_xyz_mask: (H, W) boolean mask of valid geometry points.
        is_physical_calibrated: Boolean indicating if scaled_xyz is in true mm.
        object_mask: Optional (H, W) foreground mask.
        
    Returns:
        SizeInfo instance.
    """
    area_px = int(np.sum(component_mask > 0))
    ys, xs = np.nonzero(component_mask > 0)
    
    if area_px == 0:
        return SizeInfo(
            width_px=0, height_px=0, area_px=0, area_fraction_object=0.0,
            physical_units_available=False
        )

    min_x, max_x = int(np.min(xs)), int(np.max(xs))
    min_y, max_y = int(np.min(ys)), int(np.max(ys))
    width_px = max_x - min_x + 1
    height_px = max_y - min_y + 1

    # Object area fraction
    if object_mask is not None and np.sum(object_mask) > 0:
        obj_area = float(np.sum(object_mask))
        area_fraction = float(area_px / obj_area)
    else:
        h, w = component_mask.shape[:2]
        area_fraction = float(area_px / (h * w))

    # Mask valid 3D points inside defect region
    region_valid = (component_mask > 0) & valid_xyz_mask
    valid_pts = scaled_xyz[region_valid]

    if not is_physical_calibrated or len(valid_pts) < 4:
        return SizeInfo(
            width_px=width_px,
            height_px=height_px,
            area_px=area_px,
            area_fraction_object=area_fraction,
            major_length_mm=None,
            minor_length_mm=None,
            bounding_depth_mm=None,
            projected_area_mm2=None,
            surface_area_3d_mm2=None,
            physical_units_available=False
        )

    # 1. 3D Bounding Extents
    min_xyz = np.min(valid_pts, axis=0)
    max_xyz = np.max(valid_pts, axis=0)
    dx, dy, dz = max_xyz - min_xyz

    # 2. 3D PCA Principal Dimensions
    pts_centered = valid_pts - np.mean(valid_pts, axis=0)
    try:
        # Singular value decomposition
        _, s, vh = np.linalg.svd(pts_centered, full_matrices=False)
        # Project points onto principal component axes
        proj = pts_centered @ vh.T  # (N, 3)
        pca_extents = np.max(proj, axis=0) - np.min(proj, axis=0)
        major_length_mm = float(pca_extents[0])
        minor_length_mm = float(pca_extents[1])
        bounding_depth_mm = float(pca_extents[2]) if len(pca_extents) > 2 else float(dz)
    except Exception:
        major_length_mm = float(max(dx, dy))
        minor_length_mm = float(min(dx, dy))
        bounding_depth_mm = float(dz)

    # 3. 3D Surface Area via Local 2.5D Mesh Triangulation
    surface_area_3d = _estimate_triangulated_surface_area(component_mask, scaled_xyz, valid_xyz_mask)

    # 4. 2D Projected Area in physical space (approximate from major/minor ellipse)
    projected_area_mm2 = float(np.pi * (major_length_mm / 2.0) * (minor_length_mm / 2.0))

    return SizeInfo(
        width_px=width_px,
        height_px=height_px,
        area_px=area_px,
        area_fraction_object=area_fraction,
        major_length_mm=round(major_length_mm, 2),
        minor_length_mm=round(minor_length_mm, 2),
        bounding_depth_mm=round(bounding_depth_mm, 2),
        projected_area_mm2=round(projected_area_mm2, 2),
        surface_area_3d_mm2=round(surface_area_3d, 2) if surface_area_3d is not None else None,
        physical_units_available=True
    )


def _estimate_triangulated_surface_area(
    mask: np.ndarray,
    xyz: np.ndarray,
    valid_mask: np.ndarray
) -> Optional[float]:
    """Estimate 3D surface area by summing areas of valid micro-triangles on the organized grid."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) < 4:
        return None

    min_y, max_y = max(0, int(np.min(ys))), min(h - 2, int(np.max(ys)))
    min_x, max_x = max(0, int(np.min(xs))), min(w - 2, int(np.max(xs)))

    total_area = 0.0
    valid_tri_count = 0

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            # Check 2x2 grid cell: (y,x), (y, x+1), (y+1, x), (y+1, x+1)
            # Cell points
            p00 = (y, x)
            p01 = (y, x + 1)
            p10 = (y + 1, x)
            p11 = (y + 1, x + 1)

            m00, v00 = mask[p00], valid_mask[p00]
            m01, v01 = mask[p01], valid_mask[p01]
            m10, v10 = mask[p10], valid_mask[p10]
            m11, v11 = mask[p11], valid_mask[p11]

            # Triangle 1: (p00, p01, p10)
            if (m00 or m01 or m10) and (v00 and v01 and v10):
                a = xyz[p00]
                b = xyz[p01]
                c = xyz[p10]
                cross = np.cross(b - a, c - a)
                tri_area = 0.5 * np.linalg.norm(cross)
                # Sanity check against extreme jump discontinuities (e.g. edge artifacts > 500mm^2)
                if tri_area < 25.0:
                    total_area += tri_area
                    valid_tri_count += 1

            # Triangle 2: (p01, p11, p10)
            if (m01 or m11 or m10) and (v01 and v11 and v10):
                a = xyz[p01]
                b = xyz[p11]
                c = xyz[p10]
                cross = np.cross(b - a, c - a)
                tri_area = 0.5 * np.linalg.norm(cross)
                if tri_area < 25.0:
                    total_area += tri_area
                    valid_tri_count += 1

    if valid_tri_count == 0:
        return None

    return float(total_area)
