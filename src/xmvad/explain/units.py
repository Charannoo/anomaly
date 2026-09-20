"""Unit and scale validation for 3D coordinates."""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class CoordinateUnitManager:
    """Manages coordinate scaling, unit detection, and physical measurement validation.
    
    Supports:
        - "mm": Millimeters (standard industrial metric for MVTec 3D-AD physical coordinates)
        - "m": Meters (converted to mm with scale factor 1000.0)
        - "unknown": Uncalibrated or normalized coordinate frames (suppresses metric claims)
    """

    def __init__(self, declared_unit: str = "unknown"):
        self.declared_unit = declared_unit.lower()

    def validate_and_scale(
        self,
        xyz: np.ndarray,
        valid_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, str, float, bool]:
        """Validate XYZ coordinates and return scaled coordinates in millimeters if calibrated.
        
        Args:
            xyz: (H, W, 3) point cloud map.
            valid_mask: Optional (H, W) boolean mask of valid points.
            
        Returns:
            scaled_xyz: (H, W, 3) coordinates (in mm if physical units available).
            unit_name: 'mm', 'm', or 'unknown'.
            scale_to_mm: Multiplier applied to get millimeters (e.g. 1.0 or 1000.0).
            is_physical_calibrated: True if coordinates represent true physical metric dimensions.
        """
        if valid_mask is None:
            valid_mask = ~np.isnan(xyz).any(axis=-1) & ~np.isinf(xyz).any(axis=-1) & (np.abs(xyz).sum(axis=-1) > 1e-6)

        if not np.any(valid_mask):
            return xyz.copy(), "unknown", 1.0, False

        valid_pts = xyz[valid_mask]
        
        # Extent across valid points
        min_vals = np.min(valid_pts, axis=0)
        max_vals = np.max(valid_pts, axis=0)
        extents = max_vals - min_vals
        max_extent = float(np.max(extents))

        # Explicit declared unit handling
        if self.declared_unit == "mm":
            return xyz.copy(), "mm", 1.0, True
        elif self.declared_unit == "m":
            return xyz * 1000.0, "mm", 1000.0, True
        elif self.declared_unit == "unknown":
            # Auto-heuristic inspection: typical industrial parts in MVTec 3D-AD
            # If coordinates are in [0.01, 1.0] range (e.g., meters)
            if 0.01 <= max_extent <= 1.5 and np.max(np.abs(valid_pts)) <= 2.5:
                # Likely meters -> convert to mm
                return xyz * 1000.0, "mm", 1000.0, True
            # If coordinates are in [10.0, 1500.0] range (e.g., millimeters)
            elif 5.0 <= max_extent <= 2000.0:
                return xyz.copy(), "mm", 1.0, True
            else:
                # Normalized or uncalibrated synthetic coordinates
                return xyz.copy(), "unknown", 1.0, False

        return xyz.copy(), "unknown", 1.0, False
