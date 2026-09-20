"""Object-relative spatial location mapping."""

from typing import Tuple, Optional
import numpy as np
from .schema import LocationInfo


def describe_location(
    centroid_px: Tuple[float, float],
    image_shape: Tuple[int, int],
    object_mask: Optional[np.ndarray] = None
) -> LocationInfo:
    """Map pixel centroid into normalized coordinates and a 3x3 semantic grid relative to object mask.
    
    Args:
        centroid_px: (cx, cy) pixel centroid coordinates.
        image_shape: (H, W) image dimensions.
        object_mask: Optional (H, W) boolean mask for foreground object.
        
    Returns:
        LocationInfo containing semantic label and normalized coordinates.
    """
    cx, cy = centroid_px
    h, w = image_shape[:2]

    relative_to_object = False
    if object_mask is not None and np.sum(object_mask) > 10:
        ys, xs = np.nonzero(object_mask)
        obj_min_x, obj_max_x = float(np.min(xs)), float(np.max(xs))
        obj_min_y, obj_max_y = float(np.min(ys)), float(np.max(ys))
        
        obj_w = max(1.0, obj_max_x - obj_min_x)
        obj_h = max(1.0, obj_max_y - obj_min_y)

        x_norm = float(np.clip((cx - obj_min_x) / obj_w, 0.0, 1.0))
        y_norm = float(np.clip((cy - obj_min_y) / obj_h, 0.0, 1.0))
        relative_to_object = True
    else:
        x_norm = float(np.clip(cx / max(1.0, w), 0.0, 1.0))
        y_norm = float(np.clip(cy / max(1.0, h), 0.0, 1.0))

    # 3x3 Grid partition thresholds: [0, 0.33], [0.33, 0.67], [0.67, 1.0]
    if y_norm < 0.35:
        row = "upper"
    elif y_norm > 0.65:
        row = "lower"
    else:
        row = "center"

    if x_norm < 0.35:
        col = "left"
    elif x_norm > 0.65:
        col = "right"
    else:
        col = "center"

    if row == "center" and col == "center":
        label = "central surface"
    elif row == "center":
        label = f"mid-{col} surface"
    elif col == "center":
        label = f"{row}-central surface"
    else:
        label = f"{row}-{col} surface"

    return LocationInfo(
        label=label,
        centroid_px=(int(round(cx)), int(round(cy))),
        centroid_normalized=(round(x_norm, 4), round(y_norm, 4)),
        relative_to_object=relative_to_object
    )
