"""Anomaly region extraction, connected-component analysis, and component ranking."""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy import ndimage


def extract_anomaly_regions(
    anomaly_map: np.ndarray,
    threshold: Optional[float] = None,
    min_area_px: int = 12,
    max_regions: int = 10
) -> Tuple[List[Dict[str, Any]], float, np.ndarray]:
    """Extract, filter, and rank connected anomaly regions from a continuous anomaly map.
    
    Args:
        anomaly_map: (H, W) continuous anomaly heatmap from PNTC.
        threshold: Optional binary threshold. Defaults to 0.50.
        min_area_px: Minimum pixel count to filter out single-pixel sensor noise.
        max_regions: Maximum number of prominent defect regions to return.
        
    Returns:
        regions: List of dictionaries containing component mask, stats, and importance rank.
        operating_threshold: The threshold applied.
        binary_mask: (H, W) cleaned binary anomaly mask.
    """
    if threshold is None:
        # Default operating threshold calibrated on normal validation quantile
        threshold = 0.50

    # 1. Binarization
    binary_raw = anomaly_map >= threshold

    # 2. Morphological noise cleanup (remove isolated 1-2 pixel spikes)
    struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    binary_cleaned = ndimage.binary_opening(binary_raw, structure=struct)

    # 3. Connected-component labeling
    labeled_array, num_features = ndimage.label(binary_cleaned, structure=struct)

    if num_features == 0:
        return [], float(threshold), binary_cleaned.astype(np.uint8)

    # 4. Extract region statistics and calculate importance
    candidates = []
    for comp_id in range(1, num_features + 1):
        comp_mask = (labeled_array == comp_id)
        area_px = int(np.sum(comp_mask))

        if area_px < min_area_px:
            continue

        scores_in_region = anomaly_map[comp_mask]
        mean_score = float(np.mean(scores_in_region))
        max_score = float(np.max(scores_in_region))
        
        # Importance metric: integrated evidence combining severity and spatial extent
        # Importance = mean_score * sqrt(area_px)
        importance = float(mean_score * np.sqrt(area_px))

        candidates.append({
            "raw_id": comp_id,
            "mask": comp_mask,
            "area_px": area_px,
            "mean_score": mean_score,
            "max_score": max_score,
            "importance": importance
        })

    # 5. Sort components by importance descending
    candidates.sort(key=lambda x: x["importance"], reverse=True)
    ranked_regions = candidates[:max_regions]

    # Assign 1-indexed ranks
    for rank_idx, reg in enumerate(ranked_regions, start=1):
        reg["id"] = rank_idx

    return ranked_regions, float(threshold), binary_cleaned.astype(np.uint8)
