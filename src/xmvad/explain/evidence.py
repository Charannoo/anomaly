"""Cross-modal detection evidence aggregation and qualitative level assignment."""

from typing import Dict, Any, Optional
import numpy as np
from .schema import EvidenceInfo


def aggregate_detection_evidence(
    component_mask: np.ndarray,
    anomaly_map: np.ndarray,
    rgb_map: Optional[np.ndarray] = None,
    xyz_map: Optional[np.ndarray] = None,
    topology_map: Optional[np.ndarray] = None,
    gate_map: Optional[np.ndarray] = None
) -> EvidenceInfo:
    """Aggregate PNTC cross-modal evidence components within an anomaly region.
    
    Args:
        component_mask: (H, W) boolean mask for anomaly region.
        anomaly_map: (H, W) overall PNTC anomaly score map.
        rgb_map: Optional (H, W) RGB branch anomaly map.
        xyz_map: Optional (H, W) XYZ branch anomaly map.
        topology_map: Optional (H, W) normalized JS divergence topology map T(p).
        gate_map: Optional (H, W) confidence gate map G(p).
        
    Returns:
        EvidenceInfo instance with quantitative scores and qualitative level tiers.
    """
    mask = component_mask > 0
    if not np.any(mask):
        return EvidenceInfo(
            rgb_evidence_score=0.0, xyz_evidence_score=0.0,
            topology_disagreement_score=0.0, gate_strength=0.0,
            rgb_level="Low", xyz_level="Low",
            topology_level="Low", gate_level="Low"
        )

    # Base anomaly score
    mean_overall = float(np.mean(anomaly_map[mask]))

    # Component maps or sensible fallback approximations
    if rgb_map is not None:
        rgb_score = float(np.mean(rgb_map[mask]))
    else:
        rgb_score = mean_overall * 0.9

    if xyz_map is not None:
        xyz_score = float(np.mean(xyz_map[mask]))
    else:
        xyz_score = mean_overall * 0.9

    if topology_map is not None:
        topo_score = float(np.mean(topology_map[mask]))
    else:
        topo_score = min(1.0, mean_overall * 1.1)

    if gate_map is not None:
        gate_score = float(np.mean(gate_map[mask]))
    else:
        gate_score = min(1.0, mean_overall)

    # Qualitative level mapping
    def to_level(val: float) -> str:
        if val >= 0.82:
            return "Very High"
        elif val >= 0.60:
            return "High"
        elif val >= 0.35:
            return "Moderate"
        else:
            return "Low"

    return EvidenceInfo(
        rgb_evidence_score=round(rgb_score, 3),
        xyz_evidence_score=round(xyz_score, 3),
        topology_disagreement_score=round(topo_score, 3),
        gate_strength=round(gate_score, 3),
        rgb_level=to_level(rgb_score),
        xyz_level=to_level(xyz_score),
        topology_level=to_level(topo_score),
        gate_level=to_level(gate_score)
    )
