"""Data models and schemas for PNTC Explain defect characterization."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class ExplainInput:
    """Input contract for the PNTC Explain module.
    
    Attributes:
        rgb: Registered RGB image (H x W x 3), uint8 or float32 [0, 1] or [0, 255].
        xyz: Registered XYZ coordinate map (H x W x 3), float32.
        anomaly_map: Continuous 2D anomaly heatmap from PNTC (H x W), float32.
        image_score: Scalar image-level anomaly score from PNTC (e.g., mean_top_0.5%).
        category: Object category name (e.g. 'cookie', 'foam', 'cable_gland').
        sample_id: Identifier of the sample (e.g. '001', 'test_cookie_012').
        rgb_anomaly_map: Optional component RGB anomaly map (H x W).
        xyz_anomaly_map: Optional component XYZ anomaly map (H x W).
        topology_map: Optional normalized JS divergence topology map T(p) (H x W).
        confidence_gate_map: Optional confidence gate map G(p) (H x W).
        valid_xyz_mask: Optional boolean mask (H x W) indicating valid geometric points.
        coordinate_unit: Unit of XYZ coordinates ('mm', 'm', 'unknown').
        threshold: Optional operating threshold for defect extraction.
    """
    rgb: np.ndarray
    xyz: np.ndarray
    anomaly_map: np.ndarray
    image_score: float
    category: str = "generic_object"
    sample_id: str = "unknown_sample"
    rgb_anomaly_map: Optional[np.ndarray] = None
    xyz_anomaly_map: Optional[np.ndarray] = None
    topology_map: Optional[np.ndarray] = None
    confidence_gate_map: Optional[np.ndarray] = None
    valid_xyz_mask: Optional[np.ndarray] = None
    coordinate_unit: str = "unknown"
    threshold: Optional[float] = None


@dataclass
class LocationInfo:
    """Spatial location of the defect region."""
    label: str  # e.g. 'upper-right', 'center', 'bottom-left'
    centroid_px: Tuple[int, int]  # (x, y) pixel coordinates
    centroid_normalized: Tuple[float, float]  # (x_norm, y_norm) in [0, 1]
    relative_to_object: bool = True


@dataclass
class SizeInfo:
    """2D and 3D geometric dimensions."""
    width_px: int
    height_px: int
    area_px: int
    area_fraction_object: float  # Percentage of foreground object area
    major_length_mm: Optional[float] = None
    minor_length_mm: Optional[float] = None
    bounding_depth_mm: Optional[float] = None
    projected_area_mm2: Optional[float] = None
    surface_area_3d_mm2: Optional[float] = None
    physical_units_available: bool = False


@dataclass
class ShapeInfo:
    """2D morphological shape characteristics."""
    label: str  # e.g. 'elongated irregular', 'compact circular', 'thin elongated'
    aspect_ratio: float
    circularity: float
    solidity: float
    extent: float
    eccentricity: float
    compactness: float
    orientation_deg: float


@dataclass
class GeometryInfo:
    """3D local surface structure and deviation analysis."""
    label: str  # 'localized depression', 'raised protrusion', 'irregular deformation', 'appearance-only'
    mean_deviation: float  # Signed point-to-plane deviation
    max_depression: Optional[float] = None  # Positive magnitude for depression
    max_protrusion: Optional[float] = None  # Positive magnitude for protrusion
    rms_deviation: float = 0.0
    percentile_95_abs_deviation: float = 0.0
    surface_z_score: float = 0.0  # Normalized deviation relative to normal surface variance
    roughness_ratio: float = 1.0  # Anomaly roughness / Reference ring roughness
    planarity: float = 1.0
    surface_variation: float = 0.0
    measurement_confidence: float = 0.0
    unit: str = "unknown"


@dataclass
class AppearanceInfo:
    """RGB visual appearance deviations."""
    deviation_label: str  # 'minimal', 'moderate', 'strong'
    appearance_deviation_score: float  # [0, 1] normalized deviation score
    mean_rgb_diff: float
    delta_e_lab: float
    contrast_ratio: float
    texture_variance_diff: float


@dataclass
class EvidenceInfo:
    """PNTC cross-modal detection evidence breakdown."""
    rgb_evidence_score: float
    xyz_evidence_score: float
    topology_disagreement_score: float
    gate_strength: float
    rgb_level: str  # 'Low', 'Moderate', 'High', 'Very High'
    xyz_level: str  # 'Low', 'Moderate', 'High', 'Very High'
    topology_level: str  # 'Low', 'Moderate', 'High', 'Very High'
    gate_level: str  # 'Low', 'Moderate', 'High', 'Very High'


@dataclass
class DefectRegion:
    """Comprehensive characterization of a single connected anomalous region."""
    id: int
    location: LocationInfo
    size: SizeInfo
    shape: ShapeInfo
    geometry: GeometryInfo
    appearance: AppearanceInfo
    evidence: EvidenceInfo
    mean_anomaly_score: float
    max_anomaly_score: float
    integrated_importance: float
    explanation: str
    bounding_box_xyxy: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)
    mask: np.ndarray = field(repr=False, default=None)  # Binary component mask (H x W)


@dataclass
class ExplanationReport:
    """Complete structured output report for an inspected sample."""
    sample_id: str
    category: str
    decision: str  # 'normal' or 'anomalous'
    image_anomaly_score: float
    detection_confidence: float
    operating_threshold: float
    num_defects: int
    defects: List[DefectRegion]
    summary_explanation: str
    technical_report: str
    coordinate_unit: str
    physical_measurements_available: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation report to a JSON-serializable dictionary."""
        defects_list = []
        for d in self.defects:
            defect_dict = {
                "id": d.id,
                "importance_rank": d.id,
                "location": {
                    "label": d.location.label,
                    "centroid_px": list(d.location.centroid_px),
                    "centroid_normalized": [round(x, 4) for x in d.location.centroid_normalized],
                    "relative_to_object": d.location.relative_to_object
                },
                "size": {
                    "width_px": d.size.width_px,
                    "height_px": d.size.height_px,
                    "area_px": d.size.area_px,
                    "area_fraction_object_percent": round(d.size.area_fraction_object * 100, 2),
                    "major_length_mm": round(d.size.major_length_mm, 2) if d.size.major_length_mm is not None else None,
                    "minor_length_mm": round(d.size.minor_length_mm, 2) if d.size.minor_length_mm is not None else None,
                    "bounding_depth_mm": round(d.size.bounding_depth_mm, 2) if d.size.bounding_depth_mm is not None else None,
                    "projected_area_mm2": round(d.size.projected_area_mm2, 2) if d.size.projected_area_mm2 is not None else None,
                    "surface_area_3d_mm2": round(d.size.surface_area_3d_mm2, 2) if d.size.surface_area_3d_mm2 is not None else None,
                    "physical_units_available": d.size.physical_units_available
                },
                "shape": {
                    "label": d.shape.label,
                    "aspect_ratio": round(d.shape.aspect_ratio, 2),
                    "circularity": round(d.shape.circularity, 2),
                    "solidity": round(d.shape.solidity, 2),
                    "extent": round(d.shape.extent, 2),
                    "eccentricity": round(d.shape.eccentricity, 2),
                    "orientation_deg": round(d.shape.orientation_deg, 1)
                },
                "geometry": {
                    "label": d.geometry.label,
                    "mean_deviation": round(d.geometry.mean_deviation, 3),
                    "max_depression_mm": round(d.geometry.max_depression, 2) if d.geometry.max_depression is not None else None,
                    "max_protrusion_mm": round(d.geometry.max_protrusion, 2) if d.geometry.max_protrusion is not None else None,
                    "rms_deviation": round(d.geometry.rms_deviation, 3),
                    "surface_z_score": round(d.geometry.surface_z_score, 2),
                    "roughness_ratio": round(d.geometry.roughness_ratio, 2),
                    "planarity": round(d.geometry.planarity, 3),
                    "surface_variation": round(d.geometry.surface_variation, 3),
                    "measurement_confidence": round(d.geometry.measurement_confidence, 2),
                    "unit": d.geometry.unit
                },
                "appearance": {
                    "deviation_label": d.appearance.deviation_label,
                    "score": round(d.appearance.appearance_deviation_score, 2),
                    "delta_e_lab": round(d.appearance.delta_e_lab, 2),
                    "contrast_ratio": round(d.appearance.contrast_ratio, 2)
                },
                "evidence": {
                    "rgb_evidence_score": round(d.evidence.rgb_evidence_score, 3),
                    "rgb_level": d.evidence.rgb_level,
                    "xyz_evidence_score": round(d.evidence.xyz_evidence_score, 3),
                    "xyz_level": d.evidence.xyz_level,
                    "topology_disagreement_score": round(d.evidence.topology_disagreement_score, 3),
                    "topology_level": d.evidence.topology_level,
                    "gate_strength": round(d.evidence.gate_strength, 3),
                    "gate_level": d.evidence.gate_level
                },
                "scores": {
                    "mean_anomaly_score": round(d.mean_anomaly_score, 4),
                    "max_anomaly_score": round(d.max_anomaly_score, 4),
                    "integrated_importance": round(d.integrated_importance, 4)
                },
                "explanation": d.explanation
            }
            defects_list.append(defect_dict)

        return {
            "sample_id": self.sample_id,
            "category": self.category,
            "decision": self.decision,
            "image_anomaly_score": round(self.image_anomaly_score, 5),
            "detection_confidence_percent": round(self.detection_confidence * 100, 1),
            "operating_threshold": round(self.operating_threshold, 5),
            "coordinate_unit": self.coordinate_unit,
            "physical_measurements_available": self.physical_measurements_available,
            "num_defects": self.num_defects,
            "defects": defects_list,
            "summary_explanation": self.summary_explanation,
            "technical_report": self.technical_report
        }
