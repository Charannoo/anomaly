"""Data models and schemas for the PNTC Industrial Inspection Suite."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class InspectionInput:
    """Input contract for the PNTC Industrial Inspection Suite.
    
    Attributes:
        rgb: Registered RGB image (H x W x 3), uint8 or float32.
        xyz: Registered XYZ coordinate map (H x W x 3), float32.
        anomaly_map: Continuous 2D anomaly heatmap from PNTC (H x W), float32.
        image_score: Scalar image-level anomaly score from PNTC.
        category: Object category name (e.g. 'cookie', 'cable_gland').
        sample_id: Identifier of the sample.
        rgb_anomaly_map: Optional individual modality RGB anomaly map (H x W).
        xyz_anomaly_map: Optional individual modality XYZ anomaly map (H x W).
        topology_map: Optional normalized JS divergence topology map T(p) (H x W).
        confidence_gate_map: Optional confidence gate map G(p) (H x W).
        valid_xyz_mask: Optional boolean mask (H x W) indicating valid geometric points.
        coordinate_unit: Unit of XYZ coordinates ('mm', 'm', 'unknown').
        threshold: Operating threshold for anomaly segmentation.
        operating_threshold: Operating threshold for image-level decision.
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
    operating_threshold: Optional[float] = None


@dataclass
class NormalTwinMatch:
    """A single retrieved paired prototype match from normal memory."""
    rank: int
    prototype_id: int
    category: str
    training_sample_id: str
    spatial_coord: Tuple[int, int]
    rgb_distance: float
    xyz_distance: float
    joint_score: float
    modality: str  # 'rgb', 'xyz', 'joint'


@dataclass
class NormalTwinInfo:
    """Normal Twin: nearest paired normal reference retrieved from frozen training memory."""
    prototype_id: int
    training_sample_id: str
    spatial_coord: Tuple[int, int]
    rgb_distance: float
    xyz_distance: float
    joint_score: float
    rgb_similarity: str  # 'Low', 'Moderate', 'High'
    xyz_similarity: str  # 'Low', 'Moderate', 'High'
    rgb_topk_matches: List[NormalTwinMatch]
    xyz_topk_matches: List[NormalTwinMatch]
    alignment_confidence: float
    rgb_difference_metric: float
    xyz_signed_surface_diff: float
    interpretation: str
    source_location_str: str


@dataclass
class VolumeInfo:
    """Geometry Quantification: Material loss, excess material, and net signed volume."""
    missing_material_volume: float
    excess_material_volume: float
    net_signed_volume: float
    volume_unit: str  # 'mm3' or 'native_units3'
    physical_volume_available: bool
    reference_model_type: str  # 'robust_plane' or 'robust_quadratic'
    fit_residual: float
    valid_point_count: int
    fit_confidence: float
    measurement_confidence: float
    explanation: str


@dataclass
class PrototypeRankEntry:
    """Single ranked entry in prototype evidence trace."""
    rank: int
    prototype_id: int
    distance: float
    category: str
    training_sample_id: str
    spatial_coord: Tuple[int, int]


@dataclass
class PrototypeTraceInfo:
    """Detailed PNTC prototype evidence trace and score decomposition."""
    patch_coord: Tuple[int, int]
    rgb_topk: List[PrototypeRankEntry]
    xyz_topk: List[PrototypeRankEntry]
    shared_ids: int
    union_ids: int
    jaccard_overlap: float
    rank_overlap: float
    js_divergence: float
    gate_value: float
    topology_contribution: float
    base_anomaly_evidence: float
    final_pntc_evidence: float
    lambda_param: float
    why_flagged: str
    is_normal_example: bool = False


@dataclass
class QualityInfo:
    """Measurement and sensor quality diagnostics."""
    xyz_valid_fraction: float
    missing_xyz_fraction: float
    point_density: float
    rgb_contrast: float
    surface_fit_confidence: float
    region_size_px: int
    object_boundary_proximity: float
    alignment_confidence: float
    retrieval_stability: str  # 'high', 'moderate', 'unstable'
    retrieval_margin_gap: float


@dataclass
class ReviewGuardInfo:
    """Uncertainty analysis and deployment manual-review guard."""
    pntc_decision: str  # 'normal' or 'anomalous'
    inspection_status: str  # 'ACCEPT_NORMAL', 'DEFECT_DETECTED', 'MANUAL_REVIEW_RECOMMENDED'
    decision_certainty: str  # 'High', 'Moderate', 'Borderline'
    decision_margin: float
    measurement_reliability: str  # 'High', 'Moderate', 'Low'
    reasons: List[str]
    recommendation_text: str


@dataclass
class DefectInspection:
    """Complete inspection outcome for an individual connected defect region."""
    id: int
    representative_patch_coord: Tuple[int, int]
    location: Dict[str, Any]
    morphology: Dict[str, Any]
    geometry: Dict[str, Any]
    volume: VolumeInfo
    normal_twin: NormalTwinInfo
    prototype_trace: PrototypeTraceInfo
    quality: QualityInfo
    review_guard: ReviewGuardInfo
    explanation: str
    bounding_box_xyxy: Tuple[int, int, int, int]
    mask: np.ndarray = field(repr=False, default=None)


@dataclass
class InspectionReport:
    """Unified inspection result and report."""
    sample_id: str
    category: str
    pntc: Dict[str, Any]
    inspection_status: str
    decision_certainty: str
    coordinate_unit: str
    physical_measurements_available: bool
    num_defects: int
    defects: List[DefectInspection]
    overall_quality: QualityInfo
    overall_review_guard: ReviewGuardInfo
    technical_report: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to structured JSON format."""
        defects_data = []
        for d in self.defects:
            defects_data.append({
                "id": d.id,
                "representative_patch": list(d.representative_patch_coord),
                "location": d.location,
                "morphology": d.morphology,
                "geometry": d.geometry,
                "volume": {
                    "missing_material": round(d.volume.missing_material_volume, 3),
                    "excess_material": round(d.volume.excess_material_volume, 3),
                    "net_signed_volume": round(d.volume.net_signed_volume, 3),
                    "volume_unit": d.volume.volume_unit,
                    "physical_volume_available": d.volume.physical_volume_available,
                    "reference_model_type": d.volume.reference_model_type,
                    "fit_residual": round(d.volume.fit_residual, 4),
                    "fit_confidence": round(d.volume.fit_confidence, 2),
                    "measurement_confidence": round(d.volume.measurement_confidence, 2),
                    "explanation": d.volume.explanation
                },
                "normal_twin": {
                    "prototype_id": d.normal_twin.prototype_id,
                    "training_sample_id": d.normal_twin.training_sample_id,
                    "spatial_coord": list(d.normal_twin.spatial_coord),
                    "source": d.normal_twin.source_location_str,
                    "rgb_distance": round(d.normal_twin.rgb_distance, 4),
                    "xyz_distance": round(d.normal_twin.xyz_distance, 4),
                    "joint_score": round(d.normal_twin.joint_score, 4),
                    "rgb_similarity": d.normal_twin.rgb_similarity,
                    "xyz_similarity": d.normal_twin.xyz_similarity,
                    "alignment_confidence": round(d.normal_twin.alignment_confidence, 2),
                    "interpretation": d.normal_twin.interpretation
                },
                "prototype_trace": {
                    "patch_coord": list(d.prototype_trace.patch_coord),
                    "rgb_topk": [
                        {
                            "rank": r.rank,
                            "prototype_id": r.prototype_id,
                            "distance": round(r.distance, 4),
                            "category": r.category,
                            "source_sample": r.training_sample_id,
                            "source_patch": list(r.spatial_coord)
                        } for r in d.prototype_trace.rgb_topk
                    ],
                    "xyz_topk": [
                        {
                            "rank": r.rank,
                            "prototype_id": r.prototype_id,
                            "distance": round(r.distance, 4),
                            "category": r.category,
                            "source_sample": r.training_sample_id,
                            "source_patch": list(r.spatial_coord)
                        } for r in d.prototype_trace.xyz_topk
                    ],
                    "shared_ids": d.prototype_trace.shared_ids,
                    "union_ids": d.prototype_trace.union_ids,
                    "jaccard": round(d.prototype_trace.jaccard_overlap, 4),
                    "rank_overlap": round(d.prototype_trace.rank_overlap, 4),
                    "js_divergence": round(d.prototype_trace.js_divergence, 4),
                    "gate": round(d.prototype_trace.gate_value, 4),
                    "lambda": d.prototype_trace.lambda_param,
                    "base_evidence": round(d.prototype_trace.base_anomaly_evidence, 4),
                    "topology_contribution": round(d.prototype_trace.topology_contribution, 4),
                    "final_pntc_evidence": round(d.prototype_trace.final_pntc_evidence, 4),
                    "why_flagged": d.prototype_trace.why_flagged
                },
                "quality": {
                    "xyz_valid_fraction": round(d.quality.xyz_valid_fraction, 4),
                    "missing_xyz_fraction": round(d.quality.missing_xyz_fraction, 4),
                    "surface_fit_confidence": round(d.quality.surface_fit_confidence, 2),
                    "boundary_proximity": round(d.quality.object_boundary_proximity, 2),
                    "retrieval_stability": d.quality.retrieval_stability,
                    "retrieval_margin_gap": round(d.quality.retrieval_margin_gap, 4)
                },
                "review_guard": {
                    "status": d.review_guard.inspection_status,
                    "decision_certainty": d.review_guard.decision_certainty,
                    "measurement_reliability": d.review_guard.measurement_reliability,
                    "reasons": d.review_guard.reasons,
                    "recommendation": d.review_guard.recommendation_text
                },
                "explanation": d.explanation
            })

        return {
            "sample_id": self.sample_id,
            "category": self.category,
            "pntc": self.pntc,
            "inspection_status": self.inspection_status,
            "decision_certainty": self.decision_certainty,
            "coordinate_unit": self.coordinate_unit,
            "physical_measurements_available": self.physical_measurements_available,
            "num_defects": self.num_defects,
            "defects": defects_data,
            "overall_quality": {
                "xyz_valid_fraction": round(self.overall_quality.xyz_valid_fraction, 4),
                "surface_fit_confidence": round(self.overall_quality.surface_fit_confidence, 2),
                "retrieval_stability": self.overall_quality.retrieval_stability
            },
            "overall_review_guard": {
                "pntc_decision": self.overall_review_guard.pntc_decision,
                "inspection_status": self.overall_review_guard.inspection_status,
                "decision_certainty": self.overall_review_guard.decision_certainty,
                "measurement_reliability": self.overall_review_guard.measurement_reliability,
                "reasons": self.overall_review_guard.reasons,
                "recommendation": self.overall_review_guard.recommendation_text
            },
            "technical_report": self.technical_report
        }
