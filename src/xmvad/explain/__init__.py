"""PNTC Explain: Morphology-Aware Defect Characterization & Explanation.

This package provides an evidence-grounded characterization layer built on top of the frozen
PNTC (Paired Neighborhood Topology Consistency) anomaly detection framework.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np

from .schema import (
    ExplainInput,
    ExplanationReport,
    DefectRegion,
    LocationInfo,
    SizeInfo,
    ShapeInfo,
    GeometryInfo,
    AppearanceInfo,
    EvidenceInfo
)
from .units import CoordinateUnitManager
from .regions import extract_anomaly_regions
from .morphology2d import compute_2d_morphology
from .location import describe_location
from .geometry_size import compute_3d_geometry_size
from .surface import analyze_local_surface
from .appearance import analyze_appearance
from .evidence import aggregate_detection_evidence
from .reasoning import generate_region_explanation
from .report import build_explanation_report
from .visualize import save_annotated_overlay, save_interactive_3d_html

__all__ = [
    "ExplainInput",
    "ExplanationReport",
    "DefectRegion",
    "LocationInfo",
    "SizeInfo",
    "ShapeInfo",
    "GeometryInfo",
    "AppearanceInfo",
    "EvidenceInfo",
    "explain_anomaly",
    "save_annotated_overlay",
    "save_interactive_3d_html"
]


def explain_anomaly(
    inp_or_rgb: Union[ExplainInput, np.ndarray],
    xyz: Optional[np.ndarray] = None,
    anomaly_map: Optional[np.ndarray] = None,
    image_score: Optional[float] = None,
    category: str = "generic_object",
    sample_id: str = "sample_001",
    rgb_anomaly_map: Optional[np.ndarray] = None,
    xyz_anomaly_map: Optional[np.ndarray] = None,
    topology_map: Optional[np.ndarray] = None,
    confidence_gate_map: Optional[np.ndarray] = None,
    valid_xyz_mask: Optional[np.ndarray] = None,
    coordinate_unit: str = "unknown",
    threshold: Optional[float] = None,
    min_area_px: int = 12,
    max_regions: int = 10
) -> ExplanationReport:
    """Analyze PNTC anomaly detection outputs and generate structured morphological explanation.
    
    Args:
        inp_or_rgb: ExplainInput dataclass instance, or RGB image array (H x W x 3).
        xyz: Registered (H x W x 3) point cloud map (if inp_or_rgb is array).
        anomaly_map: (H x W) continuous anomaly heatmap (if inp_or_rgb is array).
        image_score: Scalar image anomaly score.
        category: Object category name.
        sample_id: Identifier for sample.
        rgb_anomaly_map: Optional component RGB anomaly map.
        xyz_anomaly_map: Optional component XYZ anomaly map.
        topology_map: Optional normalized JS divergence topology map.
        confidence_gate_map: Optional confidence gate map.
        valid_xyz_mask: Optional boolean valid geometry mask.
        coordinate_unit: Declared unit ('mm', 'm', 'unknown').
        threshold: Operating threshold.
        min_area_px: Minimum pixel count for defect region filtering.
        max_regions: Maximum defect regions to extract and characterize.
        
    Returns:
        ExplanationReport containing structured defect characterization and text.
    """
    # 1. Standardize input
    if isinstance(inp_or_rgb, ExplainInput):
        inp = inp_or_rgb
    else:
        if xyz is None or anomaly_map is None:
            raise ValueError("When passing raw arrays, both 'xyz' and 'anomaly_map' are required.")
        inp = ExplainInput(
            rgb=inp_or_rgb,
            xyz=xyz,
            anomaly_map=anomaly_map,
            image_score=float(image_score if image_score is not None else np.max(anomaly_map)),
            category=category,
            sample_id=sample_id,
            rgb_anomaly_map=rgb_anomaly_map,
            xyz_anomaly_map=xyz_anomaly_map,
            topology_map=topology_map,
            confidence_gate_map=confidence_gate_map,
            valid_xyz_mask=valid_xyz_mask,
            coordinate_unit=coordinate_unit,
            threshold=threshold
        )

    # 2. Validate and scale coordinates
    unit_mgr = CoordinateUnitManager(declared_unit=inp.coordinate_unit)
    scaled_xyz, coord_unit, scale_factor, is_calibrated = unit_mgr.validate_and_scale(
        xyz=inp.xyz,
        valid_mask=inp.valid_xyz_mask
    )

    # Derive valid geometry mask if not provided
    if inp.valid_xyz_mask is not None:
        valid_geom = inp.valid_xyz_mask
    else:
        valid_geom = ~np.isnan(inp.xyz).any(axis=-1) & ~np.isinf(inp.xyz).any(axis=-1) & (np.abs(inp.xyz).sum(axis=-1) > 1e-6)

    # Foreground object mask
    object_mask = valid_geom.copy()

    # 3. Extract connected anomaly regions
    raw_regions, op_thresh, binary_mask = extract_anomaly_regions(
        anomaly_map=inp.anomaly_map,
        threshold=inp.threshold,
        min_area_px=min_area_px,
        max_regions=max_regions
    )

    if len(raw_regions) == 0:
        return build_explanation_report(
            inp=inp,
            defects=[],
            operating_threshold=op_thresh,
            coordinate_unit=coord_unit,
            physical_measurements_available=is_calibrated
        )

    # 4. Characterize each defect region
    defects = []
    for reg in raw_regions:
        comp_mask = reg["mask"]
        comp_id = reg["id"]

        # 2D Morphology
        morph_metrics, shape_label = compute_2d_morphology(comp_mask)
        shape_info = ShapeInfo(
            label=shape_label,
            aspect_ratio=morph_metrics["aspect_ratio"],
            circularity=morph_metrics["circularity"],
            solidity=morph_metrics["solidity"],
            extent=morph_metrics["extent"],
            eccentricity=morph_metrics["eccentricity"],
            compactness=morph_metrics["compactness"],
            orientation_deg=morph_metrics["orientation_deg"]
        )

        # Spatial Location
        loc_info = describe_location(
            centroid_px=morph_metrics["centroid_float"],
            image_shape=inp.anomaly_map.shape,
            object_mask=object_mask
        )

        # 3D Physical Size
        size_info = compute_3d_geometry_size(
            component_mask=comp_mask,
            scaled_xyz=scaled_xyz,
            valid_xyz_mask=valid_geom,
            is_physical_calibrated=is_calibrated,
            object_mask=object_mask
        )

        # Local Surface Deviation Analysis
        geo_info = analyze_local_surface(
            component_mask=comp_mask,
            anomaly_map=inp.anomaly_map,
            scaled_xyz=scaled_xyz,
            valid_xyz_mask=valid_geom,
            is_physical_calibrated=is_calibrated,
            threshold=op_thresh,
            unit=coord_unit
        )

        # Visual Appearance Contrast
        app_info = analyze_appearance(
            rgb=inp.rgb,
            component_mask=comp_mask,
            anomaly_map=inp.anomaly_map,
            threshold=op_thresh
        )

        # Cross-modal Detection Evidence
        ev_info = aggregate_detection_evidence(
            component_mask=comp_mask,
            anomaly_map=inp.anomaly_map,
            rgb_map=inp.rgb_anomaly_map,
            xyz_map=inp.xyz_anomaly_map,
            topology_map=inp.topology_map,
            gate_map=inp.confidence_gate_map
        )

        # Natural Language Rationale
        rationale_text = generate_region_explanation(
            location=loc_info,
            size=size_info,
            shape=shape_info,
            geometry=geo_info,
            appearance=app_info,
            evidence=ev_info
        )

        defect = DefectRegion(
            id=comp_id,
            location=loc_info,
            size=size_info,
            shape=shape_info,
            geometry=geo_info,
            appearance=app_info,
            evidence=ev_info,
            mean_anomaly_score=reg["mean_score"],
            max_anomaly_score=reg["max_score"],
            integrated_importance=reg["importance"],
            explanation=rationale_text,
            bounding_box_xyxy=morph_metrics["bbox"],
            mask=comp_mask
        )
        defects.append(defect)

    # 5. Assemble final report
    report = build_explanation_report(
        inp=inp,
        defects=defects,
        operating_threshold=op_thresh,
        coordinate_unit=coord_unit,
        physical_measurements_available=is_calibrated
    )

    return report
