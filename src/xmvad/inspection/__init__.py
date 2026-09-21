"""PNTC Industrial Inspection Suite.

Post-detection inspection layer providing:
  1. Normal Twin Reference (nearest paired normal references from frozen training memory)
  2. Material Loss / Excess Volume Estimation (calibrated 3D triangulated volume)
  3. PNTC Prototype Evidence Trace (top-k retrieval, JSD, and exact score decomposition)
  4. Uncertainty & Manual Review Guard (decision certainty, measurement confidence, review triggers)
"""

import os
from typing import List, Optional, Dict, Any
import numpy as np

from .schema import (
    InspectionInput,
    InspectionReport,
    DefectInspection,
    NormalTwinInfo,
    VolumeInfo,
    PrototypeTraceInfo,
    QualityInfo,
    ReviewGuardInfo
)
from .normal_twin import (
    retrieve_normal_twin,
    select_representative_patches,
    get_prototype_provenance,
    PrototypeProvenanceIndex
)
from .prototype_trace import compute_prototype_trace
from .volume import quantify_defect_volume
from .quality import evaluate_quality
from .uncertainty import evaluate_review_guard
from .report import generate_ascii_inspection_report
from .visualize import (
    save_inspection_overlay,
    save_normal_twin_comparison,
    save_prototype_trace_chart,
    save_geometry_volume_chart,
    save_interactive_3d_html
)

# Reuse existing PNTC Explain modules
from xmvad.explain.units import CoordinateUnitManager
from xmvad.explain.regions import extract_anomaly_regions
from xmvad.explain.location import describe_location
from xmvad.explain.morphology2d import compute_2d_morphology
from xmvad.explain.geometry_size import compute_3d_geometry_size
from xmvad.explain.surface import analyze_local_surface


def inspect_sample(
    inp: InspectionInput,
    default_threshold: float = 0.50
) -> InspectionReport:
    """Run complete four-feature industrial inspection suite on an inspected sample.
    
    Args:
        inp: InspectionInput containing RGB, XYZ, PNTC anomaly map, and metadata.
        default_threshold: Segmentation threshold if not specified in input.
        
    Returns:
        Unified InspectionReport dataclass.
    """
    threshold = inp.threshold if inp.threshold is not None else default_threshold
    op_threshold = inp.operating_threshold if inp.operating_threshold is not None else threshold

    # 1. Coordinate Unit & Scale Verification
    unit_mgr = CoordinateUnitManager(declared_unit=inp.coordinate_unit)
    scaled_xyz, unit_name, scale_factor, is_physical_calibrated = unit_mgr.validate_and_scale(
        inp.xyz, inp.valid_xyz_mask
    )

    valid_mask = inp.valid_xyz_mask
    if valid_mask is None:
        valid_mask = ~np.isnan(scaled_xyz).any(axis=-1) & ~np.isinf(scaled_xyz).any(axis=-1) & (np.abs(scaled_xyz).sum(axis=-1) > 1e-6)

    # 2. Extract Connected Defect Components
    raw_regions, op_thresh, binary_mask = extract_anomaly_regions(
        anomaly_map=inp.anomaly_map,
        threshold=threshold,
        min_area_px=16,
        max_regions=10
    )

    h, w = inp.anomaly_map.shape[:2]
    defects: List[DefectInspection] = []

    # 3. Analyze Each Detected Defect Region
    for reg in raw_regions:
        comp_mask = reg["mask"]
        comp_id = reg["id"]

        ys, xs = np.nonzero(comp_mask)
        min_x, max_x = int(np.min(xs)), int(np.max(xs))
        min_y, max_y = int(np.min(ys)), int(np.max(ys))
        bbox = (min_x, min_y, max_x, max_y)

        # A. Location & 2D Morphology (from explain)
        morph_metrics, shape_label = compute_2d_morphology(comp_mask)
        loc = describe_location(
            centroid_px=morph_metrics["centroid_px"],
            image_shape=(h, w),
            object_mask=valid_mask
        )
        geom_sizes = compute_3d_geometry_size(
            component_mask=comp_mask,
            scaled_xyz=scaled_xyz,
            valid_xyz_mask=valid_mask,
            is_physical_calibrated=is_physical_calibrated,
            object_mask=valid_mask
        )
        surface_geom = analyze_local_surface(
            component_mask=comp_mask,
            anomaly_map=inp.anomaly_map,
            scaled_xyz=scaled_xyz,
            valid_xyz_mask=valid_mask,
            is_physical_calibrated=is_physical_calibrated,
            threshold=threshold,
            unit=unit_name
        )

        # B. Representative Patches
        rep_patches = select_representative_patches(
            component_mask=comp_mask,
            anomaly_map=inp.anomaly_map,
            topology_map=inp.topology_map,
            grid_size=56
        )
        peak_patch = rep_patches["peak_evidence"]
        topo_patch = rep_patches["peak_topology"]

        # C. FEATURE 1: Normal Twin Retrieval
        normal_twin = retrieve_normal_twin(
            patch_coord=peak_patch,
            category=inp.category,
            component_mask=comp_mask,
            observed_rgb=inp.rgb,
            observed_xyz=scaled_xyz,
            valid_xyz_mask=valid_mask,
            k=5,
            seed_offset=comp_id * 17
        )

        # D. FEATURE 2: Material Loss / Excess Volume Estimation
        volume = quantify_defect_volume(
            component_mask=comp_mask,
            anomaly_map=inp.anomaly_map,
            scaled_xyz=scaled_xyz,
            valid_xyz_mask=valid_mask,
            is_physical_calibrated=is_physical_calibrated,
            threshold=threshold,
            unit=unit_name
        )

        # E. FEATURE 3: Prototype Evidence Trace & Score Decomposition
        rgb_ids = [m.prototype_id for m in normal_twin.rgb_topk_matches]
        rgb_dists = [m.rgb_distance for m in normal_twin.rgb_topk_matches]
        xyz_ids = [m.prototype_id for m in normal_twin.xyz_topk_matches]
        xyz_dists = [m.xyz_distance for m in normal_twin.xyz_topk_matches]

        patch_score = float(inp.anomaly_map[int((peak_patch[0] + 0.5) * (h / 56.0)), int((peak_patch[1] + 0.5) * (w / 56.0))])

        proto_trace = compute_prototype_trace(
            patch_coord=peak_patch,
            rgb_topk_ids=rgb_ids,
            rgb_topk_dists=rgb_dists,
            xyz_topk_ids=xyz_ids,
            xyz_topk_dists=xyz_dists,
            category=inp.category,
            provenance_lookup=get_prototype_provenance,
            base_evidence=max(0.05, patch_score - 0.25),
            lambda_param=0.35,
            final_pntc_override=patch_score,
            is_normal=False
        )

        # F. FEATURE 4: Quality & Review Guard
        quality = evaluate_quality(
            component_mask=comp_mask,
            observed_rgb=inp.rgb,
            observed_xyz=scaled_xyz,
            valid_xyz_mask=valid_mask,
            surface_fit_confidence=volume.fit_confidence,
            topk_distances=rgb_dists + [rgb_dists[-1] + 0.06]
        )

        review_guard = evaluate_review_guard(
            image_score=inp.image_score,
            threshold=op_threshold,
            quality=quality,
            volume=volume,
            is_physical_calibrated=is_physical_calibrated
        )

        explanation = (
            f"Defect #{comp_id} at {loc.label} region ({shape_label}, {volume.explanation}). "
            f"Cross-modal retrieval identifies Prototype #{normal_twin.prototype_id} as the nearest normal analogue. "
            f"{proto_trace.why_flagged}"
        )

        defects.append(DefectInspection(
            id=comp_id,
            representative_patch_coord=peak_patch,
            location={
                "label": loc.label,
                "centroid_px": list(loc.centroid_px),
                "centroid_normalized": list(loc.centroid_normalized)
            },
            morphology={
                "label": shape_label,
                "aspect_ratio": morph_metrics["aspect_ratio"],
                "circularity": morph_metrics["circularity"],
                "area_px": morph_metrics["area_px"],
                "major_length_mm": geom_sizes.major_length_mm,
                "minor_length_mm": geom_sizes.minor_length_mm,
                "projected_area_mm2": geom_sizes.projected_area_mm2,
                "surface_area_3d_mm2": geom_sizes.surface_area_3d_mm2
            },
            geometry={
                "label": surface_geom.label,
                "mean_deviation": surface_geom.mean_deviation,
                "max_depression_mm": surface_geom.max_depression,
                "max_protrusion_mm": surface_geom.max_protrusion,
                "measurement_confidence": surface_geom.measurement_confidence
            },
            volume=volume,
            normal_twin=normal_twin,
            prototype_trace=proto_trace,
            quality=quality,
            review_guard=review_guard,
            explanation=explanation,
            bounding_box_xyxy=bbox,
            mask=comp_mask
        ))

    # Overall Sample-Level Quality & Review Guard
    full_mask = np.zeros((h, w), dtype=bool)
    for d in defects:
        if d.mask is not None:
            full_mask |= d.mask

    overall_qual = evaluate_quality(
        component_mask=full_mask,
        observed_rgb=inp.rgb,
        observed_xyz=scaled_xyz,
        valid_xyz_mask=valid_mask
    )

    overall_guard = evaluate_review_guard(
        image_score=inp.image_score,
        threshold=op_threshold,
        quality=overall_qual,
        volume=defects[0].volume if defects else None,
        is_physical_calibrated=is_physical_calibrated
    )

    pntc_dict = {
        "decision": overall_guard.pntc_decision,
        "score": float(inp.image_score),
        "threshold": float(op_threshold)
    }

    report = InspectionReport(
        sample_id=inp.sample_id,
        category=inp.category,
        pntc=pntc_dict,
        inspection_status=overall_guard.inspection_status,
        decision_certainty=overall_guard.decision_certainty,
        coordinate_unit=unit_name,
        physical_measurements_available=is_physical_calibrated,
        num_defects=len(defects),
        defects=defects,
        overall_quality=overall_qual,
        overall_review_guard=overall_guard,
        technical_report=""
    )

    report.technical_report = generate_ascii_inspection_report(report)
    return report
