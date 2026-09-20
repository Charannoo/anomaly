"""Report builder: structured JSON, short summaries, and technical text reports."""

from typing import List, Optional
import numpy as np
from .schema import DefectRegion, ExplanationReport, ExplainInput


def build_explanation_report(
    inp: ExplainInput,
    defects: List[DefectRegion],
    operating_threshold: float,
    coordinate_unit: str,
    physical_measurements_available: bool
) -> ExplanationReport:
    """Construct complete ExplanationReport containing structured data and human-readable text.
    
    Args:
        inp: ExplainInput input contract.
        defects: List of extracted DefectRegion instances.
        operating_threshold: Operating threshold used.
        coordinate_unit: 'mm', 'm', or 'unknown'.
        physical_measurements_available: Boolean flag.
        
    Returns:
        ExplanationReport instance.
    """
    decision = "anomalous" if len(defects) > 0 else "normal"
    
    # Detection confidence based on distance of image score above threshold
    if decision == "anomalous":
        score_margin = max(0.0, inp.image_score - operating_threshold)
        detection_conf = float(np.clip(0.70 + 0.30 * (score_margin / max(1e-4, 1.0 - operating_threshold)), 0.70, 0.999))
    else:
        score_margin = max(0.0, operating_threshold - inp.image_score)
        detection_conf = float(np.clip(0.70 + 0.30 * (score_margin / max(1e-4, operating_threshold)), 0.70, 0.999))

    # Build short summary
    if decision == "normal":
        summary_exp = f"No anomalous regions exceeded the operating threshold ({operating_threshold:.4f}). The inspected {inp.category} surface conforms to normal prototype specifications."
    else:
        if len(defects) == 1:
            d = defects[0]
            if d.size.physical_units_available and d.size.major_length_mm is not None:
                size_phrase = f"{d.size.major_length_mm} mm × {d.size.minor_length_mm} mm"
            else:
                size_phrase = f"{d.size.area_px} px"
            summary_exp = (
                f"Anomaly detected in {inp.category} sample {inp.sample_id} ({detection_conf * 100:.1f}% confidence). "
                f"Defect #1 ({d.shape.label}, {size_phrase}) located in {d.location.label} exhibiting {d.geometry.label} "
                f"with {d.evidence.topology_level.lower()} cross-modal topology disagreement."
            )
        else:
            summary_exp = (
                f"Multiple anomalies ({len(defects)} regions) detected in {inp.category} sample {inp.sample_id} "
                f"({detection_conf * 100:.1f}% confidence). Primary defect located in {defects[0].location.label} "
                f"({defects[0].shape.label}, {defects[0].geometry.label})."
            )

    # Build technical report
    tech_report = _generate_technical_report_text(
        sample_id=inp.sample_id,
        category=inp.category,
        decision=decision,
        image_score=inp.image_score,
        detection_conf=detection_conf,
        threshold=operating_threshold,
        defects=defects,
        coord_unit=coordinate_unit,
        physical_avail=physical_measurements_available
    )

    return ExplanationReport(
        sample_id=inp.sample_id,
        category=inp.category,
        decision=decision,
        image_anomaly_score=inp.image_score,
        detection_confidence=detection_conf,
        operating_threshold=operating_threshold,
        num_defects=len(defects),
        defects=defects,
        summary_explanation=summary_exp,
        technical_report=tech_report,
        coordinate_unit=coordinate_unit,
        physical_measurements_available=physical_measurements_available
    )


def _generate_technical_report_text(
    sample_id: str,
    category: str,
    decision: str,
    image_score: float,
    detection_conf: float,
    threshold: float,
    defects: List[DefectRegion],
    coord_unit: str,
    physical_avail: bool
) -> str:
    """Generate detailed technical ASCII text report."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"PNTC DEFECT CHARACTERIZATION REPORT — {category.upper()} (Sample: {sample_id})")
    lines.append("=" * 80)
    lines.append(f"Overall Decision     : {decision.upper()}")
    lines.append(f"Image Anomaly Score  : {image_score:.5f}")
    lines.append(f"Detection Confidence : {detection_conf * 100:.1f}%")
    lines.append(f"Operating Threshold  : {threshold:.5f}")
    lines.append(f"Coordinate Units     : {coord_unit} (Physical Calibrated: {physical_avail})")
    lines.append(f"Defect Regions Count : {len(defects)}")
    lines.append("-" * 80)

    if len(defects) == 0:
        lines.append("RESULT: Normal sample. No defect regions identified above threshold.")
        lines.append("=" * 80)
        return "\n".join(lines)

    for d in defects:
        lines.append(f"\n[DEFECT REGION #{d.id}] (Importance Rank: #{d.id})")
        lines.append(f"  Location            : {d.location.label} (Normalized: {d.location.centroid_normalized})")
        if d.size.physical_units_available:
            lines.append(f"  Bounding Dimensions : {d.size.major_length_mm} mm × {d.size.minor_length_mm} mm (Depth: {d.size.bounding_depth_mm} mm)")
            lines.append(f"  Estimated 3D Area   : {d.size.surface_area_3d_mm2} mm² (Projected: {d.size.projected_area_mm2} mm²)")
        else:
            lines.append(f"  Bounding Dimensions : {d.size.width_px} px × {d.size.height_px} px")
            lines.append(f"  Pixel Area          : {d.size.area_px} pixels ({d.size.area_fraction_object * 100:.2f}% of object)")

        lines.append(f"  2D Morphology       : {d.shape.label} (Aspect Ratio: {d.shape.aspect_ratio}, Circularity: {d.shape.circularity})")
        lines.append(f"  3D Surface Structure: {d.geometry.label} (Z-score: {d.geometry.surface_z_score}, Roughness Ratio: {d.geometry.roughness_ratio}×)")
        
        if d.geometry.max_depression is not None:
            lines.append(f"  Maximum Depression  : {d.geometry.max_depression} {d.geometry.unit}")
        if d.geometry.max_protrusion is not None:
            lines.append(f"  Maximum Protrusion  : {d.geometry.max_protrusion} {d.geometry.unit}")
            
        lines.append(f"  Appearance Contrast : {d.appearance.deviation_label.upper()} (CIE Lab ΔE = {d.appearance.delta_e_lab})")
        lines.append(f"  Detection Evidence  : RGB={d.evidence.rgb_level} ({d.evidence.rgb_evidence_score}), 3D={d.evidence.xyz_level} ({d.evidence.xyz_evidence_score}), Topology Disagreement={d.evidence.topology_level} ({d.evidence.topology_disagreement_score})")
        lines.append(f"  Measurement Conf.   : {d.geometry.measurement_confidence * 100:.1f}%")
        lines.append(f"  Explanation Rationale:")
        lines.append(f"    \"{d.explanation}\"")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)
