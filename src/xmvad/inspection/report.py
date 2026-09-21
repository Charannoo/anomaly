"""Unified human-readable industrial inspection report generator."""

from typing import List, Dict, Any
from .schema import InspectionReport, DefectInspection


def generate_ascii_inspection_report(report: InspectionReport) -> str:
    """Format an InspectionReport into a standardized industrial technical inspection report.
    
    Adheres strictly to scientific terminology rules:
        - "retrieved normal analogue" / "nearest paired normal reference"
        - "localized depression" / "raised protrusion"
        - "decision certainty" / "measurement reliability"
        - No uncalibrated percentages or unverified mm3 claims.
    """
    lines = []
    lines.append("--------------------------------------------------")
    lines.append("PNTC INDUSTRIAL INSPECTION REPORT")
    lines.append("--------------------------------------------------")
    lines.append("")
    lines.append(f"SAMPLE ID: {report.sample_id}")
    lines.append(f"CATEGORY:  {report.category.upper()}")
    lines.append("")
    lines.append("STATUS:")
    lines.append(report.inspection_status.replace("_", " "))
    lines.append("")
    lines.append("PNTC anomaly score:")
    lines.append(f"{report.pntc.get('score', 0.0):.4f}")
    lines.append("")
    lines.append("Decision certainty:")
    lines.append(report.decision_certainty)
    lines.append("")
    lines.append("Canonical PNTC Decision:")
    lines.append(report.pntc.get("decision", "normal").upper())
    lines.append("")
    lines.append(f"Number of detected defect regions: {report.num_defects}")
    lines.append("")

    if report.num_defects == 0:
        lines.append("No anomalous regions exceed operating threshold.")
        lines.append("Component surface and texture are consistent with the nominal training manifold.")
        lines.append("")
        lines.append("OVERALL MEASUREMENT RELIABILITY:")
        lines.append(report.overall_review_guard.measurement_reliability)
        lines.append("--------------------------------------------------")
        return "\n".join(lines)

    for d in report.defects:
        lines.append(f"DEFECT #{d.id}")
        lines.append("")
        
        # Location
        loc_label = d.location.get("label", "central region") if isinstance(d.location, dict) else str(d.location)
        lines.append("Location:")
        lines.append(loc_label.replace("-", " ").capitalize() + " region")
        lines.append("")

        # Shape
        shape_label = d.morphology.get("label", "irregular") if isinstance(d.morphology, dict) else str(d.morphology)
        lines.append("Shape:")
        lines.append(shape_label.capitalize())
        lines.append("")

        # Physical Size
        if report.physical_measurements_available and "major_length_mm" in d.morphology:
            maj = d.morphology.get("major_length_mm", 0.0)
            min_len = d.morphology.get("minor_length_mm", 0.0)
            lines.append("Physical size:")
            lines.append(f"{maj:.1f} × {min_len:.1f} mm")
            lines.append("")
            
            surf_area = d.morphology.get("surface_area_3d_mm2") or d.morphology.get("projected_area_mm2", 0.0)
            lines.append("Surface area:")
            lines.append(f"{surf_area:.1f} mm²")
            lines.append("")
        else:
            lines.append("Size (2D area):")
            lines.append(f"{d.quality.region_size_px} px")
            lines.append("")

        # Geometry & Surface Structure
        geom_label = d.geometry.get("label", "localized deformation") if isinstance(d.geometry, dict) else str(d.geometry)
        lines.append("Structure:")
        lines.append(geom_label.capitalize())
        lines.append("")

        max_dep = d.geometry.get("max_depression_mm") if isinstance(d.geometry, dict) else None
        if max_dep is not None and max_dep > 0.01:
            lines.append("Maximum depression:")
            lines.append(f"{max_dep:.2f} mm")
            lines.append("")

        max_prot = d.geometry.get("max_protrusion_mm") if isinstance(d.geometry, dict) else None
        if max_prot is not None and max_prot > 0.01:
            lines.append("Maximum protrusion:")
            lines.append(f"{max_prot:.2f} mm")
            lines.append("")

        # Volume
        if d.volume.physical_volume_available:
            if d.volume.missing_material_volume > 0.01:
                lines.append("Estimated missing material:")
                lines.append(f"{d.volume.missing_material_volume:.1f} mm³")
                lines.append("")
            if d.volume.excess_material_volume > 0.01:
                lines.append("Estimated excess material:")
                lines.append(f"{d.volume.excess_material_volume:.1f} mm³")
                lines.append("")
            if d.volume.missing_material_volume > 0.01 and d.volume.excess_material_volume > 0.01:
                lines.append("Net signed volume:")
                lines.append(f"{d.volume.net_signed_volume:+.1f} mm³")
                lines.append("")
        else:
            lines.append("Volumetric estimation (native units):")
            lines.append(f"Net volume: {d.volume.net_signed_volume:+.1f} native units³ (uncalibrated)")
            lines.append("")

        # Evidence Breakdown
        lines.append("RGB evidence:")
        lines.append(d.normal_twin.rgb_similarity)
        lines.append("")
        lines.append("3D evidence:")
        lines.append(d.normal_twin.xyz_similarity)
        lines.append("")
        lines.append("RGB–3D topology disagreement:")
        topo_level = "Very High" if d.prototype_trace.js_divergence > 0.7 else ("High" if d.prototype_trace.js_divergence > 0.4 else "Moderate")
        lines.append(topo_level)
        lines.append("")

        # Normal Reference & Prototype Trace
        lines.append("Nearest normal reference:")
        lines.append(f"Prototype #{d.normal_twin.prototype_id}")
        lines.append(f"Source: {d.normal_twin.training_sample_id}")
        lines.append("")
        lines.append("RGB/XYZ top-5 overlap:")
        lines.append(f"{d.prototype_trace.shared_ids}/5")
        lines.append("")
        lines.append("JS divergence:")
        lines.append(f"{d.prototype_trace.js_divergence:.2f}")
        lines.append("")

        # Score Decomposition
        lines.append("PNTC Score Decomposition:")
        lines.append(f"  Base evidence A_base(p) : {d.prototype_trace.base_anomaly_evidence:.3f}")
        lines.append(f"  Topology divergence T(p): {d.prototype_trace.js_divergence:.3f}")
        lines.append(f"  Confidence gate G(p)    : {d.prototype_trace.gate_value:.3f}")
        lines.append(f"  Weight lambda           : {d.prototype_trace.lambda_param:.2f}")
        lines.append(f"  Topology addition       : +{d.prototype_trace.topology_contribution:.3f}")
        lines.append(f"  Final PNTC evidence     : {d.prototype_trace.final_pntc_evidence:.3f}")
        lines.append("")

        # Why Flagged
        lines.append("WHY FLAGGED:")
        lines.append(d.prototype_trace.why_flagged)
        lines.append("")

        # Reference interpretation
        lines.append("REFERENCE:")
        lines.append(d.normal_twin.interpretation)
        lines.append("")

        # Reliability
        lines.append("MEASUREMENT RELIABILITY:")
        lines.append(d.review_guard.measurement_reliability)
        lines.append("")

        if d.review_guard.reasons:
            lines.append("Inspection Warnings / Review Triggers:")
            for r in d.review_guard.reasons:
                lines.append(f"  - {r}")
            lines.append("")

    lines.append("--------------------------------------------------")
    return "\n".join(lines)
