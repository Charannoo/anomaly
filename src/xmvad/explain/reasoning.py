"""Deterministic rule-based reasoning engine for evidence-grounded anomaly explanations."""

from typing import Dict, Any
from .schema import LocationInfo, SizeInfo, ShapeInfo, GeometryInfo, AppearanceInfo, EvidenceInfo


def generate_region_explanation(
    location: LocationInfo,
    size: SizeInfo,
    shape: ShapeInfo,
    geometry: GeometryInfo,
    appearance: AppearanceInfo,
    evidence: EvidenceInfo
) -> str:
    """Generate a deterministic, evidence-grounded human-readable explanation for an anomaly region.
    
    Args:
        location: Location descriptor.
        size: Geometric size info.
        shape: 2D morphology info.
        geometry: 3D surface deviation info.
        appearance: Visual appearance info.
        evidence: PNTC cross-modal detection evidence.
        
    Returns:
        A concise, factually grounded natural language explanation paragraph.
    """
    sentences = []

    # Sentence 1: Physical morphology and location summary
    loc_str = location.label
    shape_str = shape.label
    
    if size.physical_units_available and size.major_length_mm is not None:
        dim_str = f"measuring approximately {size.major_length_mm} mm × {size.minor_length_mm} mm"
    else:
        dim_str = f"covering {size.area_px} pixels ({round(size.area_fraction_object * 100, 1)}% of the object surface)"

    sentences.append(f"A {shape_str} was detected in the {loc_str}, {dim_str}.")

    # Sentence 2: 3D geometric structure description
    geo_label = geometry.label
    if geo_label == "localized depression" and geometry.max_depression is not None:
        unit = geometry.unit if geometry.unit != "unknown" else "units"
        sentences.append(
            f"The local 3D surface shows a recessed depression with a maximum depth of {geometry.max_depression} {unit} "
            f"(surface deviation z-score = {geometry.surface_z_score})."
        )
    elif geo_label == "raised protrusion" and geometry.max_protrusion is not None:
        unit = geometry.unit if geometry.unit != "unknown" else "units"
        sentences.append(
            f"The geometry exhibits a raised protrusion extending up to {geometry.max_protrusion} {unit} above the local reference plane "
            f"(surface deviation z-score = {geometry.surface_z_score})."
        )
    elif "irregular" in geo_label:
        sentences.append(
            f"The region exhibits irregular geometric deformation with an elevated roughness ratio of {geometry.roughness_ratio}× "
            f"relative to the surrounding surface."
        )
    elif geo_label == "primarily appearance-based anomaly":
        sentences.append("The 3D geometric surface remains largely flush with the surrounding reference plane.")

    # Sentence 3: Visual appearance contrast
    if appearance.deviation_label == "strong":
        sentences.append(
            f"Visual inspection reveals strong appearance deviation from the normal reference ring (CIE Lab ΔE = {appearance.delta_e_lab})."
        )
    elif appearance.deviation_label == "moderate":
        sentences.append(
            f"Moderate visual contrast is observed relative to surrounding normal textures (ΔE = {appearance.delta_e_lab})."
        )

    # Sentence 4: Cross-modal topology and detection rationale
    rgb_lvl = evidence.rgb_level
    xyz_lvl = evidence.xyz_level
    topo_lvl = evidence.topology_level

    if topo_lvl in ["High", "Very High"] and (xyz_lvl in ["High", "Very High"] or rgb_lvl in ["High", "Very High"]):
        sentences.append(
            f"PNTC flagged this region because RGB and 3D modalities retrieve incompatible normal prototype neighborhoods "
            f"(topology disagreement: {topo_lvl}, score = {evidence.topology_disagreement_score}), coupled with {xyz_lvl.lower()} 3D geometric evidence."
        )
    elif rgb_lvl in ["High", "Very High"] and xyz_lvl == "Low":
        sentences.append(
            f"Visual discrepancy is the dominant detection indicator (RGB evidence: {rgb_lvl}), while 3D geometry remains consistent with normal prototypes."
        )
    elif xyz_lvl in ["High", "Very High"] and rgb_lvl == "Low":
        sentences.append(
            f"Geometric deformation is the dominant detection indicator (3D evidence: {xyz_lvl}), despite surface RGB appearance appearing normal."
        )
    else:
        sentences.append(
            f"Both visual and geometric branches contribute consistent anomaly evidence (RGB: {rgb_lvl}, 3D: {xyz_lvl}, Topology: {topo_lvl})."
        )

    return " ".join(sentences)
