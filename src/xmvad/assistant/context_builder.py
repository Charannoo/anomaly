"""Structured, factual context builder for the PNTC AI Assistant.

Converts verified InspectionReport objects or dictionaries into compact,
factual JSON dictionaries suitable for LLM consumption, omitting any null
or unmeasured values. Returns (context_dict, fields_used_list).
"""

from typing import Dict, Any, List, Optional, Set, Union, Tuple
from xmvad.inspection.schema import InspectionReport


def _extract_fields_used(data: Any, prefix: str = "") -> List[str]:
    """Recursively collect dotted paths of populated factual fields."""
    fields = []
    if isinstance(data, dict):
        for k, v in data.items():
            path = f"{prefix}.{k}" if prefix else k
            if v is not None and v != "" and v != []:
                if k == "defects" and isinstance(v, list):
                    for item in v:
                        fields.extend(_extract_fields_used(item, ""))
                elif isinstance(v, (dict, list)):
                    sub = _extract_fields_used(v, path)
                    if sub:
                        fields.extend(sub)
                    else:
                        fields.append(path)
                else:
                    fields.append(path)
    return list(dict.fromkeys(fields))  # unique, preserve order


def build_assistant_context(
    report_or_dict: Union[InspectionReport, Dict[str, Any]],
    selected_defect_id: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Build compact, purely factual context dictionary from inspection results.
    
    Args:
        report_or_dict: InspectionReport instance or serialized dictionary.
        selected_defect_id: Optional defect ID filter.
        
    Returns:
        Tuple of (context_dict, fields_used_list).
    """
    if isinstance(report_or_dict, InspectionReport):
        data = report_or_dict.to_dict()
    elif isinstance(report_or_dict, dict):
        data = report_or_dict
    else:
        data = {}

    # Case 1: Already structured context dictionary
    if "sample" in data and "inspection" in data and "defects" in data:
        context = {
            "sample": dict(data["sample"]),
            "inspection": dict(data["inspection"]),
            "defects": []
        }
        for d in data["defects"]:
            did = d.get("id")
            if selected_defect_id is not None and did is not None and did != selected_defect_id:
                continue
            context["defects"].append(dict(d))

        fields_used = _extract_fields_used(context)
        return context, fields_used

    # Case 2: Standard InspectionReport or raw dict format
    sample_id = data.get("sample_id") or data.get("sample", {}).get("id", "unknown_sample")
    category = data.get("category") or data.get("sample", {}).get("category", "unknown_category")
    
    insp = data.get("inspection", {})
    pntc = data.get("pntc", {})
    
    decision = insp.get("decision") or pntc.get("decision", "normal")
    score = insp.get("anomaly_score")
    if score is None:
        score = pntc.get("score", 0.0)
    score = round(float(score), 4)

    context: Dict[str, Any] = {
        "sample": {
            "id": sample_id,
            "category": category,
        },
        "inspection": {
            "decision": decision,
            "anomaly_score": score,
        }
    }

    if "inspection_status" in insp:
        context["inspection"]["inspection_status"] = insp["inspection_status"]
    elif "inspection_status" in data:
        context["inspection"]["inspection_status"] = data["inspection_status"]

    if "reliability" in insp:
        context["inspection"]["reliability"] = round(float(insp["reliability"]), 4)
    elif "decision_certainty" in data:
        context["inspection"]["decision_certainty"] = data["decision_certainty"]

    if "manual_review_recommended" in insp:
        context["inspection"]["manual_review_recommended"] = bool(insp["manual_review_recommended"])
    if "review_reasons" in insp:
        context["inspection"]["review_reasons"] = insp["review_reasons"]

    raw_defects = data.get("defects", [])
    formatted_defects: List[Dict[str, Any]] = []

    for d in raw_defects:
        defect_id = int(d.get("id", 1))
        if selected_defect_id is not None and defect_id != selected_defect_id:
            continue

        defect_entry: Dict[str, Any] = {"id": defect_id}

        # Handle direct fields or nested inspection report fields
        if "location" in d and d["location"]:
            defect_entry["location"] = d["location"]
        if "shape" in d and d["shape"]:
            defect_entry["shape"] = d["shape"]
        elif "morphology" in d and d["morphology"]:
            m = d["morphology"]
            defect_entry["shape"] = {
                "label": m.get("shape_label"),
                "aspect_ratio": m.get("aspect_ratio"),
                "orientation_deg": m.get("orientation_deg"),
            }
            defect_entry["size"] = {
                "major_length_mm": m.get("major_axis_length"),
                "minor_length_mm": m.get("minor_axis_length"),
                "surface_area_mm2": m.get("area_pixels_or_mm2"),
            }

        if "size" in d and d["size"]:
            defect_entry["size"] = d["size"]

        if "geometry" in d and d["geometry"]:
            defect_entry["geometry"] = d["geometry"]

        if "volume" in d and d["volume"]:
            defect_entry["volume"] = d["volume"]

        if "evidence" in d and d["evidence"]:
            defect_entry["evidence"] = d["evidence"]

        if "normal_reference" in d and d["normal_reference"]:
            defect_entry["normal_reference"] = d["normal_reference"]
        elif "normal_twin" in d and d["normal_twin"]:
            nt = d["normal_twin"]
            defect_entry["normal_reference"] = {
                "prototype_id": nt.get("prototype_id"),
                "training_sample_id": nt.get("training_sample_id"),
            }

        if "prototype_trace" in d and d["prototype_trace"]:
            defect_entry["prototype_trace"] = d["prototype_trace"]

        if "quality" in d and d["quality"]:
            defect_entry["quality"] = d["quality"]

        formatted_defects.append(defect_entry)

    context["defects"] = formatted_defects
    fields_used = _extract_fields_used(context)
    return context, fields_used


def extract_context_numbers(context: Dict[str, Any]) -> Set[float]:
    """Recursively extract all numeric measurements from context dictionary for grounding checks."""
    numbers: Set[float] = set()

    def _walk(val: Any):
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            numbers.add(round(float(val), 2))
        elif isinstance(val, dict):
            for v in val.values():
                _walk(v)
        elif isinstance(val, (list, tuple)):
            for v in val:
                _walk(v)

    _walk(context)
    return numbers
