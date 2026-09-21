"""FastAPI routes for verified industrial demo cases."""

import json
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter

router = APIRouter(prefix="/api/demo", tags=["demo"])
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEMO_DIR = PROJECT_ROOT / "results" / "demo_cases"


@router.get("/cases")
def list_demo_cases() -> List[Dict[str, Any]]:
    """List real verified industrial inspection cases available for one-click demo."""
    cases = [
        {
            "id": "01_strong_defect",
            "name": "01 — Strong Dual-Modal Defect",
            "category": "cookie",
            "type": "Combined Defect",
            "expected_decision": "anomalous",
            "expected_score": 0.88,
            "highlight": "Large localized depression: 3.37 mm depth, 620.4 mm³ missing volume",
            "evidence": "High RGB + High 3D + Very High Topology Disagreement",
            "thumbnail_url": "/api/inspections/INSP-01_strong_defect/artifacts/overlay",
        },
        {
            "id": "02_primarily_rgb_defect",
            "name": "02 — Primarily RGB Appearance Defect",
            "category": "cookie",
            "type": "Surface Contamination",
            "expected_decision": "anomalous",
            "expected_score": 0.74,
            "highlight": "Significant RGB color blemish with flat, nominal 3D surface profile",
            "evidence": "High RGB + Low 3D + Disagreement",
            "thumbnail_url": "/api/inspections/INSP-02_primarily_rgb_defect/artifacts/overlay",
        },
        {
            "id": "03_primarily_geometric_defect",
            "name": "03 — Primarily Geometric Structural Defect",
            "category": "cookie",
            "type": "Surface Dent / Void",
            "expected_decision": "anomalous",
            "expected_score": 0.82,
            "highlight": "Shallow physical dent without perceptible texture or color alteration",
            "evidence": "Low RGB + High 3D + Disagreement",
            "thumbnail_url": "/api/inspections/INSP-03_primarily_geometric_defect/artifacts/overlay",
        },
        {
            "id": "04_strong_topology_disagreement",
            "name": "04 — Strong Cross-Modal Topology Disagreement",
            "category": "cookie",
            "type": "Structural Conflict",
            "expected_decision": "anomalous",
            "expected_score": 0.91,
            "highlight": "RGB and 3D map to disjoint prototype clusters (JS Div: 1.00)",
            "evidence": "Max Topology Divergence (+0.157 bonus evidence)",
            "thumbnail_url": "/api/inspections/INSP-04_strong_topology_disagreement/artifacts/overlay",
        },
        {
            "id": "05_borderline_manual_review",
            "name": "05 — Borderline Uncertainty / Review Guard",
            "category": "cookie",
            "type": "Near-Threshold Anomaly",
            "expected_decision": "manual_review",
            "expected_score": 0.52,
            "highlight": "Low measurement certainty; triggers automated Review Guard flag",
            "evidence": "Marginal Threshold Crossing",
            "thumbnail_url": "/api/inspections/INSP-05_borderline_manual_review/artifacts/overlay",
        },
        {
            "id": "06_failure_limitation_boundary",
            "name": "06 — Boundary Limitation Transparency",
            "category": "cookie",
            "type": "Edge / Boundary Artifact",
            "expected_decision": "anomalous",
            "expected_score": 0.71,
            "highlight": "Demonstrates boundary proximity warning without false certainty",
            "evidence": "Boundary Uncertainty Detected",
            "thumbnail_url": "/api/inspections/INSP-06_failure_limitation_boundary/artifacts/overlay",
        },
        {
            "id": "07_nominal_sample",
            "name": "07 — Nominal Potato (Normal / Pass)",
            "category": "potato",
            "type": "Nominal Production",
            "expected_decision": "normal",
            "expected_score": 0.12,
            "highlight": "Zero defects detected; full compliance with nominal surface envelope",
            "evidence": "All modalities nominal",
            "thumbnail_url": "/api/inspections/INSP-07_nominal_sample/artifacts/overlay",
        },
        {
            "id": "08_nominal_cookie",
            "name": "08 — Nominal Cookie (Normal / Pass)",
            "category": "cookie",
            "type": "Nominal Production",
            "expected_decision": "normal",
            "expected_score": 0.11,
            "highlight": "Zero defects detected; nominal appearance and physical surface envelope",
            "evidence": "All modalities nominal",
            "thumbnail_url": "/api/inspections/INSP-08_nominal_cookie/artifacts/rgb",
        },
    ]
    return cases
