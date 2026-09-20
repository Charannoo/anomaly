"""Tests for rule-based explanation generation and normal sample behavior."""

import numpy as np
import pytest
from xmvad.explain import explain_anomaly, ExplainInput


def test_normal_sample_clean_behavior():
    h, w = 60, 60
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 180
    xyz = np.ones((h, w, 3), dtype=np.float32) * 50.0
    anomaly_map = np.zeros((h, w), dtype=np.float32)  # Low scores everywhere
    valid_mask = np.ones((h, w), dtype=bool)

    inp = ExplainInput(
        rgb=rgb,
        xyz=xyz,
        anomaly_map=anomaly_map,
        image_score=0.12,
        category="cookie",
        sample_id="normal_001",
        valid_xyz_mask=valid_mask,
        threshold=0.5
    )

    report = explain_anomaly(inp)
    assert report.decision == "normal"
    assert report.num_defects == 0
    assert len(report.defects) == 0
    assert "conforms to normal prototype specifications" in report.summary_explanation


def test_dual_modality_evidence_explanation():
    h, w = 60, 60
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 180
    xyz = np.ones((h, w, 3), dtype=np.float32) * 50.0
    
    defect_mask = np.zeros((h, w), dtype=bool)
    defect_mask[15:45, 15:45] = True
    xyz[defect_mask, 2] -= 2.0  # Depression
    rgb[defect_mask, 0] -= 50    # Color shift

    anomaly_map = np.zeros((h, w), dtype=np.float32)
    anomaly_map[defect_mask] = 0.88
    
    # Topology map with high disagreement
    topo_map = np.zeros((h, w), dtype=np.float32)
    topo_map[defect_mask] = 0.92

    inp = ExplainInput(
        rgb=rgb,
        xyz=xyz,
        anomaly_map=anomaly_map,
        image_score=0.88,
        category="foam",
        sample_id="anom_002",
        topology_map=topo_map,
        valid_xyz_mask=np.ones((h, w), dtype=bool),
        coordinate_unit="mm",
        threshold=0.5
    )

    report = explain_anomaly(inp)
    assert report.decision == "anomalous"
    assert report.num_defects == 1
    d = report.defects[0]
    assert d.evidence.topology_level in ["High", "Very High"]
    assert "topology disagreement" in d.explanation.lower()
