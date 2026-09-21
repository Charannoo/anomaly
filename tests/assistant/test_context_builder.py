"""Tests for Context Builder, missing fields handling, normal/anomalous samples, and multiple defects."""

import pytest
from xmvad.assistant.context_builder import build_assistant_context, extract_context_numbers


def test_context_builder_anomalous_sample():
    """Verify context builder parses a rich anomalous inspection result."""
    raw_result = {
        "sample": {"id": "sample_101", "category": "cable"},
        "inspection": {
            "decision": "anomalous",
            "anomaly_score": 0.88,
            "reliability": 0.92,
            "manual_review_recommended": False,
        },
        "defects": [
            {
                "id": 1,
                "location": {"label": "center", "centroid_normalized": [0.5, 0.5]},
                "shape": {"label": "compact", "aspect_ratio": 1.1},
                "size": {"surface_area_mm2": 45.2, "major_length_mm": 7.5, "minor_length_mm": 6.8},
                "geometry": {"structure": "localized depression", "max_depression_mm": 1.45, "mean_depression_mm": 0.62},
                "volume": {"missing_material_mm3": 28.3, "excess_material_mm3": 0.0, "measurement_confidence": 0.95},
                "evidence": {
                    "rgb": {"value": 0.40, "level": "Moderate"},
                    "xyz": {"value": 0.85, "level": "High"},
                    "topology_disagreement": {"value": 0.79, "level": "High"},
                },
                "normal_reference": {"prototype_id": 501, "training_sample_id": "cable_norm_05"},
                "prototype_trace": {"shared_top5": 0, "js_divergence": 0.79, "gate": 0.85},
            }
        ]
    }

    context, fields = build_assistant_context(raw_result)

    assert context["sample"]["id"] == "sample_101"
    assert context["inspection"]["decision"] == "anomalous"
    assert len(context["defects"]) == 1
    d0 = context["defects"][0]
    assert d0["geometry"]["max_depression_mm"] == 1.45
    assert d0["volume"]["missing_material_mm3"] == 28.3
    assert d0["normal_reference"]["prototype_id"] == 501
    assert "geometry.max_depression_mm" in fields
    assert "volume.missing_material_mm3" in fields

    numbers = extract_context_numbers(context)
    assert 1.45 in numbers
    assert 28.3 in numbers
    assert 501.0 in numbers


def test_context_builder_normal_sample():
    """Verify context builder on normal sample with zero defects."""
    raw_result = {
        "sample": {"id": "normal_002", "category": "foam"},
        "inspection": {
            "decision": "normal",
            "anomaly_score": 0.08,
            "reliability": 0.99,
        },
        "defects": []
    }

    context, fields = build_assistant_context(raw_result)
    assert context["inspection"]["decision"] == "normal"
    assert len(context["defects"]) == 0
    assert "inspection.decision" in fields


def test_context_builder_missing_optional_fields():
    """Verify context builder handles sparse or missing fields without failing."""
    sparse_result = {
        "inspection": {
            "decision": "anomalous",
            "anomaly_score": 0.75,
        },
        "defects": [
            {
                "id": 1,
                "location": {"label": "lower-left"},
                # Missing geometry, volume, prototype_trace, etc.
            }
        ]
    }

    context, fields = build_assistant_context(sparse_result)
    assert context["inspection"]["decision"] == "anomalous"
    assert context["defects"][0]["location"]["label"] == "lower-left"
    assert "geometry" not in context["defects"][0]
    assert "volume" not in context["defects"][0]


def test_context_builder_multiple_defects():
    """Verify context builder cleanly preserves multiple distinct defect entries."""
    multi_defect = {
        "sample": {"id": "multi_defect_sample"},
        "inspection": {"decision": "anomalous", "anomaly_score": 0.95},
        "defects": [
            {"id": 1, "geometry": {"structure": "localized depression", "max_depression_mm": 1.2}},
            {"id": 2, "geometry": {"structure": "localized protrusion", "max_protrusion_mm": 2.1}},
        ]
    }

    context, fields = build_assistant_context(multi_defect)
    assert len(context["defects"]) == 2
    assert context["defects"][0]["geometry"]["max_depression_mm"] == 1.2
    assert context["defects"][1]["geometry"]["max_protrusion_mm"] == 2.1


def test_context_builder_manual_review_case():
    """Verify manual review recommendation is captured in context."""
    review_sample = {
        "inspection": {
            "decision": "anomalous",
            "anomaly_score": 0.65,
            "reliability": 0.42,
            "manual_review_recommended": True,
            "review_reasons": ["Low depth map quality", "Marginal threshold crossing"],
        }
    }

    context, fields = build_assistant_context(review_sample)
    assert context["inspection"]["manual_review_recommended"] is True
    assert "inspection.manual_review_recommended" in fields
    assert len(context["inspection"]["review_reasons"]) == 2
