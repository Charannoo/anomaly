"""Tests for Multi-Turn Chat, Session Memory, UI Actions, and Caching."""

import pytest
from unittest.mock import MagicMock

from xmvad.assistant.chat import PNTCAssistant
from xmvad.assistant.base import LLMProvider
from xmvad.assistant.schema import ResponseMode, UIActionType


class DummyProvider(LLMProvider):
    def __init__(self):
        self._name = "gemini"
        self._model = "mock-gemini"

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def default_model(self) -> str:
        return self._model

    @property
    def model_name(self) -> str:
        return self._model

    def is_configured(self) -> bool:
        return True

    def generate(self, system_prompt: str, messages: list, context: dict) -> str:
        last_msg = messages[-1].content if messages else ""
        if "normal example" in last_msg.lower():
            return "The nearest paired normal reference is prototype #1842."
        if "how deep" in last_msg.lower():
            return "The maximum depression is 1.82 mm."
        if "how much material" in last_msg.lower():
            return "Missing material is 84.6 mm³."
        return "An anomalous region was detected with maximum depression of 1.82 mm."


@pytest.fixture
def sample_data():
    return {
        "sample": {"id": "sample_001", "category": "potato"},
        "inspection": {
            "decision": "anomalous",
            "anomaly_score": 0.91,
            "inspection_status": "DEFECT_DETECTED",
        },
        "defects": [
            {
                "id": 1,
                "location": {"label": "upper-right"},
                "shape": {"label": "elongated irregular"},
                "geometry": {"structure": "localized depression", "max_depression_mm": 1.82},
                "volume": {"missing_material_mm3": 84.6},
                "normal_reference": {"prototype_id": 1842},
            }
        ]
    }


def test_default_explanation_and_caching(sample_data):
    """Verify default summary generation and Phase A31 deterministic caching."""
    prov = DummyProvider()
    asst = PNTCAssistant(provider=prov)

    # First call - generated
    resp1 = asst.generate_inspection_summary(sample_data, response_mode=ResponseMode.TECHNICAL)
    assert resp1.cached is False
    assert "1.82 mm" in resp1.message

    # Second call - cached
    resp2 = asst.generate_inspection_summary(sample_data, response_mode=ResponseMode.TECHNICAL)
    assert resp2.cached is True
    assert resp2.message == resp1.message


def test_ui_action_normal_twin(sample_data):
    """Phase A12: 'Show me the nearest normal example.' -> SHOW_NORMAL_TWIN."""
    prov = DummyProvider()
    asst = PNTCAssistant(provider=prov)

    resp = asst.chat(
        sample_id="sample_001",
        message="Show me the nearest normal example.",
        inspection_result=sample_data,
    )
    assert resp.ui_action is not None
    assert resp.ui_action.type == UIActionType.SHOW_NORMAL_TWIN.value
    assert resp.ui_action.prototype_id == 1842


def test_ui_action_3d_view(sample_data):
    """Phase A12: 'Show 3D view' -> SHOW_3D_VIEW."""
    prov = DummyProvider()
    asst = PNTCAssistant(provider=prov)

    resp = asst.chat(
        sample_id="sample_001",
        message="Please show 3D view of the surface.",
        inspection_result=sample_data,
    )
    assert resp.ui_action is not None
    assert resp.ui_action.type == UIActionType.SHOW_3D_VIEW.value


def test_ui_action_prototype_trace(sample_data):
    """Phase A12: 'Show prototype evidence' -> SHOW_PROTOTYPE_TRACE."""
    prov = DummyProvider()
    asst = PNTCAssistant(provider=prov)

    resp = asst.chat(
        sample_id="sample_001",
        message="Show prototype evidence and retrieval trace.",
        inspection_result=sample_data,
    )
    assert resp.ui_action is not None
    assert resp.ui_action.type == UIActionType.SHOW_PROTOTYPE_TRACE.value


def test_multi_turn_conversation_context(sample_data):
    """Phase A16: Multi-turn session context tracking."""
    prov = DummyProvider()
    asst = PNTCAssistant(provider=prov)
    conv_id = "test_conv_123"

    # Turn 1
    resp1 = asst.chat(
        sample_id="sample_001",
        message="How deep is defect 1?",
        conversation_id=conv_id,
        inspection_result=sample_data,
    )
    assert "1.82 mm" in resp1.message

    # Turn 2: pronoun reference 'it'
    resp2 = asst.chat(
        sample_id="sample_001",
        message="And how much material is missing from it?",
        conversation_id=conv_id,
        inspection_result=sample_data,
    )
    assert "84.6 mm³" in resp2.message
    assert resp2.defect_id == 1
