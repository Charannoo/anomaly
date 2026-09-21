"""Tests for LLM Provider Abstraction, Gemini & Grok mocks, Failover, and Security."""

import os
import pytest
from unittest.mock import MagicMock, patch

from xmvad.assistant.base import LLMProvider
from xmvad.assistant.gemini_provider import GeminiProvider
from xmvad.assistant.grok_provider import GrokProvider
from xmvad.assistant.provider_factory import create_provider, get_provider_status
from xmvad.assistant.schema import ChatMessage


class MockTestProvider(LLMProvider):
    def __init__(self, name: str = "mock", model: str = "mock-model"):
        self._name = name
        self._model = model

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def default_model(self) -> str:
        return self._model

    def is_configured(self) -> bool:
        return True

    def generate(self, system_prompt: str, messages: list, context: dict) -> str:
        return "Mock generated explanation."


def test_provider_abstraction():
    """Verify common interface conformance."""
    prov = MockTestProvider()
    assert isinstance(prov, LLMProvider)
    assert prov.provider_name == "mock"
    assert prov.default_model == "mock-model"
    resp = prov.generate("sys", [ChatMessage(role="user", content="hi")], {})
    assert resp == "Mock generated explanation."


def test_gemini_provider_mock():
    """Verify Gemini provider generation via mock."""
    prov = GeminiProvider(api_key="fake_gemini_key", model_name="gemini-2.5-flash")
    assert prov.provider_name == "gemini"

    # Mock _generate_via_rest to avoid real API network call
    with patch.object(prov, "_generate_via_rest", return_value=("Gemini mock response", {})):
        resp = prov.generate("system prompt", [ChatMessage(role="user", content="Explain")], {"sample": {"id": "s1"}})
        text = resp[0] if isinstance(resp, tuple) else resp
        assert text == "Gemini mock response"


def test_grok_provider_mock():
    """Verify Grok provider generation via mock."""
    prov = GrokProvider(api_key="fake_grok_key", model_name="grok-2-latest")
    assert prov.provider_name == "grok"

    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {
        "choices": [{"message": {"content": "Grok mock response"}}]
    }

    with patch("httpx.Client.post", return_value=mock_http_response):
        resp = prov.generate("system prompt", [ChatMessage(role="user", content="Explain")], {})
        text = resp[0] if isinstance(resp, tuple) else resp
        assert text == "Grok mock response"


def test_provider_timeout_handling():
    """Verify provider timeout gracefully raises or handles exception."""
    import httpx
    prov = GrokProvider(api_key="fake_grok_key", timeout_seconds=0.1)

    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Connection timed out")):
        with pytest.raises(RuntimeError) as exc_info:
            prov.generate("sys", [ChatMessage(role="user", content="q")], {})
        assert "timed out" in str(exc_info.value).lower()


def test_invalid_api_key_handling():
    """Verify provider returns clean error without leaking key when authentication fails."""
    import httpx
    prov = GrokProvider(api_key="SECRET_KEY_12345")

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized: Invalid API Key"
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("401 Unauthorized", request=MagicMock(), response=mock_resp)

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(RuntimeError) as exc_info:
            prov.generate("sys", [ChatMessage(role="user", content="q")], {})
        # Error message must NOT leak the secret key
        assert "SECRET_KEY_12345" not in str(exc_info.value)
        assert "401" in str(exc_info.value)


def test_no_api_key_exposure_in_status():
    """Verify get_provider_status never exposes the API key."""
    prov = GeminiProvider(api_key="SECRET_GEMINI_KEY_XYZ")
    status = get_provider_status(prov)
    assert status["provider"] == "gemini"
    assert status["configured"] is True
    # Assure secret key does not appear anywhere in the serialized dict
    status_str = str(status)
    assert "SECRET_GEMINI_KEY_XYZ" not in status_str
    assert "key" not in status or status.get("key") is None


def test_provider_factory_switch(monkeypatch):
    """Verify provider factory correctly switches between gemini and grok."""
    monkeypatch.setenv("PNTC_LLM_PROVIDER", "grok")
    monkeypatch.setenv("XAI_API_KEY", "mock_xai_key")
    p = create_provider("grok")
    assert p.provider_name == "grok"

    p2 = create_provider("gemini")
    assert p2.provider_name == "gemini"
