"""xAI Grok API provider implementation for the PNTC AI Assistant."""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
import httpx

from .base import LLMProvider

logger = logging.getLogger("xmvad.assistant.grok")


class GrokProvider(LLMProvider):
    """Provider for xAI Grok models (e.g. grok-2-latest, grok-beta) via OpenAI-compatible REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: str = "https://api.x.ai/v1",
        timeout_seconds: float = 30.0
    ):
        self._api_key = api_key or os.environ.get("XAI_API_KEY", "").strip()
        self._default_model = model_name or default_model or os.environ.get("PNTC_LLM_MODEL", "").strip() or "grok-2-latest"
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "grok"

    @property
    def default_model(self) -> str:
        return self._default_model

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        context: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.1,
        response_mode: str = "TECHNICAL"
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.is_configured():
            raise RuntimeError(
                "GrokProvider is not configured: XAI_API_KEY is missing or empty."
            )

        target_model = model or self.default_model
        start_time = time.time()

        full_system_text = (
            f"{system_prompt}\n\n"
            f"RESPONSE STYLE MODE: {response_mode}\n\n"
            f"FACTUAL INSPECTION CONTEXT (JSON):\n"
            f"```json\n{json.dumps(context, indent=2)}\n```\n"
        )

        api_messages = [{"role": "system", "content": full_system_text}]
        for m in messages:
            role = m.role if hasattr(m, "role") else m["role"]
            content = m.content if hasattr(m, "content") else m["content"]
            api_messages.append({"role": role, "content": content})

        payload = {
            "model": target_model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": 1024
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }

        url = f"{self.base_url}/chat/completions"

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            latency_ms = (time.time() - start_time) * 1000.0

            choices = data.get("choices", [])
            text_out = ""
            if choices:
                text_out = choices[0].get("message", {}).get("content", "")

            usage = data.get("usage")
            token_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "candidate_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            } if usage else None

            return text_out, {
                "provider": "grok",
                "model": target_model,
                "latency_ms": latency_ms,
                "token_usage": token_usage,
                "via": "xai-rest"
            }
        except httpx.TimeoutException as e:
            safe_err = f"xAI Grok API request timed out ({self.timeout_seconds}s)"
            logger.error(safe_err)
            raise RuntimeError(safe_err) from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            safe_err = f"xAI Grok API error (HTTP {status}): {e.response.text[:200]}"
            logger.error(safe_err)
            raise RuntimeError(safe_err) from e
        except Exception as e:
            logger.error(f"xAI Grok REST error: {type(e).__name__}")
            raise
