"""Google Gemini API provider implementation for the PNTC AI Assistant."""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
import httpx

from .base import LLMProvider

logger = logging.getLogger("xmvad.assistant.gemini")


class GeminiProvider(LLMProvider):
    """Provider for Google Gemini API models (e.g. gemini-2.5-flash, gemini-1.5-flash)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: float = 30.0
    ):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        self._default_model = model_name or default_model or os.environ.get("PNTC_LLM_MODEL", "").strip() or "gemini-2.5-flash"
        self.timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "gemini"

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
                "GeminiProvider is not configured: GEMINI_API_KEY is missing or empty."
            )

        target_model = model or self.default_model
        start_time = time.time()

        # Try official google-genai SDK first
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self._api_key)
            
            # Formulate structured prompt with context
            full_system_instruction = (
                f"{system_prompt}\n\n"
                f"RESPONSE STYLE MODE: {response_mode}\n\n"
                f"FACTUAL INSPECTION CONTEXT (JSON):\n"
                f"```json\n{json.dumps(context, indent=2)}\n```\n"
            )

            contents = []
            for msg in messages:
                role_val = msg.role if hasattr(msg, "role") else msg["role"]
                content_val = msg.content if hasattr(msg, "content") else msg["content"]
                role = "user" if role_val == "user" else "model"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=content_val)]
                ))

            config = types.GenerateContentConfig(
                system_instruction=full_system_instruction,
                temperature=temperature,
                max_output_tokens=1024
            )

            resp = client.models.generate_content(
                model=target_model,
                contents=contents,
                config=config
            )

            latency_ms = (time.time() - start_time) * 1000.0
            text_out = resp.text if hasattr(resp, "text") and resp.text else ""
            
            usage_dict = None
            if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                usage_dict = {
                    "prompt_tokens": getattr(resp.usage_metadata, "prompt_token_count", 0),
                    "candidate_tokens": getattr(resp.usage_metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(resp.usage_metadata, "total_token_count", 0)
                }

            return text_out, {
                "provider": "gemini",
                "model": target_model,
                "latency_ms": latency_ms,
                "token_usage": usage_dict,
                "via": "google-genai"
            }

        except ImportError:
            # Fallback to direct HTTP REST API via httpx
            return self._generate_via_rest(
                system_prompt=system_prompt,
                messages=messages,
                context=context,
                model=target_model,
                temperature=temperature,
                response_mode=response_mode,
                start_time=start_time
            )
        except Exception as e:
            logger.warning(f"google-genai SDK call failed: {type(e).__name__}. Falling back to REST.")
            return self._generate_via_rest(
                system_prompt=system_prompt,
                messages=messages,
                context=context,
                model=target_model,
                temperature=temperature,
                response_mode=response_mode,
                start_time=start_time
            )

    def _generate_via_rest(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        context: Dict[str, Any],
        model: str,
        temperature: float,
        response_mode: str,
        start_time: float
    ) -> Tuple[str, Dict[str, Any]]:
        """Direct fallback REST call to Google Generative Language API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
        full_system_text = (
            f"{system_prompt}\n\n"
            f"RESPONSE STYLE MODE: {response_mode}\n\n"
            f"FACTUAL INSPECTION CONTEXT (JSON):\n"
            f"```json\n{json.dumps(context, indent=2)}\n```\n"
        )

        contents_payload = []
        for msg in messages:
            role_val = msg.role if hasattr(msg, "role") else msg["role"]
            content_val = msg.content if hasattr(msg, "content") else msg["content"]
            role = "user" if role_val == "user" else "model"
            contents_payload.append({
                "role": role,
                "parts": [{"text": content_val}]
            })

        payload = {
            "systemInstruction": {
                "parts": [{"text": full_system_text}]
            },
            "contents": contents_payload,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 1024
            }
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            latency_ms = (time.time() - start_time) * 1000.0

            candidates = data.get("candidates", [])
            text_out = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text_out = parts[0].get("text", "")

            usage_metadata = data.get("usageMetadata", {})
            token_usage = {
                "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                "candidate_tokens": usage_metadata.get("candidatesTokenCount", 0),
                "total_tokens": usage_metadata.get("totalTokenCount", 0)
            } if usage_metadata else None

            return text_out, {
                "provider": "gemini",
                "model": model,
                "latency_ms": latency_ms,
                "token_usage": token_usage,
                "via": "httpx-rest"
            }
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            safe_err = f"Gemini API error (HTTP {status}): {e.response.text[:200]}"
            logger.error(safe_err)
            raise RuntimeError(safe_err) from e
        except Exception as e:
            logger.error(f"Gemini REST error: {type(e).__name__}")
            raise
