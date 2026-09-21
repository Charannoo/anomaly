"""Factory and failover coordinator for LLM providers."""

import os
import logging
from typing import Dict, Optional, Any, Tuple, List, Union

from .base import LLMProvider
from .gemini_provider import GeminiProvider
from .grok_provider import GrokProvider

logger = logging.getLogger("xmvad.assistant.factory")


def create_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """Instantiate configured LLMProvider.
    
    Args:
        provider_name: 'gemini' or 'grok'. Defaults to PNTC_LLM_PROVIDER env var, or 'gemini'.
        
    Returns:
        LLMProvider instance.
    """
    name = (provider_name or os.environ.get("PNTC_LLM_PROVIDER", "gemini")).strip().lower()

    if name == "gemini":
        return GeminiProvider()
    elif name == "grok":
        return GrokProvider()
    else:
        raise ValueError(
            f"Unsupported LLM provider '{name}'. Must be 'gemini' or 'grok'."
        )


def get_default_provider() -> LLMProvider:
    """Return the default configured provider instance."""
    return create_provider()


def get_provider_status(provider: Optional[LLMProvider] = None) -> Dict[str, Any]:
    """Return configured status of a provider or default provider without exposing API keys."""
    prov = provider or get_default_provider()
    return {
        "provider": prov.provider_name,
        "configured": prov.is_configured(),
        "model": prov.default_model,
        "available_providers": ["gemini", "grok"],
    }


def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """Return configured status of all providers without exposing API keys."""
    gemini = GeminiProvider()
    grok = GrokProvider()
    current_default = os.environ.get("PNTC_LLM_PROVIDER", "gemini").strip().lower()

    return {
        "gemini": {
            "name": "Google Gemini",
            "configured": gemini.is_configured(),
            "default_model": gemini.default_model,
            "is_active": current_default == "gemini"
        },
        "grok": {
            "name": "xAI Grok",
            "configured": grok.is_configured(),
            "default_model": grok.default_model,
            "is_active": current_default == "grok"
        }
    }


def generate_with_failover(
    system_prompt: str,
    messages: List[Any],
    context: Dict[str, Any],
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
    response_mode: str = "TECHNICAL",
    allow_failover: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """Execute generation on primary provider, with optional transparent failover to secondary.
    
    Logs fallback without leaking credentials.
    """
    primary_name = (provider_name or os.environ.get("PNTC_LLM_PROVIDER", "gemini")).strip().lower()
    primary_provider = create_provider(primary_name)

    try:
        res = primary_provider.generate(
            system_prompt=system_prompt,
            messages=messages,
            context=context,
            model=model,
            temperature=temperature,
            response_mode=response_mode
        )
        if isinstance(res, tuple):
            return res
        return res, {"provider": primary_name, "model": primary_provider.default_model}
    except Exception as primary_err:
        logger.warning(
            f"Primary provider '{primary_name}' failed: {type(primary_err).__name__}: {str(primary_err)[:120]}"
        )

        if not allow_failover:
            raise

        secondary_name = "grok" if primary_name == "gemini" else "gemini"
        secondary_provider = create_provider(secondary_name)

        if not secondary_provider.is_configured():
            logger.info(f"Secondary provider '{secondary_name}' is not configured. Raising primary error.")
            raise primary_err

        logger.info(f"Executing failover from '{primary_name}' to secondary provider '{secondary_name}'.")
        res = secondary_provider.generate(
            system_prompt=system_prompt,
            messages=messages,
            context=context,
            temperature=temperature,
            response_mode=response_mode
        )
        if isinstance(res, tuple):
            text_out, meta = res
        else:
            text_out, meta = res, {"provider": secondary_name, "model": secondary_provider.default_model}
        meta["failed_over_from"] = primary_name
        return text_out, meta
