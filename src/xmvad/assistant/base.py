"""Abstract base class and contract for PNTC LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple


class LLMProvider(ABC):
    """Abstract interface for all conversational LLM providers (Gemini, Grok, etc.)."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier name of the provider ('gemini', 'grok', etc.)."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model identifier for this provider."""
        pass

    @property
    def model_name(self) -> str:
        """Model identifier property."""
        return self.default_model

    @abstractmethod
    def is_configured(self) -> bool:
        """Check whether the necessary API key or credentials are present."""
        pass

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        context: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.1,
        response_mode: str = "TECHNICAL"
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response text given system prompt, conversation history, and factual context.
        
        Args:
            system_prompt: Versioned, strict system prompt.
            messages: List of conversation turn dicts [{"role": "user"|"assistant", "content": "..."}].
            context: Pure-factual structured JSON context dictionary.
            model: Optional model override.
            temperature: Sampling temperature (default 0.1 for high factual consistency).
            response_mode: 'SIMPLE', 'TECHNICAL', or 'VIVA'.
            
        Returns:
            Tuple of (response_text, metadata_dict) where metadata_dict contains latency_ms,
            model_used, token_usage, etc.
        """
        pass
