"""PNTC Industrial Anomaly Detection AI Assistant Subsystem.

Provides natural-language explanation and multi-turn conversational Q&A
grounded strictly in verified PNTC structured inspection results.
"""

from .schema import (
    ResponseMode,
    UIActionType,
    UIAction,
    GroundingInfo,
    AssistantResponse,
    ChatMessage,
    ChatSession,
)
from .base import LLMProvider
from .gemini_provider import GeminiProvider
from .grok_provider import GrokProvider
from .provider_factory import (
    create_provider,
    get_default_provider,
    get_provider_status,
)
from .context_builder import (
    build_assistant_context,
    extract_context_numbers,
)
from .system_prompt import (
    PNTC_ASSISTANT_PROMPT_VERSION,
    build_system_prompt,
)
from .guardrails import (
    detect_prompt_injection,
    verify_numeric_grounding,
    pre_process_guardrails,
    post_process_guardrails,
)
from .history import (
    SessionManager,
    global_session_manager,
)
from .chat import (
    PNTCAssistant,
)

__all__ = [
    "ResponseMode",
    "UIActionType",
    "UIAction",
    "GroundingInfo",
    "AssistantResponse",
    "ChatMessage",
    "ChatSession",
    "LLMProvider",
    "GeminiProvider",
    "GrokProvider",
    "create_provider",
    "get_default_provider",
    "get_provider_status",
    "build_assistant_context",
    "extract_context_numbers",
    "PNTC_ASSISTANT_PROMPT_VERSION",
    "build_system_prompt",
    "detect_prompt_injection",
    "verify_numeric_grounding",
    "pre_process_guardrails",
    "post_process_guardrails",
    "SessionManager",
    "global_session_manager",
    "PNTCAssistant",
]
