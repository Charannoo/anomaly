"""Core conversation engine and explanation generator for PNTC AI Assistant.

Implements Phase A10 (Default Explanation), Phase A11 (Chat Q&A),
Phase A12 (UI Action generation), Phase A14 (Provider Failover),
Phase A17 (Context Size Control), Phase A21 (Output Schema),
Phase A22 (Explanation Trace), and Phase A31 (Caching).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import LLMProvider
from .context_builder import build_assistant_context, extract_context_numbers
from .guardrails import (
    post_process_guardrails,
    pre_process_guardrails,
    sanitize_user_message,
)
from .history import global_session_manager
from .provider_factory import get_default_provider
from .schema import (
    AssistantResponse,
    ChatMessage,
    GroundingInfo,
    ResponseMode,
    UIAction,
    UIActionType,
)
from .system_prompt import PNTC_ASSISTANT_PROMPT_VERSION, build_system_prompt

logger = logging.getLogger("pntc.assistant")

# In-memory explanation cache: hash -> AssistantResponse
_EXPLANATION_CACHE: Dict[str, AssistantResponse] = {}

# Path to verified project context for Viva mode
_PROJECT_CONTEXT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "docs" / "PNTC_ASSISTANT_PROJECT_CONTEXT.json"


def _load_project_context() -> Dict[str, Any]:
    """Load verified project context facts for Viva mode."""
    if _PROJECT_CONTEXT_PATH.exists():
        try:
            with open(_PROJECT_CONTEXT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read project context file: %s", e)
    return {}


def _detect_ui_action(user_message: str, context: dict, defect_id: Optional[int]) -> Optional[UIAction]:
    """Detect if the user's inquiry requests or warrants a specific frontend UI action."""
    msg_lower = user_message.lower()
    defects = context.get("defects", [])

    # Target defect
    target_defect = None
    if defect_id is not None:
        for d in defects:
            if d.get("id") == defect_id:
                target_defect = d
                break
    if target_defect is None and defects:
        target_defect = defects[0]
        defect_id = target_defect.get("id")

    # 1. Normal twin / reference
    if any(k in msg_lower for k in ["normal example", "normal twin", "normal reference", "nearest normal", "closest normal"]):
        proto_id = None
        if target_defect:
            proto_id = target_defect.get("normal_reference", {}).get("prototype_id")
        return UIAction(
            type=UIActionType.SHOW_NORMAL_TWIN.value,
            defect_id=defect_id,
            prototype_id=proto_id,
        )

    # 2. 3D view / depth scan
    if any(k in msg_lower for k in ["3d view", "show 3d", "3d scan", "surface profile", "surface mesh"]):
        return UIAction(
            type=UIActionType.SHOW_3D_VIEW.value,
            defect_id=defect_id,
        )

    # 3. Prototype trace / retrieval evidence
    if any(k in msg_lower for k in ["prototype trace", "prototype evidence", "retrieval trace", "top-5", "top 5"]):
        return UIAction(
            type=UIActionType.SHOW_PROTOTYPE_TRACE.value,
            defect_id=defect_id,
        )

    # 4. Heatmap
    if any(k in msg_lower for k in ["heatmap", "anomaly map", "overlay"]):
        return UIAction(
            type=UIActionType.SHOW_HEATMAP.value,
            defect_id=defect_id,
        )

    # 5. Defect comparison
    if "compare" in msg_lower and any(k in msg_lower for k in ["defect", "defects"]):
        return UIAction(
            type=UIActionType.COMPARE_DEFECTS.value,
            defect_id=defect_id,
        )

    # 6. Highlight defect
    if any(k in msg_lower for k in ["highlight", "where is", "locate"]):
        return UIAction(
            type=UIActionType.HIGHLIGHT_DEFECT.value,
            defect_id=defect_id,
        )

    return None


def _calculate_cache_key(
    context: dict,
    provider_name: str,
    model_name: str,
    response_mode: str,
    prompt_version: str = PNTC_ASSISTANT_PROMPT_VERSION,
) -> str:
    """Generate SHA-256 hash for deterministic caching."""
    raw_str = json.dumps(context, sort_keys=True) + f"::{provider_name}::{model_name}::{response_mode}::{prompt_version}"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()


class PNTCAssistant:
    """High-level conversational inspection assistant."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or get_default_provider()
        self.project_context = _load_project_context()

    def generate_inspection_summary(
        self,
        inspection_result: Any,
        response_mode: ResponseMode = ResponseMode.TECHNICAL,
        force_refresh: bool = False,
    ) -> AssistantResponse:
        """Generate or retrieve cached default explanation for a verified inspection result."""
        context, fields_used = build_assistant_context(inspection_result)
        sample_id = context.get("sample", {}).get("id", "unknown_sample")
        defects = context.get("defects", [])
        defect_id = defects[0].get("id") if defects else None

        # Check cache
        cache_key = _calculate_cache_key(
            context=context,
            provider_name=self.provider.provider_name,
            model_name=self.provider.model_name,
            response_mode=response_mode.value,
        )

        if not force_refresh and cache_key in _EXPLANATION_CACHE:
            cached_resp = _EXPLANATION_CACHE[cache_key]
            # Return copy with cached=True
            return AssistantResponse(
                message=cached_resp.message,
                provider=cached_resp.provider,
                model=cached_resp.model,
                sample_id=cached_resp.sample_id,
                defect_id=cached_resp.defect_id,
                grounding=cached_resp.grounding,
                ui_action=cached_resp.ui_action,
                warnings=cached_resp.warnings,
                response_mode=cached_resp.response_mode,
                latency_ms=0.0,
                token_usage=cached_resp.token_usage,
                cached=True,
            )

        # Handle normal sample deterministically
        is_normal = context.get("inspection", {}).get("decision") == "normal" or not defects
        if is_normal:
            resp = AssistantResponse(
                message=(
                    "The inspected object was classified as normal by PNTC. "
                    "No defect region exceeded the configured detection criterion."
                ),
                provider=self.provider.provider_name,
                model=self.provider.model_name,
                sample_id=sample_id,
                defect_id=None,
                grounding=GroundingInfo(
                    source="PNTC_INSPECTION_JSON",
                    fields_used=["inspection.decision", "inspection.anomaly_score"],
                    passed_guardrails=True,
                ),
                response_mode=response_mode.value,
                cached=False,
            )
            _EXPLANATION_CACHE[cache_key] = resp
            return resp

        # Build prompt for summary
        system_prompt = build_system_prompt(response_mode, self.project_context)
        summary_prompt = (
            "Provide a comprehensive, grounded explanation summary for this inspected sample based solely on the structured context. "
            "Describe the detected defect's location, shape, 3D geometry measurements (depression/protrusion and volume if available), "
            "and explain the cross-modal evidence (RGB, XYZ, and topology disagreement)."
        )

        messages = [ChatMessage(role="user", content=summary_prompt)]
        t0 = time.time()
        raw_res = self.provider.generate(system_prompt, messages, context)
        raw_text = raw_res[0] if isinstance(raw_res, tuple) else raw_res
        latency_ms = (time.time() - t0) * 1000.0

        # Post-process guardrails
        clean_text, warnings = post_process_guardrails(raw_text, context, summary_prompt)

        allowed_numbers = extract_context_numbers(context)
        grounding = GroundingInfo(
            source="PNTC_INSPECTION_JSON",
            fields_used=fields_used,
            verified_numbers=allowed_numbers,
            passed_guardrails=len(warnings) == 0,
        )

        resp = AssistantResponse(
            message=clean_text,
            provider=self.provider.provider_name,
            model=self.provider.model_name,
            sample_id=sample_id,
            defect_id=defect_id,
            grounding=grounding,
            warnings=warnings,
            response_mode=response_mode.value,
            latency_ms=latency_ms,
            cached=False,
        )

        _EXPLANATION_CACHE[cache_key] = resp
        return resp

    def chat(
        self,
        sample_id: str,
        message: str,
        conversation_id: Optional[str] = None,
        inspection_result: Optional[Any] = None,
        defect_id: Optional[int] = None,
        response_mode: ResponseMode = ResponseMode.TECHNICAL,
    ) -> AssistantResponse:
        """Handle multi-turn conversational questions about an inspection result."""
        # 1. Prepare structured context
        context, fields_used = build_assistant_context(inspection_result)
        available_defect_ids = [d.get("id") for d in context.get("defects", []) if "id" in d]

        # 2. Manage session
        session = global_session_manager.get_or_create_session(
            conversation_id=conversation_id,
            sample_id=sample_id,
            defect_id=defect_id,
            response_mode=response_mode.value,
        )

        # Disambiguate defect reference
        resolved_defect_id = global_session_manager.resolve_defect_reference(
            session.conversation_id,
            message,
            available_defect_ids,
        )

        clean_user_message = sanitize_user_message(message)

        # 3. Detect UI Action
        ui_action = _detect_ui_action(clean_user_message, context, resolved_defect_id)

        # 4. Pre-process guardrails (injection, unsupported causal claim, unvalidated defect class)
        pre_check = pre_process_guardrails(clean_user_message, context)
        if pre_check.override_response:
            # Store in session and return deterministic safe answer
            global_session_manager.add_message(session.conversation_id, "user", clean_user_message)
            global_session_manager.add_message(
                session.conversation_id,
                "assistant",
                pre_check.override_response,
                ui_action=ui_action,
            )

            return AssistantResponse(
                message=pre_check.override_response,
                provider=self.provider.provider_name,
                model=self.provider.model_name,
                sample_id=sample_id,
                defect_id=resolved_defect_id,
                grounding=GroundingInfo(
                    source="PNTC_INSPECTION_JSON",
                    fields_used=fields_used,
                    passed_guardrails=True,
                ),
                ui_action=ui_action,
                warnings=pre_check.warnings,
                response_mode=response_mode.value,
                latency_ms=0.0,
            )

        # 5. Build prompt and message list
        system_prompt = build_system_prompt(response_mode, self.project_context)

        # Append message history (up to last 10 messages for context size control)
        recent_history = session.messages[-10:] if len(session.messages) > 10 else session.messages
        chat_messages = list(recent_history)
        chat_messages.append(ChatMessage(role="user", content=clean_user_message))

        # 6. Call provider
        t0 = time.time()
        raw_res = self.provider.generate(system_prompt, chat_messages, context)
        raw_text = raw_res[0] if isinstance(raw_res, tuple) else raw_res
        latency_ms = (time.time() - t0) * 1000.0

        # 7. Post-process guardrails
        clean_text, warnings = post_process_guardrails(raw_text, context, clean_user_message)
        warnings.extend(pre_check.warnings)

        # 8. Record in session
        global_session_manager.add_message(session.conversation_id, "user", clean_user_message)
        global_session_manager.add_message(session.conversation_id, "assistant", clean_text, ui_action=ui_action)

        allowed_numbers = extract_context_numbers(context)
        grounding = GroundingInfo(
            source="PNTC_INSPECTION_JSON",
            fields_used=fields_used,
            verified_numbers=allowed_numbers,
            passed_guardrails=len(warnings) == 0,
        )

        return AssistantResponse(
            message=clean_text,
            provider=self.provider.provider_name,
            model=self.provider.model_name,
            sample_id=sample_id,
            defect_id=resolved_defect_id,
            grounding=grounding,
            ui_action=ui_action,
            warnings=warnings,
            response_mode=response_mode.value,
            latency_ms=latency_ms,
        )
