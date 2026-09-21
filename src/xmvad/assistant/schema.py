"""Data models and schemas for the PNTC AI Assistant subsystem."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import time


class ResponseMode(str, Enum):
    """Supported response style modes."""
    SIMPLE = "SIMPLE"          # Operator / non-technical summary
    TECHNICAL = "TECHNICAL"    # Full engineering metrics & physical values
    VIVA = "VIVA"              # Educational / research viva explanations


class UIActionType(str, Enum):
    """Actions emitted to the frontend dashboard."""
    SHOW_NORMAL_TWIN = "SHOW_NORMAL_TWIN"
    SHOW_3D_VIEW = "SHOW_3D_VIEW"
    SHOW_PROTOTYPE_TRACE = "SHOW_PROTOTYPE_TRACE"
    SHOW_HEATMAP = "SHOW_HEATMAP"
    HIGHLIGHT_DEFECT = "HIGHLIGHT_DEFECT"
    COMPARE_DEFECTS = "COMPARE_DEFECTS"


@dataclass
class UIAction:
    """Structured UI interaction command emitted alongside an assistant response."""
    type: str
    defect_id: Optional[int] = None
    prototype_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type}
        if self.defect_id is not None:
            d["defect_id"] = self.defect_id
        if self.prototype_id is not None:
            d["prototype_id"] = self.prototype_id
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class GroundingInfo:
    """Audit trace of facts and measurements utilized by the model."""
    source: str = "PNTC_INSPECTION_JSON"
    fields_used: List[str] = field(default_factory=list)
    verified_numbers: List[float] = field(default_factory=list)
    unverified_numbers: List[float] = field(default_factory=list)
    passed_guardrails: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "fields_used": self.fields_used,
            "verified_numbers": self.verified_numbers,
            "unverified_numbers": self.unverified_numbers,
            "passed_guardrails": self.passed_guardrails
        }


@dataclass
class AssistantResponse:
    """Standardized output schema for PNTC AI Assistant."""
    message: str
    provider: str
    model: str
    sample_id: str
    defect_id: Optional[int] = None
    grounding: GroundingInfo = field(default_factory=GroundingInfo)
    ui_action: Optional[UIAction] = None
    warnings: List[str] = field(default_factory=list)
    response_mode: str = "TECHNICAL"
    latency_ms: float = 0.0
    token_usage: Optional[Dict[str, int]] = None
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "provider": self.provider,
            "model": self.model,
            "sample_id": self.sample_id,
            "defect_id": self.defect_id,
            "grounding": self.grounding.to_dict(),
            "ui_action": self.ui_action.to_dict() if self.ui_action else None,
            "warnings": self.warnings,
            "response_mode": self.response_mode,
            "latency_ms": round(self.latency_ms, 2),
            "token_usage": self.token_usage,
            "cached": self.cached
        }


@dataclass
class ChatMessage:
    """Individual conversational turn."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: float = field(default_factory=time.time)
    ui_action: Optional[UIAction] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "ui_action": self.ui_action.to_dict() if self.ui_action else None
        }


@dataclass
class ChatSession:
    """Persistent chat session state for an inspected sample."""
    conversation_id: str
    sample_id: str
    selected_defect_id: Optional[int] = None
    response_mode: str = "TECHNICAL"
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "sample_id": self.sample_id,
            "selected_defect_id": self.selected_defect_id,
            "response_mode": self.response_mode,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
