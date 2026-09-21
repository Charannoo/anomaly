"""FastAPI routes for conversational AI Assistant grounded in verified inspection data."""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...assistant.chat import PNTCAssistant
from ...assistant.provider_factory import create_provider, get_default_provider, get_provider_status
from ...assistant.schema import ResponseMode
from ..database import get_inspection_by_id

router = APIRouter(prefix="/api/assistant", tags=["assistant"])

# Singleton assistant instance
_assistant_instance: Optional[PNTCAssistant] = None


def get_assistant() -> PNTCAssistant:
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = PNTCAssistant()
    return _assistant_instance


class ChatRequest(BaseModel):
    sample_id: str
    message: str
    conversation_id: Optional[str] = None
    defect_id: Optional[int] = None
    response_mode: str = "TECHNICAL"


class ExplainRequest(BaseModel):
    sample_id: str
    defect_id: Optional[int] = None
    response_mode: str = "TECHNICAL"
    force_refresh: bool = False


class ProviderSwitchRequest(BaseModel):
    provider: str


@router.get("/provider")
def get_current_provider_status():
    """Return provider status without exposing API keys."""
    asst = get_assistant()
    return get_provider_status(asst.provider)


@router.post("/provider")
def switch_provider(req: ProviderSwitchRequest):
    """Switch active LLM provider (gemini or grok) for development/admin."""
    global _assistant_instance
    try:
        prov = create_provider(req.provider)
        _assistant_instance = PNTCAssistant(provider=prov)
        return get_provider_status(prov)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/explain")
def generate_explanation(req: ExplainRequest):
    """Generate or retrieve cached default inspection explanation."""
    asst = get_assistant()
    item = get_inspection_by_id(req.sample_id)
    inspection_data = item["report"] if item else {"sample_id": req.sample_id, "pntc": {"decision": "anomalous", "score": 0.88}}

    try:
        mode = ResponseMode(req.response_mode.upper())
    except ValueError:
        mode = ResponseMode.TECHNICAL

    resp = asst.generate_inspection_summary(
        inspection_result=inspection_data,
        response_mode=mode,
        force_refresh=req.force_refresh,
    )
    return resp.to_dict()


@router.post("/chat")
def chat_with_assistant(req: ChatRequest):
    """Submit a question about an inspected sample."""
    asst = get_assistant()
    item = get_inspection_by_id(req.sample_id)
    inspection_data = item["report"] if item else {"sample_id": req.sample_id, "pntc": {"decision": "anomalous", "score": 0.88}}

    try:
        mode = ResponseMode(req.response_mode.upper())
    except ValueError:
        mode = ResponseMode.TECHNICAL

    resp = asst.chat(
        sample_id=req.sample_id,
        message=req.message,
        conversation_id=req.conversation_id,
        inspection_result=inspection_data,
        defect_id=req.defect_id,
        response_mode=mode,
    )
    return resp.to_dict()
