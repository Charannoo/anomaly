"""Session state management for PNTC AI Assistant conversations.

Phase A16: Chat Session State.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from typing import Dict, List, Optional
from .schema import ChatMessage, ChatSession, UIAction


class SessionManager:
    """Thread-safe conversational session manager storing explicit application state."""

    def __init__(self, max_sessions: int = 1000):
        self._sessions: Dict[str, ChatSession] = {}
        self._lock = threading.Lock()
        self._max_sessions = max_sessions

    def get_or_create_session(
        self,
        conversation_id: Optional[str],
        sample_id: str,
        defect_id: Optional[int] = None,
        response_mode: str = "TECHNICAL",
    ) -> ChatSession:
        """Retrieve existing session or instantiate a new session."""
        with self._lock:
            cid = conversation_id or str(uuid.uuid4())
            if cid in self._sessions:
                sess = self._sessions[cid]
                # Update sample if changed
                if sample_id and sess.sample_id != sample_id:
                    sess.sample_id = sample_id
                    sess.selected_defect_id = defect_id
                elif defect_id is not None:
                    sess.selected_defect_id = defect_id
                if response_mode:
                    sess.response_mode = response_mode
                sess.updated_at = time.time()
                return sess

            # Evict oldest if capacity reached
            if len(self._sessions) >= self._max_sessions:
                oldest_key = min(self._sessions.keys(), key=lambda k: self._sessions[k].updated_at)
                del self._sessions[oldest_key]

            sess = ChatSession(
                conversation_id=cid,
                sample_id=sample_id,
                selected_defect_id=defect_id,
                response_mode=response_mode,
                created_at=time.time(),
                updated_at=time.time(),
            )
            self._sessions[cid] = sess
            return sess

    def get_session(self, conversation_id: str) -> Optional[ChatSession]:
        with self._lock:
            return self._sessions.get(conversation_id)

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        ui_action: Optional[UIAction] = None,
    ) -> Optional[ChatMessage]:
        """Append a message turn to the session."""
        with self._lock:
            sess = self._sessions.get(conversation_id)
            if not sess:
                return None
            msg = ChatMessage(
                role=role,
                content=content,
                timestamp=time.time(),
                ui_action=ui_action,
            )
            sess.messages.append(msg)
            sess.updated_at = time.time()
            return msg

    def update_defect_selection(self, conversation_id: str, defect_id: Optional[int]) -> None:
        """Update the actively selected defect ID in the session."""
        with self._lock:
            sess = self._sessions.get(conversation_id)
            if sess:
                sess.selected_defect_id = defect_id
                sess.updated_at = time.time()

    def resolve_defect_reference(
        self,
        conversation_id: str,
        user_message: str,
        available_defect_ids: List[int],
    ) -> Optional[int]:
        """Resolve defect ID from user text or fallback to session state."""
        # Check if user explicitly mentioned a defect number (e.g. "defect 2", "region 1", "#2")
        match = re.search(r"(?:defect|region|anomaly|#)\s*([0-9]+)", user_message, re.IGNORECASE)
        if match:
            try:
                did = int(match.group(1))
                if did in available_defect_ids:
                    self.update_defect_selection(conversation_id, did)
                    return did
            except ValueError:
                pass

        with self._lock:
            sess = self._sessions.get(conversation_id)
            if sess and sess.selected_defect_id in available_defect_ids:
                return sess.selected_defect_id

        # Default to first defect if available
        if available_defect_ids:
            return available_defect_ids[0]
        return None

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


# Global singleton instance
global_session_manager = SessionManager()
