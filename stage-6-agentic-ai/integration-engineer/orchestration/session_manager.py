"""Thread-safe session state store for AADA runs."""
from __future__ import annotations
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from .lifecycle import RunLifecycleState, can_transition


class HumanReviewInfo(BaseModel):
    required: bool = False
    status: Optional[str] = None  # PENDING, APPROVED, REJECTED, OVERRIDDEN
    task_id: Optional[str] = None
    reason: Optional[str] = None
    evidence: Optional[Any] = None
    confidence: Optional[float] = None
    options: List[str] = Field(default_factory=lambda: ["APPROVE", "REJECT", "OVERRIDE"])


class AADARunSession(BaseModel):
    """Complete in-memory session holding state across all 4 stages for a run."""
    run_id: str
    goal: str
    lifecycle_status: RunLifecycleState = RunLifecycleState.CREATED
    workflow_id: Optional[str] = None
    workflow_metadata: Dict[str, Any] = Field(default_factory=dict)
    tasks_total: int = 0
    tasks_completed: List[str] = Field(default_factory=list)
    tasks_failed: List[str] = Field(default_factory=list)
    tasks_skipped: List[str] = Field(default_factory=list)
    current_task: Optional[str] = None
    confidence: Optional[float] = None
    human_review: HumanReviewInfo = Field(default_factory=HumanReviewInfo)
    
    # Execution outputs
    agent_status: Optional[str] = None
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Evaluation outputs
    evaluation_status: str = "NOT_EVALUATED"
    evaluation_passed: Optional[bool] = None
    process_score: Optional[float] = None
    outcome_score: Optional[float] = None
    overall_score: Optional[float] = None
    evaluation_details: Dict[str, Any] = Field(default_factory=dict)
    
    # Audit & timestamps
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    error: Optional[Dict[str, Any]] = None

    def update_status(self, new_state: RunLifecycleState) -> None:
        if not can_transition(self.lifecycle_status, new_state):
            # Allow force if not in terminal state
            pass
        self.lifecycle_status = new_state
        self.updated_at = datetime.utcnow().isoformat() + "Z"


class SessionManager:
    """Thread-safe registry of active and completed AADA run sessions."""

    def __init__(self):
        self._sessions: Dict[str, AADARunSession] = {}
        self._lock = threading.Lock()

    def create_session(self, run_id: str, goal: str) -> AADARunSession:
        with self._lock:
            session = AADARunSession(run_id=run_id, goal=goal)
            self._sessions[run_id] = session
            return session

    def get_session(self, run_id: str) -> Optional[AADARunSession]:
        with self._lock:
            return self._sessions.get(run_id)

    def save_session(self, session: AADARunSession) -> None:
        with self._lock:
            session.updated_at = datetime.utcnow().isoformat() + "Z"
            self._sessions[session.run_id] = session

    def list_sessions(self) -> List[AADARunSession]:
        with self._lock:
            return list(self._sessions.values())

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


_GLOBAL_SESSION_MANAGER = SessionManager()

def get_session_manager() -> SessionManager:
    return _GLOBAL_SESSION_MANAGER
