"""Structured auditable execution trace event schemas."""
from __future__ import annotations
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Classification of trace events for auditability."""
    WORKFLOW_STARTED = "workflow_started"
    TASK_STARTED = "task_started"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OBSERVATION = "observation"
    BRANCH_DECISION = "branch_decision"
    TASK_COMPLETED = "task_completed"
    TASK_SKIPPED = "task_skipped"
    TASK_FAILED = "task_failed"
    RETRY = "retry"
    ESCALATION = "escalation"
    HUMAN_REVIEW_REQUESTED = "human_review_requested"
    HUMAN_OVERRIDE = "human_override"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"


class TraceEvent(BaseModel):
    """Auditable trace record capturing an execution step without private chain-of-thought."""
    trace_id: str = Field(..., description="Unique event identifier e.g. TRACE-001")
    run_id: str = Field(..., description="Associated run ID")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    task_id: Optional[str] = Field(None, description="Task context for event")
    event_type: EventType = Field(..., description="Event taxonomy classification")
    tool_name: Optional[str] = Field(None, description="Invoked tool name")
    tool_input: Optional[Dict[str, Any]] = Field(None, description="Structured arguments passed to tool")
    tool_output_summary: Optional[Dict[str, Any]] = Field(None, description="Concise summary of tool outputs")
    decision: Optional[str] = Field(None, description="Action or routing decision taken")
    decision_reason: Optional[str] = Field(None, description="Auditable rationale for decision")
    confidence: Optional[float] = Field(None, description="Evidence confidence score (0.0 to 1.0)")
    status: str = Field("success", description="Status outcome: success, failure, warning, pending")
    error: Optional[Dict[str, Any]] = Field(None, description="Structured error payload if failed")
    next_task_id: Optional[str] = Field(None, description="Target task routed to")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Supplementary event context")
