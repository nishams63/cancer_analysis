"""Standardized integration event types."""
from __future__ import annotations
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class IntegrationEventType(str, Enum):
    # Run lifecycle
    RUN_STARTED = "RUN_STARTED"
    WORKFLOW_CREATED = "WORKFLOW_CREATED"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"
    RUN_CANCELLED = "RUN_CANCELLED"

    # Task execution
    TASK_STARTED = "TASK_STARTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    TASK_SKIPPED = "TASK_SKIPPED"

    # Tool & Knowledge
    TOOL_STARTED = "TOOL_STARTED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    KNOWLEDGE_RETRIEVED = "KNOWLEDGE_RETRIEVED"

    # Decisions & Escalations
    BRANCH_EVALUATED = "BRANCH_EVALUATED"
    ESCALATION_TRIGGERED = "ESCALATION_TRIGGERED"

    # Human-in-the-loop
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    HUMAN_APPROVED = "HUMAN_APPROVED"
    HUMAN_REJECTED = "HUMAN_REJECTED"
    HUMAN_OVERRIDE = "HUMAN_OVERRIDE"

    # Evaluation
    EVALUATION_STARTED = "EVALUATION_STARTED"
    EVALUATION_COMPLETED = "EVALUATION_COMPLETED"


class IntegrationEvent(BaseModel):
    """Auditable event emitted across the AADA lifecycle."""
    event_id: str = Field(..., description="Unique event identifier")
    run_id: str = Field(..., description="Run session ID")
    event_type: IntegrationEventType
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    task_id: Optional[str] = None
    message: str = Field(..., description="Concise human-readable event summary")
    payload: Dict[str, Any] = Field(default_factory=dict)
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL
