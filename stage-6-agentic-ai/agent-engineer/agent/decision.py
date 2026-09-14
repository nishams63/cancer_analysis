"""Structured decision models."""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class AgentDecision(BaseModel):
    action: str = Field("continue", description="Routing action: continue, branch, pause, escalate, terminate")
    tool: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    reason: str = Field("", description="Auditable rationale")
    target_task_id: Optional[str] = None
