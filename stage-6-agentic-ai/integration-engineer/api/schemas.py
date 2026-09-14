"""Request and response schemas for AADA Integration API."""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class StartRunRequest(BaseModel):
    goal: str = Field(..., min_length=3, description="High-level natural language analytical goal")
    sync: bool = Field(False, description="Whether to execute synchronously before returning")


class StartRunResponse(BaseModel):
    run_id: str
    status: str
    message: str = "AADA analysis started."


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    current_task: Optional[str] = None
    completed_tasks: List[str] = Field(default_factory=list)
    failed_tasks: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    escalation_status: str = "NONE"
    progress: Dict[str, Any] = Field(default_factory=dict)


class HumanDecisionRequest(BaseModel):
    reason: str = Field("Approved by analyst", description="Human reviewer justification")


class HumanOverrideRequest(BaseModel):
    decision: str = Field(..., description="Action or hypothesis override decision")
    reason: str = Field(..., description="Justification for override")
    override_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UnifiedRunResponse(BaseModel):
    """Authoritative integration response combining all component outputs."""
    run_id: str
    status: str
    goal: str
    workflow: Dict[str, Any]
    analysis: Dict[str, Any]
    confidence: Optional[float] = None
    evaluation: Dict[str, Any]
    human_review: Dict[str, Any]
    trace: Dict[str, Any]
