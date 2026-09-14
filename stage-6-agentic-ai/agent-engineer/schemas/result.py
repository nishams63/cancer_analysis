"""Final structured analytical outcome delivered by the Agent."""
from __future__ import annotations
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """Structured final result synthesizing findings, evidence, and recommendations."""
    run_id: str = Field(..., description="Run identifier")
    workflow_id: str = Field(..., description="Workflow executed")
    status: str = Field(..., description="Outcome status: completed, failed, escalated")
    objective: str = Field(..., description="High-level analytical goal")
    findings: List[Dict[str, Any]] = Field(default_factory=list, description="Ranked empirical findings")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Grounding evidence records")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Strategic recommendations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence derived from evidence")
    escalated: bool = Field(False, description="Whether escalation was triggered")
    completed_tasks: List[str] = Field(default_factory=list, description="IDs of completed tasks")
    trace_id: str = Field(..., description="Root trace ID for audit lookup")
    artifacts: List[Dict[str, Any]] = Field(default_factory=list, description="Generated artifacts catalog")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Observability metrics summary")
