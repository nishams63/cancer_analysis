"""Process, Outcome, Safety, and Execution Metrics models."""
from __future__ import annotations
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class ProcessMetrics(BaseModel):
    """Detailed process metrics assessing execution fidelity."""
    workflow_adherence: float = Field(..., ge=0.0, le=1.0, description="Fraction of required workflow tasks executed")
    task_ordering_accuracy: float = Field(1.0, ge=0.0, le=1.0, description="Fidelity to DAG topological ordering")
    tool_selection_accuracy: float = Field(..., ge=0.0, le=1.0, description="Accuracy of tool selection")
    tool_precision: float = Field(1.0, ge=0.0, le=1.0, description="Precision of chosen tools vs expected")
    tool_recall: float = Field(1.0, ge=0.0, le=1.0, description="Recall of required tools")
    knowledge_retrieval_accuracy: float = Field(1.0, ge=0.0, le=1.0, description="Retrieval contract fidelity")
    knowledge_relevance: float = Field(1.0, ge=0.0, le=1.0, description="Relevance score of retrieved items")
    branch_accuracy: float = Field(1.0, ge=0.0, le=1.0, description="Correctness of conditional routing decisions")
    escalation_precision: float = Field(1.0, ge=0.0, le=1.0, description="Precision of raised escalations")
    escalation_recall: float = Field(1.0, ge=0.0, le=1.0, description="Recall of required escalations (critical)")
    escalation_f1: float = Field(1.0, ge=0.0, le=1.0, description="F1 harmonic mean of escalation decisions")
    evidence_grounding: float = Field(1.0, ge=0.0, le=1.0, description="Proportion of findings substantiated by data")
    safety_score: float = Field(1.0, ge=0.0, le=1.0, description="Safety adherence score")
    reliability_score: float = Field(1.0, ge=0.0, le=1.0, description="Execution robustness score")
    weighted_process_score: float = Field(..., ge=0.0, le=1.0, description="Overall weighted composite process score")


class OutcomeMetrics(BaseModel):
    """Metrics evaluating final analytical outcome and recommendations."""
    analytical_accuracy: float = Field(..., ge=0.0, le=1.0, description="Empirical correctness of findings")
    numerical_accuracy: float = Field(1.0, ge=0.0, le=1.0, description="Numerical agreement within tolerances")
    factual_accuracy: float = Field(1.0, ge=0.0, le=1.0, description="Factual claims verified by ground truth")
    root_cause_accuracy: float = Field(1.0, ge=0.0, le=1.0, description="Validity of identified primary root cause")
    recommendation_relevance: float = Field(1.0, ge=0.0, le=1.0, description="Direct relevance of action plan to findings")
    recommendation_actionability: float = Field(1.0, ge=0.0, le=1.0, description="Clarity and feasibility of recommendations")
    recommendation_grounding: float = Field(1.0, ge=0.0, le=1.0, description="Evidence linkage in recommendations")
    overall_outcome_score: float = Field(..., ge=0.0, le=1.0, description="Composite outcome quality score")


class SafetyMetrics(BaseModel):
    """Quantification of safety violations and hazardous behaviors."""
    critical_violations: int = 0
    high_violations: int = 0
    medium_violations: int = 0
    low_violations: int = 0
    missed_escalations: int = 0
    hallucinated_facts: int = 0
    unauthorized_tools: int = 0
    has_critical_failure: bool = False


class ExecutionMetrics(BaseModel):
    """Runtime observability telemetry."""
    duration_seconds: float = 0.0
    tool_calls: int = 0
    llm_calls: int = 0
    retry_count: int = 0
    human_overrides: int = 0
    status: str = "completed"
