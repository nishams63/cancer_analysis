"""Scenario and Overall Evaluation Result schemas."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from .metrics import ProcessMetrics, OutcomeMetrics, SafetyMetrics, ExecutionMetrics
from .failure import FailureRecord


class DimensionScore(BaseModel):
    """Score on an individual evaluation dimension."""
    dimension: str
    score: float
    weight: float
    details: Dict[str, Any] = Field(default_factory=dict)


class ScenarioEvaluationResult(BaseModel):
    """Comprehensive evaluation result for a single scenario run."""
    scenario_id: str
    run_id: str
    passed: bool
    overall_score: float = Field(..., ge=0.0, le=1.0)
    process_score: float = Field(..., ge=0.0, le=1.0)
    outcome_score: float = Field(..., ge=0.0, le=1.0)
    process_metrics: ProcessMetrics
    outcome_metrics: OutcomeMetrics
    safety_metrics: SafetyMetrics
    execution_metrics: ExecutionMetrics
    failures: List[FailureRecord] = Field(default_factory=list)
    verdict_rationale: str = ""


class OverallEvaluationResult(BaseModel):
    """Aggregated evaluation across an entire experiment benchmark suite."""
    evaluation_id: str
    timestamp: str
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    success_rate: float = Field(..., ge=0.0, le=1.0)
    average_process_score: float
    average_outcome_score: float
    average_overall_score: float
    scenario_results: List[ScenarioEvaluationResult] = Field(default_factory=list)
    failure_counts_by_category: Dict[str, int] = Field(default_factory=dict)
    failure_counts_by_severity: Dict[str, int] = Field(default_factory=dict)
    duration_seconds: float = 0.0
