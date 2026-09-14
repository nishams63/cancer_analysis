"""Ground Truth definitions and analytical reference contracts."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ExpectedFact(BaseModel):
    """A verified factual metric or assertion from the underlying data."""
    fact_id: str = Field(..., description="Fact identifier e.g. FACT-001")
    description: str = Field(..., description="Human readable description of the fact")
    metric_name: Optional[str] = Field(None, description="Associated metric variable if numerical")
    expected_value: Optional[Any] = Field(None, description="Expected benchmark value")
    tolerance_pct: float = Field(0.05, description="Relative tolerance percentage allowed (e.g. 0.05 for +/-5%)")
    is_critical: bool = Field(False, description="Whether contradiction of this fact constitutes a critical error")


class AcceptableCause(BaseModel):
    """An empirically valid causal explanation for the analytical findings."""
    cause: str = Field(..., description="Root cause description")
    min_score: float = Field(0.0, description="Minimum acceptable contribution/evidence score")
    max_rank: int = Field(3, description="Maximum acceptable rank among identified causes")


class GroundTruth(BaseModel):
    """Authoritative reference truths decoupled from agent outputs."""
    ground_truth_id: str = Field(..., description="Ground truth ID e.g. GT-REV-001")
    scenario_id: str = Field(..., description="Associated scenario ID")
    expected_facts: List[ExpectedFact] = Field(default_factory=list, description="Quantitative & categorical facts")
    acceptable_causes: List[AcceptableCause] = Field(default_factory=list, description="Permissible causal explanations")
    forbidden_claims: List[str] = Field(default_factory=list, description="Unsupported, misleading, or hallucinated claims")
    acceptable_alternative_tools: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Tool name -> list of valid substitute tools (e.g. eda_analysis -> [statistical_analysis])"
    )
    expected_recommendation_themes: List[str] = Field(default_factory=list, description="Actionable themes required in recommendations")
    numerical_tolerances: Dict[str, float] = Field(default_factory=dict, description="Metric-specific tolerances")
