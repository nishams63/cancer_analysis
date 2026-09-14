"""Evaluation Scenario schema definition."""
from __future__ import annotations
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ScenarioCategory(str, Enum):
    SALES = "sales"
    CUSTOMER = "customer"
    ANOMALY = "anomaly"
    EDGE_CASE = "edge_case"
    GENERAL = "general"


class ScenarioDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


class EvaluationScenario(BaseModel):
    """Complete specification of a test scenario against which an agent is evaluated."""
    scenario_id: str = Field(..., description="Unique scenario identifier, e.g. SC-REV-001")
    name: str = Field(..., description="Human-readable scenario title")
    description: str = Field(..., description="Detailed analytical premise")
    category: ScenarioCategory = Field(..., description="Analytical problem domain")
    dataset: str = Field(..., description="Path or reference to dataset")
    user_goal: str = Field(..., description="High-level user request presented to agent")
    expected_workflow: str = Field(..., description="Workflow ID expected to be selected, e.g. WF-REVENUE-001")
    expected_tasks: List[str] = Field(default_factory=list, description="IDs of tasks expected to be completed")
    expected_tools: List[str] = Field(default_factory=list, description="Analytical tools expected to be invoked")
    expected_knowledge: List[str] = Field(default_factory=list, description="Knowledge items or queries expected to be retrieved")
    expected_branches: List[Dict[str, Any]] = Field(default_factory=list, description="Expected branch conditions and target tasks")
    expected_escalations: List[str] = Field(default_factory=list, description="Escalation rules or triggers expected to fire")
    ground_truth_id: Optional[str] = Field(None, description="Linked ground truth specification ID")
    difficulty: ScenarioDifficulty = Field(ScenarioDifficulty.MEDIUM, description="Scenario complexity tier")
    tags: List[str] = Field(default_factory=list, description="Taxonomy and search tags")
