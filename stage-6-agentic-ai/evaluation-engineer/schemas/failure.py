"""Standardized Failure Taxonomy (F01 - F20) and failure record models."""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class FailureSeverity(str, Enum):
    CRITICAL = "critical"  # Immediate scenario failure (safety violation, missed escalation)
    HIGH = "high"          # Major analytical inaccuracy or missing key step
    MEDIUM = "medium"      # Suboptimal tool choice, minor numerical deviation
    LOW = "low"            # Minor formatting or stylistic flaw


class FailureCategory(str, Enum):
    F01_GOAL_MISUNDERSTANDING = "F01_GOAL_MISUNDERSTANDING"
    F02_WRONG_WORKFLOW = "F02_WRONG_WORKFLOW"
    F03_TASK_ORDER_ERROR = "F03_TASK_ORDER_ERROR"
    F04_DEPENDENCY_VIOLATION = "F04_DEPENDENCY_VIOLATION"
    F05_WRONG_TOOL = "F05_WRONG_TOOL"
    F06_MISSING_KNOWLEDGE_RETRIEVAL = "F06_MISSING_KNOWLEDGE_RETRIEVAL"
    F07_IRRELEVANT_KNOWLEDGE = "F07_IRRELEVANT_KNOWLEDGE"
    F08_WRONG_BRANCH = "F08_WRONG_BRANCH"
    F09_MISSED_ESCALATION = "F09_MISSED_ESCALATION"
    F10_UNNECESSARY_ESCALATION = "F10_UNNECESSARY_ESCALATION"
    F11_UNSUPPORTED_CAUSALITY = "F11_UNSUPPORTED_CAUSALITY"
    F12_HALLUCINATED_FACT = "F12_HALLUCINATED_FACT"
    F13_NUMERICAL_ERROR = "F13_NUMERICAL_ERROR"
    F14_CONFLICT_RESOLUTION_ERROR = "F14_CONFLICT_RESOLUTION_ERROR"
    F15_LOW_CONFIDENCE_NOT_ESCALATED = "F15_LOW_CONFIDENCE_NOT_ESCALATED"
    F16_TOOL_EXECUTION_FAILURE = "F16_TOOL_EXECUTION_FAILURE"
    F17_INFINITE_OR_EXCESSIVE_LOOP = "F17_INFINITE_OR_EXCESSIVE_LOOP"
    F18_UNAUTHORIZED_TOOL = "F18_UNAUTHORIZED_TOOL"
    F19_INCOMPLETE_RESULT = "F19_INCOMPLETE_RESULT"
    F20_UNSUPPORTED_RECOMMENDATION = "F20_UNSUPPORTED_RECOMMENDATION"
    # Deliberation Failure Taxonomy (D01 - D14)
    D01_NO_PLAN = "D01_NO_PLAN"
    D02_INCOMPLETE_PLAN = "D02_INCOMPLETE_PLAN"
    D03_WRONG_PLAN = "D03_WRONG_PLAN"
    D04_PREMATURE_CONCLUSION = "D04_PREMATURE_CONCLUSION"
    D05_INSUFFICIENT_EVIDENCE = "D05_INSUFFICIENT_EVIDENCE"
    D06_FAILED_CONTRADICTION_DETECTION = "D06_FAILED_CONTRADICTION_DETECTION"
    D07_UNSUPPORTED_HYPOTHESIS = "D07_UNSUPPORTED_HYPOTHESIS"
    D08_UNCALIBRATED_CONFIDENCE = "D08_UNCALIBRATED_CONFIDENCE"
    D09_FAILED_REPLANNING = "D09_FAILED_REPLANNING"
    D10_MISSED_HUMAN_REVIEW = "D10_MISSED_HUMAN_REVIEW"
    D11_FAILED_VERIFICATION = "D11_FAILED_VERIFICATION"
    D12_PATIENT_FACT_HALLUCINATION = "D12_PATIENT_FACT_HALLUCINATION"
    D13_EVIDENCE_MISATTRIBUTION = "D13_EVIDENCE_MISATTRIBUTION"
    D14_UNSUPPORTED_RECOMMENDATION = "D14_UNSUPPORTED_RECOMMENDATION"


DeliberationFailureCategory = FailureCategory


class FailureRecord(BaseModel):
    """Auditable diagnostics record for an identified agent execution defect."""
    failure_id: str = Field(..., description="Unique defect ID e.g. FAIL-001")
    scenario_id: str = Field(..., description="Scenario where defect occurred")
    run_id: str = Field(..., description="Agent execution run ID")
    task_id: Optional[str] = Field(None, description="Task ID where failure occurred")
    category: FailureCategory = Field(..., description="Taxonomy classification code e.g. F01 or D01")
    severity: FailureSeverity = Field(..., description="Impact severity classification")
    expected: str = Field(..., description="Expected behavior or contract")
    actual: str = Field(..., description="Actual agent behavior observed")
    evidence: str = Field(..., description="Excerpt or trace proof")
    explanation: str = Field(..., description="Detailed diagnostic explanation")
    recommended_fix: str = Field(..., description="Prescription for resolving defect")
