"""Escalation rules and trigger specifications."""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from .condition import ComparisonOperator


class EscalationTrigger(str, Enum):
    """Standard event triggers requiring supervisor intervention."""
    POOR_DATA_QUALITY = "poor_data_quality"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    SIMILAR_CAUSE_SCORES = "similar_cause_scores"
    MISSING_REQUIRED_COLUMNS = "missing_required_columns"
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
    HIGH_IMPACT_ACTION = "high_impact_action"
    ANOMALY_SEVERITY_CRITICAL = "anomaly_severity_critical"


class EscalationAction(str, Enum):
    """Resolution pathway for triggered escalation."""
    HUMAN_REVIEW = "human_review"
    WORKFLOW_PAUSE = "workflow_pause"
    WORKFLOW_ESCALATION = "workflow_escalation"
    TERMINATE = "terminate"


class EscalationRule(BaseModel):
    """Formal escalation policy definition to prevent silent failures."""
    rule_id: str = Field(..., description="Unique rule ID e.g. ESC-001")
    name: str = Field(..., description="Human-readable rule name")
    trigger: EscalationTrigger = Field(..., description="Categorical trigger type")
    condition_metric: str = Field(..., description="Context variable monitored")
    operator: ComparisonOperator = Field(..., description="Comparison operator")
    threshold: Any = Field(..., description="Trigger threshold")
    action: EscalationAction = Field(..., description="Resolution action required")
    severity: str = Field("critical", description="Severity level: warning, high, critical")
    message: str = Field(..., description="Explanation provided during escalation")

    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Evaluate rule against context. Returns (is_triggered, explanation)."""
        val = context.get(self.condition_metric)
        if val is None:
            return False, None

        triggered = False
        try:
            if self.operator == ComparisonOperator.LT:
                triggered = float(val) < float(self.threshold)
            elif self.operator == ComparisonOperator.LTE:
                triggered = float(val) <= float(self.threshold)
            elif self.operator == ComparisonOperator.GT:
                triggered = float(val) > float(self.threshold)
            elif self.operator == ComparisonOperator.GTE:
                triggered = float(val) >= float(self.threshold)
            elif self.operator == ComparisonOperator.EQ:
                triggered = val == self.threshold
            elif self.operator == ComparisonOperator.NEQ:
                triggered = val != self.threshold
            elif self.operator == ComparisonOperator.IN_SET:
                triggered = val in self.threshold
        except (ValueError, TypeError):
            return False, None

        if triggered:
            formatted_msg = f"[{self.severity.upper()}] {self.name}: {self.message} (Observed: {val}, Threshold: {self.threshold})"
            return True, formatted_msg

        return False, None
