"""Branch and validation condition models."""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ComparisonOperator(str, Enum):
    """Supported operators for condition evaluation."""
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"
    NEQ = "neq"
    IN_SET = "in_set"
    CONTAINS = "contains"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class BranchAction(str, Enum):
    """Action to execute if condition evaluates to True."""
    CONTINUE_ANALYSIS = "continue_analysis"
    SKIP_TASK = "skip_task"
    EXECUTE_BRANCH = "execute_branch"
    HUMAN_REVIEW = "human_review"
    DATA_CLEANING = "data_cleaning"
    TERMINATE = "terminate"


class BranchCondition(BaseModel):
    """Explicit conditional logic governing task flow execution."""
    condition_id: str = Field(..., description="Unique identifier e.g. COND-001")
    metric_name: str = Field(..., description="Context variable/metric to evaluate")
    operator: ComparisonOperator = Field(..., description="Comparison operation")
    threshold: Any = Field(None, description="Reference threshold value for comparison")
    action: BranchAction = Field(..., description="Action to take when condition is met")
    target_task_id: Optional[str] = Field(None, description="Optional target task ID to jump to or branch into")
    description: Optional[str] = Field(None, description="Human readable rationale")

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against runtime execution context."""
        val = context.get(self.metric_name)

        if self.operator == ComparisonOperator.IS_NULL:
            return val is None
        if self.operator == ComparisonOperator.NOT_NULL:
            return val is not None

        if val is None:
            return False

        try:
            if self.operator == ComparisonOperator.GT:
                return float(val) > float(self.threshold)
            elif self.operator == ComparisonOperator.GTE:
                return float(val) >= float(self.threshold)
            elif self.operator == ComparisonOperator.LT:
                return float(val) < float(self.threshold)
            elif self.operator == ComparisonOperator.LTE:
                return float(val) <= float(self.threshold)
            elif self.operator == ComparisonOperator.EQ:
                return val == self.threshold
            elif self.operator == ComparisonOperator.NEQ:
                return val != self.threshold
            elif self.operator == ComparisonOperator.IN_SET:
                return val in (self.threshold if isinstance(self.threshold, (list, set, tuple)) else [self.threshold])
            elif self.operator == ComparisonOperator.CONTAINS:
                return self.threshold in val
        except (ValueError, TypeError):
            return False

        return False
