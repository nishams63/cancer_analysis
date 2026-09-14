"""Standard branch condition builders and presets."""
from typing import Optional
from schemas.condition import (
    BranchCondition,
    ComparisonOperator,
    BranchAction,
)


def build_missing_rate_branch(
    threshold: float = 0.05,
    action: BranchAction = BranchAction.DATA_CLEANING,
    target_task_id: Optional[str] = None,
) -> BranchCondition:
    return BranchCondition(
        condition_id=f"COND-MISSING-{int(threshold*100)}",
        metric_name="missing_rate",
        operator=ComparisonOperator.GT,
        threshold=threshold,
        action=action,
        target_task_id=target_task_id,
        description=f"Trigger {action.value} if missing_rate exceeds {threshold:.0%}",
    )


def build_critical_missing_branch(
    threshold: float = 0.30,
) -> BranchCondition:
    return BranchCondition(
        condition_id="COND-CRITICAL-MISSING",
        metric_name="missing_rate",
        operator=ComparisonOperator.GT,
        threshold=threshold,
        action=BranchAction.HUMAN_REVIEW,
        description=f"Escalate for human review if missing_rate exceeds {threshold:.0%}",
    )


def build_quality_score_branch(
    threshold: float = 0.70,
    action: BranchAction = BranchAction.DATA_CLEANING,
    target_task_id: Optional[str] = None,
) -> BranchCondition:
    return BranchCondition(
        condition_id="COND-QUALITY-SCORE",
        metric_name="quality_score",
        operator=ComparisonOperator.LT,
        threshold=threshold,
        action=action,
        target_task_id=target_task_id,
        description=f"Trigger {action.value} if quality_score is below {threshold}",
    )


def build_boolean_flag_branch(
    flag_name: str,
    expected_value: bool,
    action: BranchAction,
    target_task_id: Optional[str] = None,
    description: Optional[str] = None,
) -> BranchCondition:
    return BranchCondition(
        condition_id=f"COND-FLAG-{flag_name.upper()}",
        metric_name=flag_name,
        operator=ComparisonOperator.EQ,
        threshold=expected_value,
        action=action,
        target_task_id=target_task_id,
        description=description or f"Trigger {action.value} when {flag_name} == {expected_value}",
    )
