"""Unit tests for branch conditions."""
from schemas.condition import (
    BranchCondition,
    ComparisonOperator,
    BranchAction,
)
from conditions.branch_conditions import (
    build_missing_rate_branch,
    build_critical_missing_branch,
    build_quality_score_branch,
)


def test_missing_rate_branch():
    cond = build_missing_rate_branch(threshold=0.05, action=BranchAction.DATA_CLEANING)
    assert cond.evaluate({"missing_rate": 0.08}) is True
    assert cond.evaluate({"missing_rate": 0.02}) is False


def test_critical_missing_branch():
    cond = build_critical_missing_branch(threshold=0.30)
    assert cond.evaluate({"missing_rate": 0.35}) is True
    assert cond.evaluate({"missing_rate": 0.25}) is False


def test_quality_score_branch():
    cond = build_quality_score_branch(threshold=0.70)
    assert cond.evaluate({"quality_score": 0.55}) is True
    assert cond.evaluate({"quality_score": 0.85}) is False


def test_comparison_operators():
    cond_eq = BranchCondition(
        condition_id="C1",
        metric_name="status",
        operator=ComparisonOperator.EQ,
        threshold="valid",
        action=BranchAction.CONTINUE_ANALYSIS,
    )
    assert cond_eq.evaluate({"status": "valid"}) is True
    assert cond_eq.evaluate({"status": "invalid"}) is False

    cond_in = BranchCondition(
        condition_id="C2",
        metric_name="tier",
        operator=ComparisonOperator.IN_SET,
        threshold=["gold", "platinum"],
        action=BranchAction.EXECUTE_BRANCH,
    )
    assert cond_in.evaluate({"tier": "gold"}) is True
    assert cond_in.evaluate({"tier": "bronze"}) is False
