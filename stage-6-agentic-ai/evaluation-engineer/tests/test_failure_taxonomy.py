"""Tests for standardized failure taxonomy coverage."""
from analysis.failure_taxonomy import TAXONOMY_REGISTRY
from schemas.failure import FailureCategory, FailureSeverity


def test_all_twenty_failure_codes_present():
    assert len(TAXONOMY_REGISTRY) == 20
    assert FailureCategory.F01_GOAL_MISUNDERSTANDING in TAXONOMY_REGISTRY
    assert FailureCategory.F09_MISSED_ESCALATION in TAXONOMY_REGISTRY
    assert FailureCategory.F12_HALLUCINATED_FACT in TAXONOMY_REGISTRY
    assert FailureCategory.F20_UNSUPPORTED_RECOMMENDATION in TAXONOMY_REGISTRY


def test_critical_severities():
    missed_esc = TAXONOMY_REGISTRY[FailureCategory.F09_MISSED_ESCALATION]
    assert missed_esc["default_severity"] == FailureSeverity.CRITICAL

    halluc = TAXONOMY_REGISTRY[FailureCategory.F12_HALLUCINATED_FACT]
    assert halluc["default_severity"] == FailureSeverity.CRITICAL
