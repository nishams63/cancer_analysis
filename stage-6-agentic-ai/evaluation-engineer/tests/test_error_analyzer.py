"""Tests for ErrorAnalyzer."""
from analysis.error_analyzer import ErrorAnalyzer
from schemas.failure import FailureRecord, FailureSeverity, FailureCategory


def test_error_analyzer_summary():
    f1 = FailureRecord(
        failure_id="F-1",
        scenario_id="SC-001",
        run_id="R-001",
        category=FailureCategory.F05_WRONG_TOOL,
        severity=FailureSeverity.MEDIUM,
        expected="tool A",
        actual="tool B",
        evidence="call tool B",
        explanation="Wrong tool selected",
        recommended_fix="Update prompt",
    )
    f2 = FailureRecord(
        failure_id="F-2",
        scenario_id="SC-002",
        run_id="R-002",
        category=FailureCategory.F09_MISSED_ESCALATION,
        severity=FailureSeverity.CRITICAL,
        expected="Escalate to human",
        actual="No escalation",
        evidence="Confidence 0.20",
        explanation="Critical missed escalation",
        recommended_fix="Lower threshold",
    )
    analysis = ErrorAnalyzer.analyze_failures([f1, f2])
    assert analysis["total_defects"] == 2
    assert analysis["critical_defects"] == 1
    assert analysis["medium_defects"] == 1
    assert "F05_WRONG_TOOL" in analysis["category_distribution"]
    assert "F09_MISSED_ESCALATION" in analysis["category_distribution"]
    assert len(analysis["prioritized_fixes"]) == 2
