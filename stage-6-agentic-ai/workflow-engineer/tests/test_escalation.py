"""Unit tests for escalation rules."""
from conditions.escalation_conditions import (
    build_poor_quality_escalation,
    build_low_confidence_rca_escalation,
    build_similar_cause_scores_escalation,
    build_max_retries_escalation,
)


def test_poor_quality_escalation():
    rule = build_poor_quality_escalation(threshold=0.50)
    triggered, msg = rule.evaluate({"quality_score": 0.35})
    assert triggered is True
    assert "Critical Data Quality Failure" in msg

    not_triggered, _ = rule.evaluate({"quality_score": 0.85})
    assert not_triggered is False


def test_low_confidence_rca_escalation():
    rule = build_low_confidence_rca_escalation(threshold=0.60)
    triggered, msg = rule.evaluate({"root_cause_confidence": 0.45})
    assert triggered is True
    assert "Low Confidence Root Cause" in msg


def test_similar_cause_scores_escalation():
    rule = build_similar_cause_scores_escalation(margin=0.05)
    triggered, msg = rule.evaluate({"cause_score_margin": 0.02})
    assert triggered is True
    assert "Ambiguous Competing Root Causes" in msg


def test_max_retries_escalation():
    rule = build_max_retries_escalation(max_retries=3)
    triggered, msg = rule.evaluate({"retry_count": 3})
    assert triggered is True
    assert "Task Max Retries Exceeded" in msg
