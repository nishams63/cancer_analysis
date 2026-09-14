"""Standard escalation rule builders and factory presets."""
from schemas.escalation import (
    EscalationRule,
    EscalationTrigger,
    EscalationAction,
)
from schemas.condition import ComparisonOperator


def build_poor_quality_escalation(
    threshold: float = 0.50,
) -> EscalationRule:
    return EscalationRule(
        rule_id="ESC-001-POOR-QUALITY",
        name="Critical Data Quality Failure",
        trigger=EscalationTrigger.POOR_DATA_QUALITY,
        condition_metric="quality_score",
        operator=ComparisonOperator.LT,
        threshold=threshold,
        action=EscalationAction.HUMAN_REVIEW,
        severity="critical",
        message="Dataset quality score is below minimum viable analysis threshold",
    )


def build_low_confidence_rca_escalation(
    threshold: float = 0.60,
) -> EscalationRule:
    return EscalationRule(
        rule_id="ESC-002-LOW-CONFIDENCE-RCA",
        name="Low Confidence Root Cause",
        trigger=EscalationTrigger.CONFLICTING_EVIDENCE,
        condition_metric="root_cause_confidence",
        operator=ComparisonOperator.LT,
        threshold=threshold,
        action=EscalationAction.HUMAN_REVIEW,
        severity="high",
        message="Root cause model confidence fell below acceptable decision threshold",
    )


def build_similar_cause_scores_escalation(
    margin: float = 0.05,
) -> EscalationRule:
    return EscalationRule(
        rule_id="ESC-003-SIMILAR-CAUSE-SCORES",
        name="Ambiguous Competing Root Causes",
        trigger=EscalationTrigger.SIMILAR_CAUSE_SCORES,
        condition_metric="cause_score_margin",
        operator=ComparisonOperator.LT,
        threshold=margin,
        action=EscalationAction.HUMAN_REVIEW,
        severity="high",
        message="Top root causes have indistinguishable confidence margins",
    )


def build_missing_columns_escalation() -> EscalationRule:
    return EscalationRule(
        rule_id="ESC-004-MISSING-COLUMNS",
        name="Missing Required Schema Columns",
        trigger=EscalationTrigger.MISSING_REQUIRED_COLUMNS,
        condition_metric="missing_required_columns_count",
        operator=ComparisonOperator.GT,
        threshold=0,
        action=EscalationAction.WORKFLOW_PAUSE,
        severity="critical",
        message="Critical feature columns missing from dataset",
    )


def build_max_retries_escalation(
    max_retries: int = 3,
) -> EscalationRule:
    return EscalationRule(
        rule_id="ESC-005-MAX-RETRIES",
        name="Task Max Retries Exceeded",
        trigger=EscalationTrigger.MAX_RETRIES_EXCEEDED,
        condition_metric="retry_count",
        operator=ComparisonOperator.GTE,
        threshold=max_retries,
        action=EscalationAction.WORKFLOW_ESCALATION,
        severity="critical",
        message=f"Task failed repeatedly after {max_retries} retry attempts",
    )


def build_high_impact_action_escalation() -> EscalationRule:
    return EscalationRule(
        rule_id="ESC-006-HIGH-IMPACT-ACTION",
        name="High-Impact Business Recommendation",
        trigger=EscalationTrigger.HIGH_IMPACT_ACTION,
        condition_metric="business_impact_level",
        operator=ComparisonOperator.EQ,
        threshold="high",
        action=EscalationAction.HUMAN_REVIEW,
        severity="warning",
        message="Recommendation carries high operational/financial impact; supervisor signoff required",
    )
