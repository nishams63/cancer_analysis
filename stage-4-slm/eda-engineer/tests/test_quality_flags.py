"""
Tests for Quality Flag Evaluator and Final Readiness Decision Engine.
"""

import pytest
from quality_flags import QualityFlagEvaluator


@pytest.fixture
def default_thresholds():
    return {
        "context_overflow_rate": {"warning": 0.01, "critical": 0.05},
        "negation_flip_rate": {"warning": 0.01, "critical": 0.05},
        "entity_retention_rate": {"warning": 0.95, "critical": 0.90},
        "patient_leakage": {"critical": 0},
        "duplicate_rate": {"warning": 0.01, "critical": 0.03},
        "risk_class_imbalance": {"warning": 0.20}
    }


def test_quality_evaluator_all_pass(default_thresholds):
    evaluator = QualityFlagEvaluator(default_thresholds)
    clean_metrics = {
        "token_overflow_rate": 0.0,
        "negation_flip_rate": 0.005,
        "entity_retention_rate": 0.99,
        "patient_leakage": 0,
        "exact_duplicate_rate": 0.0,
        "minority_risk_class_percentage": 25.0
    }
    decision = evaluator.evaluate_all_flags(clean_metrics)
    assert decision["final_status"] == "READY"
    assert decision["critical_count"] == 0
    assert decision["warning_count"] == 0


def test_quality_evaluator_warning(default_thresholds):
    evaluator = QualityFlagEvaluator(default_thresholds)
    # Minority risk class is 9.2% (< 20% warning threshold)
    warning_metrics = {
        "token_overflow_rate": 0.0,
        "negation_flip_rate": 0.005,
        "entity_retention_rate": 0.99,
        "patient_leakage": 0,
        "exact_duplicate_rate": 0.0,
        "minority_risk_class_percentage": 9.2
    }
    decision = evaluator.evaluate_all_flags(warning_metrics)
    assert decision["final_status"] == "READY WITH WARNINGS"
    assert decision["warning_count"] >= 1
    assert decision["critical_count"] == 0


def test_quality_evaluator_critical(default_thresholds):
    evaluator = QualityFlagEvaluator(default_thresholds)
    # Patient leakage > 0 is critical
    critical_metrics = {
        "token_overflow_rate": 0.0,
        "negation_flip_rate": 0.005,
        "entity_retention_rate": 0.99,
        "patient_leakage": 3,
        "exact_duplicate_rate": 0.0,
        "minority_risk_class_percentage": 25.0
    }
    decision = evaluator.evaluate_all_flags(critical_metrics)
    assert decision["final_status"] == "NOT READY"
    assert decision["critical_count"] >= 1
