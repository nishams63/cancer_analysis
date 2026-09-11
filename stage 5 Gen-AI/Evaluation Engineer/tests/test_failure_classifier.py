"""Tests for Failure Classifier."""
import pytest
from src.evaluation.failure_classifier import FailureClassifier


def test_failure_classification():
    classifier = FailureClassifier()
    stage_res = {
        "stage": "stage3",
        "scenario_id": "PROMPT-R01",
        "failure_codes": ["F01", "F04"],
        "expected_output": "MET absent",
        "actual_output": "MET present",
        "confidence": 0.94,
        "evidence": "Flipped negation"
    }
    classified = classifier.classify_stage_failure(stage_res)
    assert len(classified) == 2
    assert classified[0]["failure_code"] == "F01"
    assert classified[1]["failure_code"] == "F04"
    assert classified[0]["confidence"] == 0.94
