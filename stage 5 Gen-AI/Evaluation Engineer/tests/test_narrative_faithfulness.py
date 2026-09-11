"""Tests for Narrative Faithfulness Evaluator."""
import pytest
from src.evaluation.narrative_faithfulness import NarrativeFaithfulnessEvaluator


def test_narrative_faithfulness_pass(sample_valid_patient):
    evaluator = NarrativeFaithfulnessEvaluator()
    narrative = {
        "narrative_text": "A 62-year-old female presents with NSCLC. Molecular profiling shows EGFR L858R and secondary MET amplification."
    }
    res = evaluator.evaluate_narrative(sample_valid_patient, narrative)
    assert res["status"] == "PASS"
    assert res["narrative_faithfulness_score"] >= 0.85
    assert res["has_contradiction"] is False


def test_narrative_hallucination_detected(sample_valid_patient):
    evaluator = NarrativeFaithfulnessEvaluator()
    # Patient has missing PD-L1
    sample_valid_patient["biomarkers"]["pdl1_tps"] = None
    narrative = {
        "narrative_text": "A 62-year-old female with NSCLC. Note: pdl1_tps is 95% strongly positive."
    }
    res = evaluator.evaluate_narrative(sample_valid_patient, narrative)
    assert res["has_hallucination"] is True


def test_narrative_contradiction_detected(sample_valid_patient):
    evaluator = NarrativeFaithfulnessEvaluator()
    narrative = {
        "narrative_text": "Initial report: no met alteration. Follow-up: met positive amplification."
    }
    res = evaluator.evaluate_narrative(sample_valid_patient, narrative)
    assert res["has_contradiction"] is True
