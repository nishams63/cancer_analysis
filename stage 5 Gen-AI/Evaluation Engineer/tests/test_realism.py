"""Tests for Structured Scenario Realism and Plausibility."""
import pytest
from src.evaluation.realism import ScenarioRealismEvaluator
from src.evaluation.scenario_plausibility import ScenarioPlausibilityEvaluator


def test_valid_patient_realism(sample_valid_patient):
    evaluator = ScenarioRealismEvaluator()
    res = evaluator.evaluate_patient(sample_valid_patient)
    assert res["is_physiologically_plausible"] is True
    assert res["checks"]["age"] == "PASS"
    assert res["checks"]["mutations"] == "PASS"


def test_out_of_range_patient_fails():
    evaluator = ScenarioRealismEvaluator()
    bad_pt = {
        "scenario_id": "TEST-BAD",
        "demographics": {"age": 140, "sex": "Male"}, # impossible age
        "mutations": [],
        "biomarkers": {"creatinine_level": 9999.0}
    }
    res = evaluator.evaluate_patient(bad_pt)
    assert res["is_physiologically_plausible"] is False
    assert res["checks"]["age"] == "FAIL"


def test_rare_but_allowed_scenario_plausibility(sample_valid_patient, sample_scenario_def):
    plaus_eval = ScenarioPlausibilityEvaluator()
    res = plaus_eval.evaluate_scenario("PROMPT-R01", sample_valid_patient, sample_scenario_def)
    assert res["scenario_plausibility_score"] >= 0.75
    assert res["individual_plausibility"] is True
    assert res["status"] == "PASS"


def test_batch_population_similarity(sample_valid_patient):
    evaluator = ScenarioRealismEvaluator()
    batch_res = evaluator.evaluate_batch([sample_valid_patient] * 5)
    assert 0.0 <= batch_res["population_similarity_score"] <= 1.0
    assert batch_res["batch_size"] == 5
