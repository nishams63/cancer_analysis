"""Tests for Scenario Fidelity Evaluator."""
import pytest
from src.evaluation.fidelity import ScenarioFidelityEvaluator


def test_fidelity_all_satisfied(sample_valid_patient, sample_scenario_def):
    evaluator = ScenarioFidelityEvaluator()
    res = evaluator.evaluate_fidelity(sample_scenario_def, sample_valid_patient)
    assert res["fidelity_score"] >= 0.85
    assert len(res["forbidden_violations"]) == 0
    assert res["status"] == "PASS"


def test_fidelity_forbidden_condition_causes_fail(sample_valid_patient, sample_scenario_def):
    evaluator = ScenarioFidelityEvaluator()
    bad_pt = dict(sample_valid_patient)
    # Inject forbidden text: "MET negative"
    bad_pt["clinical_notes"] = "Biopsy confirmed MET negative status."
    res = evaluator.evaluate_fidelity(sample_scenario_def, bad_pt)
    assert "FORBID-01" in res["forbidden_violations"]
    assert res["status"] == "FAIL"


def test_fidelity_missing_required_mutation_drops_score(sample_valid_patient, sample_scenario_def):
    evaluator = ScenarioFidelityEvaluator()
    bad_pt = dict(sample_valid_patient)
    bad_pt["mutations"] = ["KRAS G12D"] # Neither EGFR nor MET
    res = evaluator.evaluate_fidelity(sample_scenario_def, bad_pt)
    assert res["fidelity_score"] < 0.70
