"""Tests for Counterfactual Evaluator."""
import pytest
from src.evaluation.counterfactual import CounterfactualEvaluator
from src.adapters.stage4_eval_adapter import Stage4EvalAdapter


def test_counterfactual_sensitivity_detected(sample_valid_patient):
    evaluator = CounterfactualEvaluator()
    adapter = Stage4EvalAdapter()

    factual = dict(sample_valid_patient)
    factual["mutations"] = ["EGFR", "MET"] # Dual driver bypass

    cf = dict(sample_valid_patient)
    cf["mutations"] = ["EGFR"] # Removed MET bypass

    cf_entry = {
        "scenario_id": "PROMPT-R01",
        "target_variable": "mutations",
        "factual_patient": factual,
        "counterfactual_patient": cf
    }

    res = evaluator.evaluate_pair(cf_entry, adapter)
    assert res["immutability_preserved"] is True
    assert res["sensitivity_detected"] is True
    assert res["status"] == "PASS"
