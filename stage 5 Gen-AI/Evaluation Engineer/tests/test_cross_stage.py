"""Tests for Cross-Stage Stress Test Evaluator."""
import pytest
from src.evaluation.cross_stage import CrossStageEvaluator


def test_cross_stage_evaluation(sample_valid_patient, sample_scenario_def):
    evaluator = CrossStageEvaluator()
    narrative = {"narrative_text": "Patient with EGFR and MET amplification."}
    res = evaluator.evaluate_scenario(sample_scenario_def, sample_valid_patient, narrative)

    assert "stage1" in res["stage_results"]
    assert "stage2" in res["stage_results"]
    assert "stage3" in res["stage_results"]
    assert "stage4" in res["stage_results"]
    assert len(res["failed_stages"]) >= 1
    assert res["has_cross_stage_disagreement"] is True


def test_cross_stage_batch_summary(sample_valid_patient, sample_scenario_def):
    evaluator = CrossStageEvaluator()
    sc_res = evaluator.evaluate_scenario(sample_scenario_def, sample_valid_patient)
    summary = evaluator.summarize_batch([sc_res])
    assert summary["total_scenarios"] == 1
    assert "stage_failure_rates" in summary
