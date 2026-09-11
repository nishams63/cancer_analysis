"""Tests for Prompt Drift Rules and Compliance Evaluator."""
import os
import pytest
from src.prompts.drift_rules import (
    PromptDriftEvaluator, get_default_drift_rules_catalog, export_drift_rules_yaml
)
from src.prompts.scenario_builder import get_core_scenarios


def test_evaluator_compliant_content():
    scenarios = get_core_scenarios()
    sc01 = [s for s in scenarios if s.scenario_id == "PROMPT-R01"][0]
    evaluator = PromptDriftEvaluator(sc01)

    good_text = (
        "Patient with metastatic lung adenocarcinoma has progressive disease on Osimertinib 80mg daily. "
        "Genomic re-biopsy confirms persistent EGFR T790M mutation along with new secondary MET Amplification "
        "with copy number >= 5."
    )
    res = evaluator.evaluate(good_text)
    assert res.is_compliant is True
    assert res.compliance_score >= 0.85
    assert len(res.forbidden_rule_violations) == 0


def test_evaluator_forbidden_content():
    scenarios = get_core_scenarios()
    sc01 = [s for s in scenarios if s.scenario_id == "PROMPT-R01"][0]
    evaluator = PromptDriftEvaluator(sc01)

    bad_text = (
        "Patient has EGFR T790M mutation. Biopsy reveals MET wild-type status with no secondary alteration."
    )
    res = evaluator.evaluate(bad_text)
    assert res.is_compliant is False
    assert len(res.forbidden_rule_violations) > 0
    assert res.compliance_score <= 0.40


def test_evaluator_range_violations():
    scenarios = get_core_scenarios()
    sc05 = [s for s in scenarios if s.scenario_id == "PROMPT-R05"][0]
    evaluator = PromptDriftEvaluator(sc05)

    # Creatinine out of range (min 3.2, max 4.2)
    extracted = {"serum_creatinine": 1.2, "planned_therapy": "Cisplatin"}
    res = evaluator.evaluate("Patient with normal labs", extracted_entities=extracted)
    assert len(res.range_violations) > 0


def test_drift_catalog_export(tmp_path):
    out_file = str(tmp_path / "test_drift.yaml")
    export_drift_rules_yaml(out_file)
    assert os.path.exists(out_file)