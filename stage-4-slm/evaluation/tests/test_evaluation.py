"""
Unit Tests for Stage 4 Evaluation Module.
Tests NLG metrics calculation, entity preservation F1, hallucination detection,
bootstrap confidence intervals, locked-test evaluation, and subgroup audits.
"""

import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import sys

EVAL_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(EVAL_SRC_DIR))

from metrics import ClinicalMetricsCalculator
from evaluator import MasterSLMEvaluator
from subgroup_audit import SubgroupAuditor


@pytest.fixture
def mock_test_dataset(tmp_path):
    data = [
        {
            "patient_id": "PT-001",
            "note_id": "DOC-001",
            "clinical_note": "62-year-old male with NSCLC driver mutation in EGFR. Received Erlotinib 150 mg. No acute toxicities.",
            "instruction": "Analyze the clinical note and provide the patient's Risk, Key Finding, and Action.",
            "target_risk": "Low baseline toxicity risk on Erlotinib 150 mg therapy.",
            "target_key_finding": "EGFR driver alteration confirmed; patient received Erlotinib at 150 mg; developed no acute adverse toxicities.",
            "target_action": "Continue standard clinical monitoring and maintain current Erlotinib regimen as tolerated.",
            "ner_genes": ["EGFR"],
            "ner_drugs": ["Erlotinib"],
            "ner_dosages": ["150 mg"],
            "split": "TEST"
        },
        {
            "patient_id": "PT-002",
            "note_id": "DOC-002",
            "clinical_note": "45-year-old female with breast cancer TP53 mutation receiving Docetaxel 100 mg. Severe acute dyspnea and grade 4 hypoxia.",
            "instruction": "Analyze the clinical note and provide the patient's Risk, Key Finding, and Action.",
            "target_risk": "Critical acute pulmonary toxicity risk and systemic compromise on Docetaxel 100 mg.",
            "target_key_finding": "TP53 driver alteration confirmed; patient received Docetaxel at 100 mg; exhibits severe acute pulmonary toxicity.",
            "target_action": "Immediately hold Docetaxel, transfer to intensive care, and initiate urgent supportive resuscitation.",
            "ner_genes": ["TP53"],
            "ner_drugs": ["Docetaxel"],
            "ner_dosages": ["100 mg"],
            "split": "TEST"
        }
    ]
    df = pd.DataFrame(data)
    file_path = tmp_path / "mock_test.parquet"
    df.to_parquet(file_path)
    return str(file_path)


def test_nlg_metrics_computation():
    """Verify ROUGE and BLEU metric computations."""
    calc = ClinicalMetricsCalculator()
    preds = ["Administer Erlotinib 150 mg daily for EGFR mutated NSCLC."]
    refs = ["Administer Erlotinib 150 mg daily for EGFR mutated NSCLC."]
    metrics = calc.compute_nlg_metrics(preds, refs)

    assert metrics["rouge1"] > 0.90
    assert metrics["rouge2"] > 0.90
    assert metrics["rougeL"] > 0.90
    assert metrics["bleu4"] > 0.50


def test_extract_entities_from_text():
    """Verify extraction of clinical entities from text."""
    text = "Patient with confirmed EGFR mutation prescribed Erlotinib at 150 mg dosage."
    ents = ClinicalMetricsCalculator.extract_entities_from_text(text)

    assert "EGFR" in ents["genes"]
    assert "Erlotinib" in ents["drugs"]
    assert "150mg" in ents["dosages"]


def test_entity_preservation_and_hallucination():
    """Verify entity preservation metrics and hallucination detection."""
    calc = ClinicalMetricsCalculator()
    triads = [
        {
            "target_risk": "Risk on Erlotinib 150 mg.",
            "target_key_finding": "EGFR confirmed; Erlotinib at 150 mg.",
            "target_action": "Continue Erlotinib."
        }
    ]
    ref_ents = [{"ner_drugs": ["Erlotinib"], "ner_genes": ["EGFR"], "ner_dosages": ["150 mg"]}]
    source_notes = ["Patient with EGFR mutation received Erlotinib 150 mg."]

    res = calc.compute_entity_preservation(triads, ref_ents, source_notes)
    assert res["mean_entity_f1"] > 0.80
    assert res["hallucination_rate"] == 0.0

    # Test with hallucinated drug
    triads_hallucinated = [
        {
            "target_risk": "Risk on Cisplatin 100 mg.",
            "target_key_finding": "Patient received Cisplatin.",
            "target_action": "Hold Cisplatin."
        }
    ]
    res_hal = calc.compute_entity_preservation(triads_hallucinated, ref_ents, source_notes)
    assert res_hal["hallucination_rate"] > 0.0


def test_bootstrap_confidence_interval():
    """Verify bootstrap confidence interval calculation."""
    scores = [0.80, 0.85, 0.90, 0.82, 0.88, 0.84, 0.86]
    mean_val, low, high = ClinicalMetricsCalculator.compute_bootstrap_ci(scores, n_bootstraps=200)

    assert low <= mean_val <= high
    assert 0.75 <= low <= 0.95


def test_evaluator_execution(mock_test_dataset, tmp_path):
    """Verify MasterSLMEvaluator execution on test dataset."""
    out_dir = tmp_path / "eval_out"
    evaluator = MasterSLMEvaluator(dataset_path=mock_test_dataset, output_dir=str(out_dir))
    summary = evaluator.run_benchmark()

    assert summary["test_set_size"] == 2
    assert "comparative_stage3_benchmark" in summary
    assert summary["comparative_stage3_benchmark"]["certification_status"] == "EXCEEDS_BASELINE"
    assert (out_dir / "results" / "evaluation_metrics.json").exists()
    assert (out_dir / "reports" / "evaluation_report.md").exists()


def test_subgroup_audit(mock_test_dataset, tmp_path):
    """Verify SubgroupAuditor runs across clinical strata."""
    out_dir = tmp_path / "eval_out"
    auditor = SubgroupAuditor(dataset_path=mock_test_dataset, output_dir=str(out_dir))
    audit_results = auditor.run_subgroup_audit()

    assert len(audit_results) >= 20
    assert "Cancer Type: NSCLC / Lung" in audit_results
    assert (out_dir / "reports" / "subgroup_audit.md").exists()
