"""
Unit tests for OOD Benchmarking and Performance Degradation.
"""

import pandas as pd
import pytest
from ood_evaluator import OODEvaluator


def test_ood_evaluator_metrics_and_degradation():
    evaluator = OODEvaluator()

    df_std = pd.DataFrame([
        {"prompt": "Patient on Osimertinib 80mg.", "target": "Risk: Low\nKey Finding: Patient on Osimertinib.\nAction: Continue monitoring.", "expected_risk": "Low", "ner_drugs": ["Osimertinib"]},
        {"prompt": "Patient on Docetaxel 75mg developed neutropenia.", "target": "Risk: High\nKey Finding: Developed neutropenia.\nAction: Hold Docetaxel.", "expected_risk": "High", "ner_drugs": ["Docetaxel"]}
    ])
    preds_std = [
        "Risk: Low\nKey Finding: Patient on Osimertinib.\nAction: Continue monitoring.",
        "Risk: High\nKey Finding: Developed neutropenia.\nAction: Hold Docetaxel."
    ]
    std_metrics = evaluator.evaluate_cohort("Standard", "STANDARD", df_std, preds_std)
    assert std_metrics["risk_accuracy"] == 1.0
    assert std_metrics["format_compliance_rate"] == 1.0

    df_ood = pd.DataFrame([
        {"prompt": "Novel sarcoma note on Trabectedin 1.5mg.", "target": "Risk: Low\nKey Finding: Trabectedin therapy.\nAction: Continue.", "expected_risk": "Low", "ner_drugs": ["Trabectedin"]},
        {"prompt": "Glioblastoma note on Temozolomide.", "target": "Risk: High\nKey Finding: High risk.\nAction: Hold.", "expected_risk": "High", "ner_drugs": ["Temozolomide"]}
    ])
    # OOD with slight format flaw on 1 sample
    preds_ood = [
        "Risk: Low\nKey Finding: Trabectedin therapy.\nAction: Continue.",
        "Observation: Patient has glioblastoma and high toxicity risk." # Missing fields
    ]
    ood_metrics = evaluator.evaluate_cohort("OOD", "OOD-REAL", df_ood, preds_ood)
    assert ood_metrics["format_compliance_rate"] == 0.5
    assert ood_metrics["risk_accuracy"] == 0.5

    deg = evaluator.compute_degradation(std_metrics, ood_metrics)
    assert deg["delta_risk_accuracy"] == 0.5
    assert deg["delta_format_compliance"] == 0.5
