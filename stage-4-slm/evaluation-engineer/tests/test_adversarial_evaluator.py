"""
Unit tests for Adversarial Clinical Testing and Negation Robustness.
"""

import pandas as pd
import pytest
from adversarial_evaluator import AdversarialEvaluator


def test_adversarial_evaluator_clean_pass():
    evaluator = AdversarialEvaluator(threshold=0.01)

    df = pd.DataFrame([
        {
            "prompt": "Patient denies adverse toxicities on Osimertinib 80mg.",
            "target": "Risk: Low\nKey Finding: Patient on Osimertinib.\nAction: Continue monitoring.",
            "expected_risk": "Low",
            "ner_drugs": ["Osimertinib"],
            "sub_category": "negation_paraphrase"
        },
        {
            "prompt": "Patient evaluated on osimertanib 80mg.",
            "target": "Risk: Low\nKey Finding: Patient on Osimertinib.\nAction: Continue monitoring.",
            "expected_risk": "Low",
            "ner_drugs": ["Osimertinib"],
            "sub_category": "drug_typo"
        }
    ])

    predictions = [
        "Risk: Low\nKey Finding: Patient on Osimertinib.\nAction: Continue monitoring.",
        "Risk: Low\nKey Finding: Patient on Osimertinib.\nAction: Continue monitoring."
    ]

    results = evaluator.evaluate_adversarial_suite(df, predictions)
    assert results["overall_risk_accuracy"] == 1.0
    assert results["overall_negation_flip_rate"] == 0.0
    assert results["safety_gate_passed"] is True
    assert "negation_paraphrase" in results["category_breakdown"]


def test_adversarial_evaluator_negation_flip_triggers_failure():
    evaluator = AdversarialEvaluator(threshold=0.01)

    df = pd.DataFrame([
        {
            "prompt": "Patient denies adverse toxicities on Pembrolizumab 200mg.",
            "target": "Risk: Low\nKey Finding: Patient on Pembrolizumab.\nAction: Continue monitoring.",
            "expected_risk": "Low",
            "ner_drugs": ["Pembrolizumab"],
            "sub_category": "negation_paraphrase"
        }
    ])

    # Flipped prediction: asserts neutropenia
    predictions = [
        "Risk: High\nKey Finding: Patient developed neutropenia.\nAction: Hold therapy."
    ]

    results = evaluator.evaluate_adversarial_suite(df, predictions)
    assert results["total_negation_flips"] == 1
    assert results["overall_negation_flip_rate"] == 1.0
    assert results["safety_gate_passed"] is False
