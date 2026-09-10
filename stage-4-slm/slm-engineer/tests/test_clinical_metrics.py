"""
Tests verifying clinical metric evaluation functions:
Macro-F1, Entity Retention, Hallucination Rate, Negation Preservation, and Composite Score.
"""

from src.evaluate import ClinicalEvaluator


def test_risk_classification_metrics():
    """Verifies accuracy and macro-F1 calculations."""
    preds = ["Low", "Moderate", "High", "Low"]
    refs = ["Low", "Moderate", "High", "High"]
    res = ClinicalEvaluator.evaluate_risk_classification(preds, refs)
    assert res["risk_accuracy"] == 0.75
    assert 0.0 < res["risk_macro_f1"] <= 1.0


def test_entity_retention_metrics():
    """Verifies entity retention matching against reference entities."""
    texts = [
        "Patient received Osimertinib 80mg with EGFR mutation.",
        "Patient experienced mild rash on docetaxel."
    ]
    ref_ents = [
        {"ner_genes": ["EGFR"], "ner_drugs": ["osimertinib"], "ner_dosages": ["80mg"]},
        {"ner_genes": [], "ner_drugs": ["docetaxel"], "ner_adverse_events": ["rash"]}
    ]
    res = ClinicalEvaluator.evaluate_entity_retention(texts, ref_ents)
    assert res["total_reference_entities"] == 5
    assert res["retained_entities"] == 5
    assert res["entity_retention_rate"] == 1.0


def test_negation_preservation_metrics():
    """Verifies detection of critical negation flips."""
    # Case 1: Negation preserved
    notes_preserved = ["Patient denies toxicities, tolerating well."]
    gens_preserved = ["Risk: Low\nKey Finding: Tolerating well with no acute toxicities.\nAction: Continue."]
    res_pres = ClinicalEvaluator.evaluate_negation_preservation(gens_preserved, notes_preserved)
    assert res_pres["negation_flips"] == 0
    assert res_pres["negation_preservation_rate"] == 1.0

    # Case 2: Negation flipped (note says denies toxicities, but output claims High risk)
    gens_flipped = ["Risk: High\nKey Finding: Severe treatment-related toxicities.\nAction: Hold drug."]
    res_flip = ClinicalEvaluator.evaluate_negation_preservation(gens_flipped, notes_preserved)
    assert res_flip["negation_flips"] == 1
    assert res_flip["negation_flip_rate"] == 1.0


def test_composite_selection_score():
    """Verifies 5-component composite scoring formula."""
    metrics = {
        "risk_macro_f1": 0.95,
        "entity_retention_rate": 0.98,
        "negation_preservation_rate": 1.0,
        "format_compliance_rate": 1.0,
        "hallucination_rate": 0.02
    }
    # 0.30*0.95 + 0.25*0.98 + 0.25*1.0 + 0.10*1.0 - 0.10*0.02
    # 0.285 + 0.245 + 0.250 + 0.100 - 0.002 = 0.878
    score = ClinicalEvaluator.compute_composite_selection_score(metrics)
    assert score == 0.878
