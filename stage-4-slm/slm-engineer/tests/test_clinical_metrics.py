"""
Tests for Clinical, Structural, and Safety Evaluation Metrics Module.
"""

from clinical_metrics import ClinicalEvaluator


def test_clinical_evaluator_metrics():
    evaluator = ClinicalEvaluator()

    predictions = [
        "Risk: Low\nKey Finding: EGFR variant receiving Osimertinib at 80 mg.\nAction: Continue monitoring.",
        "Risk: High\nKey Finding: Patient developed neutropenia.\nAction: Hold Docetaxel.",
        # Corrupted / negation flip
        "Risk: High\nKey Finding: Increased acute adverse toxicities hazard.\nAction: Hold therapy."
    ]

    targets = [
        "Risk: Low\nKey Finding: EGFR variant receiving Osimertinib at 80 mg.\nAction: Continue monitoring.",
        "Risk: High\nKey Finding: Patient developed neutropenia.\nAction: Hold Docetaxel.",
        "Risk: Low\nKey Finding: Exhibits stable tolerance with no acute toxicities.\nAction: Continue monitoring."
    ]

    expected_risks = ["Low", "High", "Low"]
    clinical_notes = [
        "Patient with EGFR receiving Osimertinib 80 mg.",
        "Patient developed neutropenia.",
        "Patient demonstrates no acute adverse toxicities."
    ]

    ref_entities = [
        {"ner_genes": ["EGFR"], "ner_drugs": ["Osimertinib"], "ner_dosages": ["80 mg"], "ner_adverse_events": []},
        {"ner_genes": [], "ner_drugs": ["Docetaxel"], "ner_dosages": [], "ner_adverse_events": ["neutropenia"]},
        {"ner_genes": [], "ner_drugs": [], "ner_dosages": [], "ner_adverse_events": ["no acute adverse toxicities"]}
    ]

    metrics = evaluator.evaluate_batch(
        predictions=predictions,
        targets=targets,
        expected_risks=expected_risks,
        clinical_notes=clinical_notes,
        reference_entities_list=ref_entities
    )

    assert metrics["format_compliance_rate"] == 1.0
    assert abs(metrics["risk_accuracy"] - 2 / 3) < 1e-3
    assert metrics["risk_macro_f1"] > 0.0
    assert metrics["entity_retention_rate"] >= 0.75
    # Third case triggered negation flip
    assert metrics["negation_cases_evaluated"] == 1
    assert metrics["negation_flips"] == 1
    assert metrics["negation_flip_rate"] == 1.0
