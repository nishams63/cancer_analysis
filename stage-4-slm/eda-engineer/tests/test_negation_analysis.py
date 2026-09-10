"""
Tests for Negation Scope Analysis and Negation Flip Detection.
"""

import pandas as pd
import pytest
from negation_analysis import NegationAnalyzer


def test_negation_detection_basic():
    analyzer = NegationAnalyzer()
    assert analyzer.is_concept_negated_in_sentence("Patient shows no evidence of pneumonitis.", "pneumonitis")
    assert analyzer.is_concept_negated_in_sentence("Denies fatigue or nausea.", "fatigue")
    assert not analyzer.is_concept_negated_in_sentence("Patient developed grade 3 pneumonitis.", "pneumonitis")


def test_pseudo_negation_handling():
    analyzer = NegationAnalyzer()
    # Pseudo negation should NOT be classified as negation
    assert not analyzer.is_concept_negated_in_sentence("CT scan showed no change in tumor size.", "tumor size")


def test_negation_flip_detection():
    analyzer = NegationAnalyzer()
    df = pd.DataFrame({
        "patient_id": ["PT-001", "PT-002"],
        "note_id": ["DOC-001", "DOC-002"],
        "clinical_note": [
            "Cycle 1. Patient shows no evidence of nephrotoxicity.",
            "Cycle 1. Patient shows no evidence of nephrotoxicity."
        ],
        "target_risk": [
            "Increased nephrotoxicity and renal hazard associated with Cisplatin.", # FLIP!
            "Baseline risk with no acute nephrotoxicity."                          # PRESERVED
        ],
        "target_key_finding": ["Cisplatin administered.", "Cisplatin administered."],
        "target_action": ["Hold Cisplatin.", "Continue monitoring."],
        "ner_adverse_events": [["nephrotoxicity"], ["nephrotoxicity"]]
    })

    metrics, review_df = analyzer.audit_negation_in_dataset(df)
    assert metrics["total_negated_entities_detected"] == 2
    assert metrics["negation_flips"] == 1
    assert metrics["correctly_preserved"] == 1
    assert len(review_df) == 1
    assert review_df.iloc[0]["detected_issue"] == "NEGATION_FLIP_AFFIRMED_HAZARD"
