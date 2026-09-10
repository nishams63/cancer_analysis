"""
Unit tests for Post-Inference Clinical Safety Firewall.
"""

import pytest
from safety_firewall import ClinicalSafetyFirewall


def test_firewall_valid_generation_passes():
    firewall = ClinicalSafetyFirewall(strict_mode=True, min_confidence_threshold=0.70)
    source_note = "Patient with EGFR exon 19 receiving Osimertinib 80 mg daily. Denies adverse toxicities. Routine visit."
    generation = "Risk: Low\nKey Finding: EGFR variant on Osimertinib at 80 mg with stable tolerance.\nAction: Continue standard clinical monitoring."

    result = firewall.validate(source_note, generation, confidence=0.92)
    assert result.passed is True
    assert result.route == "CLINICAL_UI"
    assert len(result.infractions) == 0
    assert result.parsed_fields["Risk"] == "Low"


def test_firewall_schema_failure():
    firewall = ClinicalSafetyFirewall()
    source_note = "Patient on Cisplatin 75 mg."
    # Missing 'Action:' field
    bad_generation = "Risk: High\nKey Finding: Patient developed neutropenia."

    result = firewall.validate(source_note, bad_generation, confidence=0.95)
    assert result.passed is False
    assert result.route == "HUMAN_REVIEW"
    assert any("SCHEMA_ERROR" in inf for inf in result.infractions)


def test_firewall_negation_flip_rejection():
    firewall = ClinicalSafetyFirewall()
    source_note = "Patient demonstrates no acute adverse toxicities on Pembrolizumab 200 mg."
    # Model incorrectly asserts neutropenia
    flipped_gen = "Risk: High\nKey Finding: Patient developed neutropenia.\nAction: Hold therapy."

    result = firewall.validate(source_note, flipped_gen, confidence=0.88)
    assert result.passed is False
    assert result.route == "HUMAN_REVIEW"
    assert any("NEGATION_FLIP" in inf for inf in result.infractions)


def test_firewall_hallucinated_drug_rejection():
    firewall = ClinicalSafetyFirewall()
    source_note = "Patient receiving Osimertinib for lung cancer."
    # Model hallucinates Doxorubicin
    halluc_gen = "Risk: High\nKey Finding: Patient on Doxorubicin.\nAction: Hold anthracycline therapy."

    result = firewall.validate(source_note, halluc_gen, confidence=0.85)
    assert result.passed is False
    assert result.route == "HUMAN_REVIEW"
    assert any("HALLUCINATED_DRUG" in inf for inf in result.infractions)


def test_firewall_action_incoherence_rejection():
    firewall = ClinicalSafetyFirewall()
    source_note = "Patient has severe neutropenia."
    # High risk with passive monitoring action
    incoherent_gen = "Risk: High\nKey Finding: Severe neutropenia observed.\nAction: Continue standard clinical monitoring."

    result = firewall.validate(source_note, incoherent_gen, confidence=0.90)
    assert result.passed is False
    assert result.route == "HUMAN_REVIEW"
    assert any("ACTION_INCOHERENCE" in inf for inf in result.infractions)


def test_firewall_low_confidence_routing():
    firewall = ClinicalSafetyFirewall(min_confidence_threshold=0.80)
    source_note = "Patient on Docetaxel."
    valid_gen = "Risk: Low\nKey Finding: Patient on Docetaxel.\nAction: Continue monitoring."

    # Low confidence should route to human review even if text is valid
    result = firewall.validate(source_note, valid_gen, confidence=0.65)
    assert result.passed is False
    assert result.route == "HUMAN_REVIEW"
    assert any("LOW_CONFIDENCE" in inf for inf in result.infractions)
