"""
Unit Tests for Section 6 Entity-Preservation Quality Gate.
Tests PASS, REJECT, missing entity detection, invented entity detection,
and coverage calculations.
"""

import pytest
from pathlib import Path
import sys

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
CONFIG_DIR = TESTS_DIR.parent / "config"
sys.path.insert(0, str(SRC_DIR))

from entity_validator import EntityPreservationQualityGate, EntityNormalizer


@pytest.fixture
def quality_gate():
    synonyms_path = CONFIG_DIR / "drug_synonyms.yaml"
    normalizer = EntityNormalizer(drug_synonyms_path=str(synonyms_path))
    return EntityPreservationQualityGate(normalizer=normalizer)


def test_correct_entity_preservation_passes(quality_gate):
    """Test that complete entity preservation across all 4 categories passes."""
    note = "Patient was prescribed Warfarin 5 mg and subsequently developed bleeding. Genetic testing identified a CYP2C9 variant."
    t_risk = "Increased bleeding risk associated with Warfarin 5 mg."
    t_kf = "CYP2C9 variation was identified and bleeding occurred after Warfarin treatment."
    t_act = "Monitor bleeding and review Warfarin therapy."

    res = quality_gate.validate_example(
        source_note=note,
        target_risk=t_risk,
        target_key_finding=t_kf,
        target_action=t_act,
        ner_genes=["CYP2C9"],
        ner_drugs=["Warfarin"],
        ner_dosages=["5 mg"],
        ner_adverse_events=["bleeding"]
    )

    assert res["entity_check_status"] == "PASS"
    assert res["entity_coverage"] == 1.0
    assert len(res["missing_genes"]) == 0
    assert len(res["missing_drugs"]) == 0
    assert len(res["missing_dosages"]) == 0
    assert len(res["missing_adverse_events"]) == 0
    assert len(res["invented_entities"]) == 0


def test_missing_drug_rejection(quality_gate):
    """Test that omitting a prescribed drug leads to rejection."""
    note = "Patient was prescribed Warfarin 5 mg and subsequently developed bleeding."
    # Target omits Warfarin
    t_risk = "Increased bleeding risk associated with antineoplastic therapy."
    t_kf = "Bleeding occurred after medical intervention."
    t_act = "Monitor bleeding."

    res = quality_gate.validate_example(
        source_note=note,
        target_risk=t_risk,
        target_key_finding=t_kf,
        target_action=t_act,
        ner_genes=[],
        ner_drugs=["Warfarin"],
        ner_dosages=["5 mg"],
        ner_adverse_events=["bleeding"]
    )

    assert res["entity_check_status"] == "REJECT"
    assert "Warfarin" in res["missing_drugs"]
    assert res["entity_coverage"] < 1.0


def test_invented_drug_rejection(quality_gate):
    """Test that hallucinating an unprescribed drug leads to rejection."""
    note = "Patient was prescribed Warfarin 5 mg and developed bleeding."
    # Target hallucinates Cisplatin
    t_risk = "Increased bleeding risk associated with Warfarin 5 mg."
    t_kf = "Bleeding occurred after Warfarin and Cisplatin treatment."
    t_act = "Hold Cisplatin and monitor Warfarin."

    res = quality_gate.validate_example(
        source_note=note,
        target_risk=t_risk,
        target_key_finding=t_kf,
        target_action=t_act,
        ner_genes=[],
        ner_drugs=["Warfarin"],
        ner_dosages=["5 mg"],
        ner_adverse_events=["bleeding"]
    )

    assert res["entity_check_status"] == "REJECT"
    assert any("cisplatin" in inv for inv in res["invented_entities"])


def test_synonym_preservation_passes(quality_gate):
    """Test that using an approved brand synonym (Coumadin for Warfarin) preserves entity."""
    note = "Patient was prescribed Warfarin 5 mg."
    t_risk = "Bleeding risk on Coumadin 5 mg."
    t_kf = "Patient received Coumadin 5mg."
    t_act = "Review Coumadin."

    res = quality_gate.validate_example(
        source_note=note,
        target_risk=t_risk,
        target_key_finding=t_kf,
        target_action=t_act,
        ner_genes=[],
        ner_drugs=["Warfarin"],
        ner_dosages=["5 mg"],
        ner_adverse_events=[]
    )

    assert res["entity_check_status"] == "PASS"
    assert len(res["missing_drugs"]) == 0
