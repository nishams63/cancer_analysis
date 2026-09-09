"""
Unit and Integration Tests for Stage 3 Clinical NLP Data Augmentation Pipeline.
Verifies entity-span preservation, negation invariance, 10-point QC filter,
duplicate detection, and strict partition leakage prevention.
"""

import sys
from pathlib import Path
import random
import json
import pytest
import pandas as pd

# Add src to sys.path
TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from annotation_propagation import propagate_entity_spans
from terminology_augmentation import apply_terminology_augmentation
from sentence_augmentation import permute_vitals_clauses, permute_lab_chemistry_clauses
from context_augmentation import apply_context_framing_augmentation
from entity_preserving_augmentation import verify_entity_spans, augment_document_preserving_entities
from duplicate_detection import compute_sha256, compute_canonical_hash, compute_jaccard_similarity, DuplicateRegistry
from quality_control import QualityControlGate
from leakage_checks import verify_patient_split_isolation


def test_annotation_propagation_basic():
    """Verify offset projection when text outside entity spans is modified."""
    text = "Vitals: Blood pressure 120/80 mmHg. Administer Pembrolizumab at 200 mg IV."
    entities = [
        {"start": 47, "end": 60, "label": "DRUG_NAME", "text": "Pembrolizumab"},
        {"start": 64, "end": 70, "label": "DOSAGE", "text": "200 mg"}
    ]
    # Replace "Vitals: Blood pressure 120/80 mmHg." (0 to 35) with "Intake Vitals: BP 120/80 mmHg." (len 30)
    # Delta is -5 characters
    replacements = [(0, 35, "Intake Vitals: BP 120/80 mmHg.")]
    new_text, new_ents, valid = propagate_entity_spans(text, entities, replacements)

    assert valid is True
    assert len(new_ents) == 2
    for ent in new_ents:
        s, e, t = ent["start"], ent["end"], ent["text"]
        assert new_text[s:e] == t
    assert new_ents[0]["text"] == "Pembrolizumab"
    assert new_ents[1]["text"] == "200 mg"


def test_annotation_propagation_collision_rejected():
    """Verify that any replacement overlapping an entity span is strictly rejected."""
    text = "Patient prescribed Pembrolizumab for therapy."
    entities = [{"start": 19, "end": 32, "label": "DRUG_NAME", "text": "Pembrolizumab"}]
    # Attempt replacement that overlaps with "Pembrolizumab"
    replacements = [(10, 25, "administered")]
    new_text, new_ents, valid = propagate_entity_spans(text, entities, replacements)
    assert valid is False


def test_verify_entity_spans_validator():
    """Verify span validator catches text mismatches and overlaps."""
    text = "Confirmed EGFR mutation."
    valid_ents = [{"start": 10, "end": 14, "label": "GENE_MUTATION", "text": "EGFR"}]
    is_valid, _ = verify_entity_spans(text, valid_ents)
    assert is_valid is True

    # Corrupted text
    corrupt_ents = [{"start": 10, "end": 14, "label": "GENE_MUTATION", "text": "KRAS"}]
    is_valid, err = verify_entity_spans(text, corrupt_ents)
    assert is_valid is False
    assert "text_mismatch" in err


def test_terminology_augmentation_preserves_entities():
    """Verify carrier phrase substitution preserves exact entities."""
    text = (
        "ONCOLOGY CONSULTATION PROGRESS NOTE\n"
        "CLINICAL ASSESSMENT & LABORATORY REVIEW:\n"
        "Physical examination reveals clear lung fields bilaterally.\n"
        "Administer scheduled therapy with carboplatin at a dosage of 450.0 mg.\n"
        "Pre-medications ordered per protocol."
    )
    entities = [
        {"start": text.find("carboplatin"), "end": text.find("carboplatin") + len("carboplatin"), "label": "DRUG_NAME", "text": "carboplatin"},
        {"start": text.find("450.0 mg"), "end": text.find("450.0 mg") + len("450.0 mg"), "label": "DOSAGE", "text": "450.0 mg"}
    ]
    rng = random.Random(42)
    new_text, new_ents, success, desc = apply_terminology_augmentation(text, entities, rng)

    assert success is True
    assert len(new_ents) == len(entities)
    for ent in new_ents:
        s, e, t = ent["start"], ent["end"], ent["text"]
        assert new_text[s:e] == t


def test_sentence_augmentation_vitals():
    """Verify vitals permutation reorders measurements without affecting entities."""
    text = "Vitals: Blood pressure 130/80 mmHg, Heart rate 72 bpm, SpO2 99% on room air."
    entities = []
    rng = random.Random(42)
    new_text, new_ents, success, desc = permute_vitals_clauses(text, entities, rng)

    assert success is True
    assert "Heart rate" in new_text
    assert "Blood pressure" in new_text
    assert "SpO2" in new_text


def test_duplicate_registry():
    """Verify exact and canonical duplicate detection."""
    reg = DuplicateRegistry()
    text = "Patient evaluated for cycle 2."
    reg.register(text)

    # Exact duplicate
    is_dup, reason = reg.is_duplicate(text)
    assert is_dup is True
    assert reason == "exact_duplicate"

    # Whitespace/case canonical duplicate
    is_dup, reason = reg.is_duplicate("  Patient   evaluated for   Cycle 2.  ")
    assert is_dup is True
    assert reason == "canonical_duplicate"

    # Unique text
    is_dup, reason = reg.is_duplicate("Completely novel clinical statement.")
    assert is_dup is False


def test_quality_control_gate():
    """Verify 10-point QC gate checks."""
    qc = QualityControlGate(min_words=5, max_words=50, min_chars=20, max_chars=300)
    source_row = {
        "document_id": "DOC-001",
        "patient_id": "PT-001",
        "text": "Baseline clinical note with no acute toxicities and pulse 70 bpm.",
        "urgency_level": "LOW",
        "hazard_type": "NONE"
    }
    train_patients = {"PT-001"}

    # Pass case
    cand_text = "Baseline clinical note with no acute toxicities and heart rate 70 bpm."
    cand_ents = []
    passed, msg = qc.evaluate(cand_text, cand_ents, source_row, "synonym", train_patients)
    assert passed is True

    # Fail check 5: Negation dropped
    no_neg_text = "Baseline clinical note with acute toxicities and pulse 70 bpm."
    passed, msg = qc.evaluate(no_neg_text, cand_ents, source_row, "synonym", train_patients)
    assert passed is False
    assert "Check 5 Failed" in msg

    # Fail check 10: Non-train patient
    passed, msg = qc.evaluate(cand_text, cand_ents, source_row, "synonym", {"PT-999"})
    assert passed is False
    assert "Check 10 Failed" in msg


def test_leakage_checks_clean():
    """Verify leakage detector flags no leakage for valid train-derived rows."""
    df_train = pd.DataFrame({
        "document_id": ["DOC-1", "DOC-2"],
        "patient_id": ["PT-1", "PT-2"],
        "encounter_id": ["ENC-1", "ENC-2"],
        "text": ["Note A", "Note B"]
    })
    df_val = pd.DataFrame({
        "document_id": ["DOC-3"],
        "patient_id": ["PT-3"],
        "encounter_id": ["ENC-3"],
        "text": ["Note C"]
    })
    df_aug = pd.DataFrame({
        "document_id": ["AUG-DOC-1-001"],
        "patient_id": ["PT-1"],
        "encounter_id": ["ENC-1"],
        "text": ["Note A paraphrased"]
    })

    is_clean, metrics = verify_patient_split_isolation(df_aug, df_val, df_train)
    assert is_clean is True
    assert metrics["patient_overlap_with_val"] == 0
    assert metrics["encounter_overlap_with_val"] == 0
