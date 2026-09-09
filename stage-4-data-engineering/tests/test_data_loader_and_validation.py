"""
Unit Tests for Data Loader & Validator Modules.
Tests handling of missing notes, empty notes, missing IDs, duplicate notes,
and malformed NER entity structures.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from validator import Stage4DataValidator
from data_loader import Stage4DataLoader


def test_validator_catches_missing_and_empty_notes():
    """Verify null and empty notes are flagged as invalid."""
    df = pd.DataFrame([
        {"document_id": "DOC-1", "patient_id": "PT-1", "text": "Valid length clinical consultation note text here."},
        {"document_id": "DOC-2", "patient_id": "PT-2", "text": None},
        {"document_id": "DOC-3", "patient_id": "PT-3", "text": "   "}
    ])
    validator = Stage4DataValidator()
    clean_df, invalid_df, audit = validator.validate_dataset(df)

    assert audit["missing_note_count"] == 1
    assert audit["empty_note_count"] == 1
    assert len(clean_df) == 1
    assert len(invalid_df) == 2


def test_validator_catches_missing_ids():
    """Verify missing patient or note identifiers are caught."""
    df = pd.DataFrame([
        {"document_id": "", "patient_id": "PT-1", "text": "Valid note with missing document ID."},
        {"document_id": "DOC-2", "patient_id": "", "text": "Valid note with missing patient ID."},
        {"document_id": "DOC-3", "patient_id": "PT-3", "text": "Fully valid note with both IDs present."}
    ])
    validator = Stage4DataValidator()
    clean_df, invalid_df, audit = validator.validate_dataset(df)

    assert audit["missing_note_id_count"] == 1
    assert audit["missing_patient_id_count"] == 1
    assert len(clean_df) == 1


def test_validator_catches_malformed_ner():
    """Verify malformed NER entities are properly flagged."""
    df = pd.DataFrame([
        {
            "document_id": "DOC-1",
            "patient_id": "PT-1",
            "text": "Valid clinical consultation narrative note here.",
            "ner_entities": "INVALID_JSON{{{"
        }
    ])
    validator = Stage4DataValidator()
    clean_df, invalid_df, audit = validator.validate_dataset(df)

    assert audit["invalid_ner_record_count"] == 1
    assert len(invalid_df) == 1


def test_data_loader_parses_ner_categories():
    """Verify entity parser correctly separates Genes, Drugs, Dosages, and AEs."""
    ner_raw = [
        {"label": "GENE_MUTATION", "text": "EGFR T790M"},
        {"label": "DRUG_NAME", "text": "Osimertinib"},
        {"label": "DOSAGE", "text": "80 mg"},
        {"label": "ADVERSE_EVENT", "text": "diarrhea"}
    ]
    genes, drugs, doses, aes, raw_list = Stage4DataLoader.parse_ner_entities(ner_raw)
    assert genes == ["EGFR T790M"]
    assert drugs == ["Osimertinib"]
    assert doses == ["80 mg"]
    assert aes == ["diarrhea"]
    assert len(raw_list) == 4
