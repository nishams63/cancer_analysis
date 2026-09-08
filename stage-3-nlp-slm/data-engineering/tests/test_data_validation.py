"""
Test Suite: Data Validation & Schema Integrity.
Verifies that processed NLP datasets meet all quality gate requirements.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import pytest

# Add src to sys.path
TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (
    PROCESSED_PARQUET_PATH,
    VALID_DOC_TYPES,
    VALID_URGENCY_LEVELS,
    VALID_HAZARD_TYPES
)

@pytest.fixture(scope="module")
def processed_df():
    """Load processed parquet dataset for testing."""
    assert PROCESSED_PARQUET_PATH.exists(), f"Processed parquet file not found at: {PROCESSED_PARQUET_PATH}"
    df = pd.read_parquet(PROCESSED_PARQUET_PATH)
    return df

def test_mandatory_columns_exist(processed_df):
    """Verify all mandatory columns are present."""
    required = [
        "document_id",
        "patient_id",
        "encounter_id",
        "document_type",
        "document_date",
        "index_date",
        "text",
        "cleaned_text",
        "word_count",
        "char_count",
        "urgency_level",
        "hazard_type",
        "ner_entities",
        "slm_summary",
        "source",
        "data_split",
        "quality_status",
        "disclaimer"
    ]
    missing = [c for c in required if c not in processed_df.columns]
    assert len(missing) == 0, f"Missing required columns: {missing}"

def test_no_null_values_in_mandatory_fields(processed_df):
    """Verify that no mandatory column contains null values."""
    mandatory = ["document_id", "patient_id", "encounter_id", "text", "urgency_level", "hazard_type", "data_split"]
    for col in mandatory:
        null_count = processed_df[col].isnull().sum()
        assert null_count == 0, f"Column '{col}' has {null_count} null values!"

def test_document_id_uniqueness(processed_df):
    """Verify every document has a strictly unique document_id."""
    assert processed_df["document_id"].is_unique, "Duplicate document_id detected in processed dataset!"

def test_valid_categorical_values(processed_df):
    """Verify target classes and document types belong to approved vocabulary."""
    invalid_docs = set(processed_df["document_type"].unique()) - VALID_DOC_TYPES
    assert len(invalid_docs) == 0, f"Invalid document types found: {invalid_docs}"

    invalid_urgencies = set(processed_df["urgency_level"].unique()) - VALID_URGENCY_LEVELS
    assert len(invalid_urgencies) == 0, f"Invalid urgency levels found: {invalid_urgencies}"

    invalid_hazards = set(processed_df["hazard_type"].unique()) - VALID_HAZARD_TYPES
    assert len(invalid_hazards) == 0, f"Invalid hazard types found: {invalid_hazards}"

def test_temporal_validity(processed_df):
    """Verify document_date does not exceed index_date (no future notes in intake)."""
    doc_dates = pd.to_datetime(processed_df["document_date"])
    idx_dates = pd.to_datetime(processed_df["index_date"])
    violations = (doc_dates > idx_dates).sum()
    assert violations == 0, f"Detected {violations} temporal order violations (document_date > index_date)!"

def test_text_length_thresholds(processed_df):
    """Verify all documents conform to length thresholds."""
    assert (processed_df["word_count"] >= 15).all(), "Found documents with < 15 words!"
    assert (processed_df["word_count"] <= 850).all(), "Found documents with > 850 words!"
    assert (processed_df["char_count"] >= 80).all(), "Found documents with < 80 characters!"
