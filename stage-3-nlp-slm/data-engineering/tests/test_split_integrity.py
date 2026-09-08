"""
Test Suite: Split Integrity and Patient Leakage Prevention.
Verifies that train, validation, and locked test splits have zero patient overlap,
zero encounter overlap, correct patient counts, and consistent label representations.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (
    PROCESSED_DATA_DIR,
    PROCESSED_PARQUET_PATH,
    VALID_URGENCY_LEVELS,
    VALID_DOC_TYPES
)

@pytest.fixture(scope="module")
def split_dfs():
    """Load train, validation, and locked test parquet splits."""
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    val_path = PROCESSED_DATA_DIR / "validation.parquet"
    test_path = PROCESSED_DATA_DIR / "locked_test.parquet"

    assert train_path.exists(), f"Train split missing at {train_path}"
    assert val_path.exists(), f"Validation split missing at {val_path}"
    assert test_path.exists(), f"Locked test split missing at {test_path}"

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    return {
        "train": df_train,
        "validation": df_val,
        "locked_test": df_test
    }

def test_zero_patient_leakage(split_dfs):
    """Verify strictly zero patient ID overlap between any pair of splits."""
    train_pts = set(split_dfs["train"]["patient_id"].unique())
    val_pts = set(split_dfs["validation"]["patient_id"].unique())
    test_pts = set(split_dfs["locked_test"]["patient_id"].unique())

    train_val_overlap = train_pts.intersection(val_pts)
    train_test_overlap = train_pts.intersection(test_pts)
    val_test_overlap = val_pts.intersection(test_pts)

    assert len(train_val_overlap) == 0, f"Patient leakage between Train and Validation: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Patient leakage between Train and Test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Patient leakage between Validation and Test: {val_test_overlap}"

def test_zero_encounter_leakage(split_dfs):
    """Verify strictly zero encounter ID overlap between splits."""
    train_encs = set(split_dfs["train"]["encounter_id"].unique())
    val_encs = set(split_dfs["validation"]["encounter_id"].unique())
    test_encs = set(split_dfs["locked_test"]["encounter_id"].unique())

    assert len(train_encs.intersection(val_encs)) == 0, "Encounter overlap between Train and Val!"
    assert len(train_encs.intersection(test_encs)) == 0, "Encounter overlap between Train and Test!"
    assert len(val_encs.intersection(test_encs)) == 0, "Encounter overlap between Val and Test!"

def test_patient_split_proportions(split_dfs):
    """Verify patient counts match 70% / 15% / 15% split design (700, 150, 150)."""
    n_train = split_dfs["train"]["patient_id"].nunique()
    n_val = split_dfs["validation"]["patient_id"].nunique()
    n_test = split_dfs["locked_test"]["patient_id"].nunique()

    assert n_train == 700, f"Expected 700 train patients, got {n_train}"
    assert n_val == 150, f"Expected 150 val patients, got {n_val}"
    assert n_test == 150, f"Expected 150 test patients, got {n_test}"
    assert (n_train + n_val + n_test) == 1000, "Total split patients does not equal 1000 master cohort!"

def test_document_conservation(split_dfs):
    """Verify total split documents equals the full clean processed dataset."""
    full_df = pd.read_parquet(PROCESSED_PARQUET_PATH)
    total_split_docs = len(split_dfs["train"]) + len(split_dfs["validation"]) + len(split_dfs["locked_test"])
    assert total_split_docs == len(full_df), f"Split sum {total_split_docs} != Master dataset size {len(full_df)}"

def test_split_label_representation(split_dfs):
    """Verify that every split contains representation across all target urgency classes and document types."""
    for split_name, df in split_dfs.items():
        present_urgencies = set(df["urgency_level"].unique())
        assert present_urgencies == VALID_URGENCY_LEVELS, f"Split '{split_name}' missing classes: {VALID_URGENCY_LEVELS - present_urgencies}"

        present_doc_types = set(df["document_type"].unique())
        assert present_doc_types == VALID_DOC_TYPES, f"Split '{split_name}' missing document types: {VALID_DOC_TYPES - present_doc_types}"
