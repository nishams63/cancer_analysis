"""
Test Suite: NLP Data Loader & Schema Verification.
Verifies read-only loading of official partitions and schema completeness.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import (
    load_train_data,
    load_validation_data,
    load_locked_test_data,
    load_full_dataset,
    verify_patient_split_isolation,
    MANDATORY_COLUMNS
)


def test_load_train_data():
    """Verify train partition loads with correct dimensions and columns."""
    df_train = load_train_data()
    assert len(df_train) == 4261, f"Expected 4261 train rows, got {len(df_train)}"
    assert df_train["patient_id"].nunique() == 700
    for col in MANDATORY_COLUMNS:
        assert col in df_train.columns
        assert df_train[col].isnull().sum() == 0


def test_load_validation_data():
    """Verify validation partition loads with correct dimensions and columns."""
    df_val = load_validation_data()
    assert len(df_val) == 909, f"Expected 909 validation rows, got {len(df_val)}"
    assert df_val["patient_id"].nunique() == 150
    for col in MANDATORY_COLUMNS:
        assert col in df_val.columns
        assert df_val[col].isnull().sum() == 0


def test_load_locked_test_data():
    """Verify locked test partition loads with correct dimensions (for verification only)."""
    df_test = load_locked_test_data()
    assert len(df_test) == 928, f"Expected 928 locked test rows, got {len(df_test)}"
    assert df_test["patient_id"].nunique() == 150
    for col in MANDATORY_COLUMNS:
        assert col in df_test.columns


def test_load_full_dataset():
    """Verify canonical full dataset loads with 6,098 records."""
    df_full = load_full_dataset()
    assert len(df_full) == 6098
    assert df_full["patient_id"].nunique() == 1000
    assert df_full["encounter_id"].nunique() == 2038


def test_split_isolation_check():
    """Verify patient and encounter isolation helper function."""
    df_train = load_train_data()
    df_val = load_validation_data()
    df_test = load_locked_test_data()
    res = verify_patient_split_isolation(df_train, df_val, df_test)
    assert res["is_strictly_isolated"] is True
    assert res["patient_overlap"]["train_val"] == 0
    assert res["patient_overlap"]["train_test"] == 0
    assert res["patient_overlap"]["val_test"] == 0
