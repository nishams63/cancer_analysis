"""
Test Suite: Patient-Level Split Integrity and Upstream Read-Only Invariance.
Enforces strictly zero patient leakage, zero encounter leakage, and guarantees
that upstream Data Engineering and EDA files were NEVER modified during NLP engineering.
"""

import sys
import hashlib
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
    verify_patient_split_isolation
)
from config import UPSTREAM_DATA_DIR

# Frozen baseline SHA256 hashes from Data Engineering
FROZEN_UPSTREAM_HASHES = {
    "clinical_nlp_dataset_v1.parquet": "426ea0c51f354a5fa9e1177ec0d8fabc0ed26ed708f748ee8f8f4b0f18554230",
    "train.parquet": "5862db7ec4e42fb71bf207906c9acf21f9d43280a32336ebf43dff77e6f55133",
    "validation.parquet": "2b6787a394d262c8f8b94b92914777f495e30a7be4eaf925ce862f1302899a4c",
    "locked_test.parquet": "491562c892a6af2f08695eefefe782daeae12d52f1fef5a6ad11060d96af4509",
}


def test_zero_patient_leakage():
    """Verify strictly zero patient overlap between any pair of splits."""
    df_train = load_train_data()
    df_val = load_validation_data()
    df_test = load_locked_test_data()

    res = verify_patient_split_isolation(df_train, df_val, df_test)
    assert res["is_strictly_isolated"] is True
    assert res["patient_overlap"]["train_val"] == 0
    assert res["patient_overlap"]["train_test"] == 0
    assert res["patient_overlap"]["val_test"] == 0


def test_zero_encounter_leakage():
    """Verify strictly zero encounter overlap between any pair of splits."""
    df_train = load_train_data()
    df_val = load_validation_data()
    df_test = load_locked_test_data()

    res = verify_patient_split_isolation(df_train, df_val, df_test)
    assert res["encounter_overlap"]["train_val"] == 0
    assert res["encounter_overlap"]["train_test"] == 0
    assert res["encounter_overlap"]["val_test"] == 0


def test_split_proportions():
    """Verify patient counts match 700 / 150 / 150 distribution."""
    df_train = load_train_data()
    df_val = load_validation_data()
    df_test = load_locked_test_data()

    assert df_train["patient_id"].nunique() == 700
    assert df_val["patient_id"].nunique() == 150
    assert df_test["patient_id"].nunique() == 150


def test_upstream_data_read_only_invariance():
    """Verify that every upstream parquet file has identical cryptographic hash."""
    for fname, expected_hash in FROZEN_UPSTREAM_HASHES.items():
        fpath = UPSTREAM_DATA_DIR / fname
        assert fpath.exists(), f"Upstream file vanished: {fpath}"
        actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, (
            f"CRITICAL VIOLATION: Upstream data file {fname} was modified! "
            f"Expected {expected_hash}, got {actual_hash}"
        )
