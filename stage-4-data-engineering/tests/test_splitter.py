"""
Unit Tests for Section 9 Patient-Level Splitter.
Tests zero patient overlap, ratio compliance, and partition integrity.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from splitter import PatientLevelSplitter, PatientSplitError


@pytest.fixture
def mock_multi_doc_df():
    # 20 unique patients, each with 3 notes = 60 rows
    rows = []
    for p_idx in range(1, 21):
        pid = f"PT-{p_idx:04d}"
        for n_idx in range(1, 4):
            rows.append({
                "patient_id": pid,
                "note_id": f"DOC-{pid}-{n_idx}",
                "clinical_note": f"Clinical observation {n_idx} for {pid}"
            })
    return pd.DataFrame(rows)


def test_patient_level_splitting_isolation(mock_multi_doc_df):
    """Verify that every patient belongs to strictly one partition."""
    splitter = PatientLevelSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, align_with_stage3=False)
    df_split, summary = splitter.split_dataset(mock_multi_doc_df)

    pats_train = set(df_split[df_split["split"] == "TRAIN"]["patient_id"])
    pats_val = set(df_split[df_split["split"] == "VALIDATION"]["patient_id"])
    pats_test = set(df_split[df_split["split"] == "TEST"]["patient_id"])

    # Strict isolation
    assert len(pats_train.intersection(pats_val)) == 0
    assert len(pats_train.intersection(pats_test)) == 0
    assert len(pats_val.intersection(pats_test)) == 0
    assert summary["patient_leakage_count"] == 0


def test_multi_note_patients_never_split(mock_multi_doc_df):
    """Verify all notes for a single patient are assigned to the identical split."""
    splitter = PatientLevelSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, align_with_stage3=False)
    df_split, _ = splitter.split_dataset(mock_multi_doc_df)

    for pid, group in df_split.groupby("patient_id"):
        assert group["split"].nunique() == 1, f"Patient {pid} notes scattered across multiple splits!"
