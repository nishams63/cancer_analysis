"""
Tests for Data Loader and Profiling Modules.
"""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from data_loader import Stage4DataLoader, calculate_file_sha256
from profiling import profile_dataset_overview, analyze_missingness, analyze_duplicates


@pytest.fixture
def sample_valid_df():
    return pd.DataFrame({
        "patient_id": ["PT-001", "PT-001", "PT-002"],
        "note_id": ["DOC-001", "DOC-002", "DOC-003"],
        "clinical_note": [
            "Patient has lung cancer with EGFR mutation receiving Osimertinib 80 mg.",
            "Cycle 2 evaluation with no adverse events.",
            "Patient with KRAS G12C receiving Sotorasib."
        ],
        "instruction": ["Analyze clinical note."] * 3,
        "target_risk": ["Low toxicity risk.", "Low toxicity risk.", "Moderate risk."],
        "target_key_finding": ["EGFR identified.", "Stable.", "KRAS identified."],
        "target_action": ["Continue monitoring.", "Continue monitoring.", "Monitor closely."],
        "ner_genes": [["EGFR"], [], ["KRAS"]],
        "ner_drugs": [["Osimertinib"], [], ["Sotorasib"]],
        "ner_dosages": [["80 mg"], [], []],
        "ner_adverse_events": [[], [], []],
        "entity_check_status": ["PASS"] * 3,
        "entity_coverage": [1.0] * 3,
        "missing_entities": [[]] * 3,
        "invented_entities": [[]] * 3,
        "generation_model_version": ["clinical-draft-v1"] * 3,
        "split": ["TRAIN", "TRAIN", "VALIDATION"]
    })


def test_schema_validation_success(sample_valid_df, tmp_path):
    p = tmp_path / "valid.parquet"
    sample_valid_df.to_parquet(p, index=False)

    loader = Stage4DataLoader(str(p))
    df, meta = loader.load_data()
    assert len(df) == 3
    assert meta["row_count"] == 3
    assert "sha256" in meta


def test_schema_validation_missing_column(sample_valid_df, tmp_path):
    bad_df = sample_valid_df.drop(columns=["target_risk"])
    p = tmp_path / "bad.parquet"
    bad_df.to_parquet(p, index=False)

    loader = Stage4DataLoader(str(p))
    with pytest.raises(ValueError, match="missing required columns"):
        loader.load_data()


def test_profiling_and_missingness(sample_valid_df):
    overview = profile_dataset_overview(sample_valid_df)
    assert overview["total_records"] == 3
    assert overview["unique_patients"] == 2
    assert overview["unique_notes"] == 3

    missing = analyze_missingness(sample_valid_df)
    assert missing["total_missing_cells"] == 0

    dups = analyze_duplicates(sample_valid_df)
    assert dups["exact_row_duplicates"] == 0
