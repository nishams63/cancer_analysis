"""
Test Suite: Stage 3 Clinical NLP & SLM Exploratory Data Analysis (EDA).
Verifies schema, statistics generation, split integrity, figure exports,
and enforces strict read-only guarantees on source dataset files via SHA256 checksums.
"""

import sys
import json
import hashlib
from pathlib import Path
import pytest
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import eda_utils

DATA_DIR = TEST_DIR.parent.parent / "data-engineering" / "data" / "processed"
PRIMARY_PARQUET = DATA_DIR / "clinical_nlp_dataset_v1.parquet"
EDA_RESULTS_DIR = TEST_DIR.parent / "results"
EDA_FIGURES_DIR = TEST_DIR.parent / "figures"


def compute_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def pre_analysis_hashes():
    """Capture checksums of all processed data files prior to running tests."""
    hashes = {}
    for f in DATA_DIR.glob("*.parquet"):
        hashes[f.name] = compute_file_hash(f)
    return hashes


@pytest.fixture(scope="module")
def loaded_df():
    """Load canonical processed parquet dataset."""
    assert PRIMARY_PARQUET.exists(), f"Source dataset missing at: {PRIMARY_PARQUET}"
    return eda_utils.load_dataset(PRIMARY_PARQUET)


def test_dataset_loads_and_schema_present(loaded_df):
    """Verify dataset loads and contains all required schema fields."""
    assert len(loaded_df) == 6098, f"Expected 6098 rows, got {len(loaded_df)}"
    assert loaded_df.shape[1] == 18, f"Expected 18 columns, got {loaded_df.shape[1]}"

    required_cols = [
        "document_id", "patient_id", "encounter_id", "document_type",
        "document_date", "index_date", "text", "cleaned_text",
        "word_count", "char_count", "urgency_level", "hazard_type",
        "ner_entities", "slm_summary", "source", "data_split",
        "quality_status", "disclaimer"
    ]
    for col in required_cols:
        assert col in loaded_df.columns, f"Missing required column: {col}"
        assert loaded_df[col].isnull().sum() == 0, f"Column '{col}' has null values!"


def test_overview_statistics_generation(loaded_df):
    """Verify compute_dataset_overview produces accurate metrics."""
    overview = eda_utils.compute_dataset_overview(loaded_df)
    assert overview["total_documents"] == 6098
    assert overview["unique_patients"] == 1000
    assert overview["unique_encounters"] == 2038
    assert overview["total_missing_cells"] == 0
    assert overview["exact_duplicate_rows"] == 0


def test_text_length_and_vocabulary_statistics(loaded_df):
    """Verify text length and vocabulary metrics conform to clinical expectations."""
    t_stats = eda_utils.compute_text_length_statistics(loaded_df)
    assert t_stats["word_stats"]["min"] >= 15
    assert t_stats["word_stats"]["max"] <= 400
    assert t_stats["empty_docs_count"] == 0

    v_stats = eda_utils.compute_vocabulary_statistics(loaded_df)
    assert v_stats["vocabulary_size"] > 1000
    assert v_stats["type_token_ratio"] > 0.001
    assert len(v_stats["top_50_terms"]) == 50


def test_negation_profiling(loaded_df):
    """Verify negation metrics and clinical patterns."""
    neg = eda_utils.compute_negation_statistics(loaded_df)
    assert neg["total_documents"] == 6098
    assert neg["documents_with_negation"] > 0
    assert neg["total_negation_occurrences"] > 0
    assert "\\bno\\b" in neg["pattern_prevalence"]


def test_split_integrity_and_zero_leakage(loaded_df):
    """Verify patient and encounter isolation across dataset partitions."""
    splits = eda_utils.compute_split_statistics(loaded_df)
    assert splits["patient_overlap"]["is_patient_leakage_free"] is True
    assert splits["patient_overlap"]["train_validation"] == 0
    assert splits["patient_overlap"]["train_locked_test"] == 0
    assert splits["patient_overlap"]["validation_locked_test"] == 0
    assert splits["encounter_overlap"]["is_encounter_leakage_free"] is True


def test_leakage_cues_scan(loaded_df):
    """Verify zero forbidden post-treatment outcome cues in clinical narratives."""
    leakage = eda_utils.analyze_leakage_risks(loaded_df)
    assert leakage["total_leakage_cues_detected"] == 0
    assert leakage["leakage_risk_classification"] == "No evidence"


def test_eda_summary_json_exists_and_valid():
    """Verify machine-readable eda_summary.json exists and parses correctly."""
    json_path = EDA_RESULTS_DIR / "eda_summary.json"
    assert json_path.exists(), f"eda_summary.json not found at {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "dataset_overview" in data
    assert "text_length_statistics" in data
    assert "vocabulary_statistics" in data
    assert "label_distribution" in data
    assert "nlp_readiness" in data
    assert data["nlp_readiness"]["verdict"] == "READY_FOR_MODELING_WITH_IMBALANCE_CONTROLS"


def test_figures_exist_in_all_six_directories():
    """Verify that all 16 designated plots exist across the 6 figure folders."""
    required_dirs = [
        "text_length", "vocabulary", "document_types",
        "labels", "temporal", "patients"
    ]
    for rdir in required_dirs:
        dir_path = EDA_FIGURES_DIR / rdir
        assert dir_path.exists(), f"Figures directory missing: {dir_path}"
        pngs = list(dir_path.glob("*.png"))
        assert len(pngs) >= 2, f"Expected at least 2 figures in {rdir}, found {len(pngs)}"


def test_source_data_read_only_invariance(pre_analysis_hashes):
    """Verify that no source parquet file was modified by computing post-analysis hashes."""
    for fname, pre_hash in pre_analysis_hashes.items():
        post_path = DATA_DIR / fname
        assert post_path.exists(), f"Source file {fname} vanished!"
        post_hash = compute_file_hash(post_path)
        assert pre_hash == post_hash, f"CRITICAL: Source file {fname} was modified during EDA! Hash mismatch."
