"""
Unit tests for Stage 4 EDA Analyzer.
Tests token length profiling, entity distribution calculation,
partition uniformity verification, and artifact generation.
"""

import os
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import sys

EDA_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(EDA_SRC_DIR))

from eda_analyzer import DatasetEDAAnalyzer


@pytest.fixture
def mock_dataset_path(tmp_path):
    """Creates a miniature synthetic parquet dataset mimicking Stage 4 schema."""
    data = [
        {
            "patient_id": "PT-001",
            "note_id": "DOC-001",
            "clinical_note": "Patient presents with breast cancer receiving Docetaxel 100 mg. Experienced mild rash.",
            "instruction": "Analyze the clinical note and provide the patient's Risk, Key Finding, and Action.",
            "target_risk": "Moderate dermatologic toxicity risk associated with Docetaxel.",
            "target_key_finding": "Docetaxel administered at 100 mg; developed mild rash.",
            "target_action": "Prescribe topical hydrocortisone and continue therapy.",
            "ner_genes": ["BRCA1"],
            "ner_drugs": ["Docetaxel"],
            "ner_dosages": ["100 mg"],
            "ner_adverse_events": ["mild rash"],
            "entity_check_status": "PASS",
            "entity_coverage": 1.0,
            "missing_entities": [],
            "invented_entities": [],
            "generation_model_version": "test-v1",
            "split": "TRAIN"
        },
        {
            "patient_id": "PT-002",
            "note_id": "DOC-002",
            "clinical_note": "Patient with NSCLC EGFR mutation given Erlotinib 150 mg. No acute toxicities noted.",
            "instruction": "Analyze the clinical note and provide the patient's Risk, Key Finding, and Action.",
            "target_risk": "Low toxicity hazard with Erlotinib 150 mg.",
            "target_key_finding": "EGFR positive; patient well-tolerating Erlotinib 150 mg.",
            "target_action": "Continue standard outpatient dosing.",
            "ner_genes": ["EGFR"],
            "ner_drugs": ["Erlotinib"],
            "ner_dosages": ["150 mg"],
            "ner_adverse_events": ["no acute toxicities"],
            "entity_check_status": "PASS",
            "entity_coverage": 1.0,
            "missing_entities": [],
            "invented_entities": [],
            "generation_model_version": "test-v1",
            "split": "VALIDATION"
        },
        {
            "patient_id": "PT-003",
            "note_id": "DOC-003",
            "clinical_note": "Colorectal cancer KRAS wild-type treated with Cisplatin 75 mg/m2. Acute nephrotoxicity.",
            "instruction": "Analyze the clinical note and provide the patient's Risk, Key Finding, and Action.",
            "target_risk": "Severe renal risk with Cisplatin.",
            "target_key_finding": "Cisplatin administered; acute creatinine increase.",
            "target_action": "Hold chemotherapy and initiate vigorous IV saline hydration.",
            "ner_genes": ["KRAS"],
            "ner_drugs": ["Cisplatin"],
            "ner_dosages": ["75 mg/m2"],
            "ner_adverse_events": ["Acute nephrotoxicity"],
            "entity_check_status": "PASS",
            "entity_coverage": 1.0,
            "missing_entities": [],
            "invented_entities": [],
            "generation_model_version": "test-v1",
            "split": "TEST"
        }
    ]
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_dataset.parquet"
    df.to_parquet(file_path)
    return str(file_path)


def test_token_estimation():
    """Verify BPE token estimation behaves realistically."""
    text = "Administer Cisplatin 100 mg IV."
    tokens = DatasetEDAAnalyzer.estimate_tokens(text)
    assert tokens > 0
    assert DatasetEDAAnalyzer.estimate_tokens("") == 0
    assert DatasetEDAAnalyzer.estimate_tokens(None) == 0


def test_length_statistics(mock_dataset_path, tmp_path):
    """Verify sequence length computations and truncation risk metrics."""
    out_dir = tmp_path / "eda_out"
    analyzer = DatasetEDAAnalyzer(mock_dataset_path, str(out_dir))
    stats = analyzer.compute_length_statistics()

    assert "clinical_note" in stats
    assert "target_risk" in stats
    assert "full_sequence" in stats
    assert "truncation_risk" in stats
    assert stats["clinical_note"]["word_count"]["mean"] > 0
    assert "threshold_512" in stats["truncation_risk"]
    assert stats["truncation_risk"]["threshold_512"]["exceeded_pct"] >= 0.0


def test_entity_coverage(mock_dataset_path, tmp_path):
    """Verify entity frequency and vocabulary coverage calculations."""
    out_dir = tmp_path / "eda_out"
    analyzer = DatasetEDAAnalyzer(mock_dataset_path, str(out_dir))
    coverage = analyzer.compute_entity_and_vocab_coverage()

    assert coverage["entities"]["unique_drugs"] == 3
    assert "Docetaxel" in coverage["entities"]["top_drugs"]
    assert "Erlotinib" in coverage["entities"]["top_drugs"]
    assert coverage["entities"]["unique_genes"] == 3
    assert coverage["lexical_diversity"]["clinical_note_vocab_size"] > 0
    assert coverage["lexical_diversity"]["target_vocab_grounded_in_notes_pct"] > 0.0


def test_partition_uniformity(mock_dataset_path, tmp_path):
    """Verify split partition profiling."""
    out_dir = tmp_path / "eda_out"
    analyzer = DatasetEDAAnalyzer(mock_dataset_path, str(out_dir))
    uniformity = analyzer.compute_partition_uniformity()

    assert "TRAIN" in uniformity
    assert "VALIDATION" in uniformity
    assert "TEST" in uniformity
    assert uniformity["TRAIN"]["record_count"] == 1
    assert uniformity["VALIDATION"]["record_count"] == 1
    assert uniformity["TEST"]["record_count"] == 1


def test_full_eda_run_generates_artifacts(mock_dataset_path, tmp_path):
    """Verify full EDA workflow creates all expected reports, figures, and JSON outputs."""
    out_dir = tmp_path / "eda_out"
    analyzer = DatasetEDAAnalyzer(mock_dataset_path, str(out_dir))
    summary = analyzer.run()

    assert summary["dataset_overview"]["total_records"] == 3
    assert (out_dir / "results" / "eda_summary.json").exists()
    assert (out_dir / "reports" / "eda_report.md").exists()
    assert (out_dir / "figures" / "token_length_distributions.png").exists()
    assert (out_dir / "figures" / "entity_frequency_top20.png").exists()
    assert (out_dir / "figures" / "split_partition_balance.png").exists()
    assert (out_dir / "figures" / "context_window_truncation_risk.png").exists()
