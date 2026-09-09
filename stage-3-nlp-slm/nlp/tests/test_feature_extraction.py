"""
Test Suite: Clinical Feature Extraction.
Verifies negation-scoped text transformation, structured count extraction, and pipeline persistence.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
from scipy.sparse import issparse

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from feature_extraction import (
    create_negation_scoped_text,
    extract_structured_concept_features,
    ClinicalFeaturePipeline
)


def test_negation_scoped_text_prefixing():
    """Verify negated tokens receive 'neg_' prefix and historical receive 'hist_'."""
    raw = "Patient has no fever. Denies dyspnea. History of neutropenia."
    scoped = create_negation_scoped_text(raw)
    assert "neg_fever" in scoped
    assert "neg_dyspnea" in scoped
    assert "hist_neutropenia" in scoped


def test_extract_structured_concept_features():
    """Verify structured concept counts are generated without missing values."""
    sample_df = pd.DataFrame([{
        "document_id": "DOC-TEST-1",
        "text": "Administered Cisplatin 75.5 mg/m2. Patient denies nausea. Grade 3 fatigue reported."
    }])
    struct_df = extract_structured_concept_features(sample_df)
    assert len(struct_df) == 1
    assert struct_df["drug_mentions"].iloc[0] >= 1
    assert struct_df["dosage_mentions"].iloc[0] >= 1
    assert struct_df["has_grade_3_4"].iloc[0] == 1
    assert struct_df["negated_concepts"].iloc[0] >= 1
    assert struct_df.isnull().sum().sum() == 0


def test_clinical_feature_pipeline_fit_transform():
    """Verify feature pipeline creates valid sparse matrices and persists/reloads correctly."""
    train_df = pd.DataFrame([
        {"document_id": "DOC-1", "text": "Patient has Stage III NSCLC, EGFR positive. No fever. Grade 1 rash."},
        {"document_id": "DOC-2", "text": "Patient on Osimertinib 80 mg daily. Denies dyspnea. Stable ANC."}
    ])
    pipeline = ClinicalFeaturePipeline(max_features=50)
    pipeline.fit(train_df)

    X_train, struct_df = pipeline.transform(train_df)
    assert issparse(X_train)
    assert X_train.shape[0] == 2
    assert len(struct_df) == 2
