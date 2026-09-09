"""
Regression and safety tests to verify baseline integrity and dataset isolation.
"""

import os
import joblib
import pandas as pd
from pathlib import Path


def test_baseline_artifacts_exist_and_load():
    base_dir = Path("stage-3-nlp-slm/nlp/artifacts")
    assert (base_dir / "urgency_baseline_model.joblib").exists(), "Baseline urgency model missing!"
    assert (base_dir / "hazard_baseline_model.joblib").exists(), "Baseline hazard model missing!"
    assert (base_dir / "tokenizers" / "tfidf_vectorizer.joblib").exists(), "Baseline TF-IDF vectorizer missing!"
    
    # Load and check model attributes
    urgency_model = joblib.load(base_dir / "urgency_baseline_model.joblib")
    assert hasattr(urgency_model, "predict"), "Loaded urgency model must have predict method"
    hazard_model = joblib.load(base_dir / "hazard_baseline_model.joblib")
    assert hasattr(hazard_model, "predict"), "Loaded hazard model must have predict method"


def test_dataset_partitions_integrity():
    data_dir = Path("stage-3-nlp-slm/data-engineering/data/processed")
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "validation.parquet"
    test_path = data_dir / "locked_test.parquet"
    
    assert train_path.exists(), "train.parquet missing"
    assert val_path.exists(), "validation.parquet missing"
    assert test_path.exists(), "locked_test.parquet missing"
    
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)
    
    # Strictly check row counts from frozen dataset
    assert len(df_train) == 4261, f"Train row count changed: {len(df_train)}"
    assert len(df_val) == 909, f"Validation row count changed: {len(df_val)}"
    assert len(df_test) == 928, f"Locked test row count changed: {len(df_test)}"
    
    # Strict patient leakage verification
    train_pts = set(df_train["patient_id"].unique())
    val_pts = set(df_val["patient_id"].unique())
    test_pts = set(df_test["patient_id"].unique())
    
    assert len(train_pts.intersection(val_pts)) == 0, "Patient leakage between train and val!"
    assert len(train_pts.intersection(test_pts)) == 0, "Patient leakage between train and locked-test!"
    assert len(val_pts.intersection(test_pts)) == 0, "Patient leakage between val and locked-test!"
