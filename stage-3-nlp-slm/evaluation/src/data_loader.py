"""
Read-Only Data Loader for Stage 3 Clinical NLP Evaluation.
Loads upstream datasets with strict schema validation and invariant checks.
"""

from typing import Dict, Any, List
import pandas as pd
from config import (
    RAW_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
    VAL_PARQUET_PATH,
    LOCKED_TEST_PARQUET_PATH,
    VALID_DOC_TYPES,
    VALID_URGENCY_LEVELS,
    VALID_HAZARD_TYPES
)

MANDATORY_COLUMNS: List[str] = [
    "document_id", "patient_id", "encounter_id", "document_type",
    "document_date", "index_date", "text", "cleaned_text",
    "word_count", "char_count", "urgency_level", "hazard_type",
    "ner_entities", "slm_summary", "source", "data_split",
    "quality_status", "disclaimer"
]


def validate_schema(df: pd.DataFrame, source_name: str) -> None:
    """Ensure dataframe conforms to the frozen upstream schema without any missing columns or invalid values."""
    missing = [col for col in MANDATORY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Schema violation in {source_name}! Missing columns: {missing}")

    for col in ["document_id", "patient_id", "encounter_id", "text", "urgency_level", "hazard_type"]:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise ValueError(f"Null values detected in mandatory column '{col}' ({null_count} nulls) in {source_name}!")

    invalid_docs = set(df["document_type"].unique()) - VALID_DOC_TYPES
    if invalid_docs:
        raise ValueError(f"Unrecognized document types {invalid_docs} in {source_name}!")

    invalid_urg = set(df["urgency_level"].unique()) - set(VALID_URGENCY_LEVELS)
    if invalid_urg:
        raise ValueError(f"Unrecognized urgency levels {invalid_urg} in {source_name}!")

    invalid_haz = set(df["hazard_type"].unique()) - set(VALID_HAZARD_TYPES)
    if invalid_haz:
        raise ValueError(f"Unrecognized hazard types {invalid_haz} in {source_name}!")


def load_train_data() -> pd.DataFrame:
    """Load the official TRAIN partition (N=4,261) in read-only mode."""
    if not TRAIN_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Train split missing at: {TRAIN_PARQUET_PATH}")
    df = pd.read_parquet(TRAIN_PARQUET_PATH).copy(deep=True)
    validate_schema(df, "TRAIN")
    return df


def load_validation_data() -> pd.DataFrame:
    """Load the official VALIDATION partition (N=909) in read-only mode."""
    if not VAL_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Validation split missing at: {VAL_PARQUET_PATH}")
    df = pd.read_parquet(VAL_PARQUET_PATH).copy(deep=True)
    validate_schema(df, "VALIDATION")
    return df


def load_locked_test_data() -> pd.DataFrame:
    """
    Load the official LOCKED TEST partition (N=928) in read-only mode.
    WARNING: Strictly read-only; never used for optimization or tuning.
    """
    if not LOCKED_TEST_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Locked test split missing at: {LOCKED_TEST_PARQUET_PATH}")
    df = pd.read_parquet(LOCKED_TEST_PARQUET_PATH).copy(deep=True)
    validate_schema(df, "LOCKED_TEST")
    return df


def load_full_dataset() -> pd.DataFrame:
    """Load the complete 6,098-record canonical dataset in read-only mode."""
    if not RAW_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Canonical dataset missing at: {RAW_PARQUET_PATH}")
    df = pd.read_parquet(RAW_PARQUET_PATH).copy(deep=True)
    validate_schema(df, "CANONICAL_DATASET")
    return df


def verify_patient_split_isolation(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, Any]:
    """Verify strictly zero patient or encounter leakage across split boundaries."""
    train_pts = set(df_train["patient_id"].unique())
    val_pts = set(df_val["patient_id"].unique())
    test_pts = set(df_test["patient_id"].unique())

    train_encs = set(df_train["encounter_id"].unique())
    val_encs = set(df_val["encounter_id"].unique())
    test_encs = set(df_test["encounter_id"].unique())

    train_val_pt_overlap = len(train_pts & val_pts)
    train_test_pt_overlap = len(train_pts & test_pts)
    val_test_pt_overlap = len(val_pts & test_pts)

    train_val_enc_overlap = len(train_encs & val_encs)
    train_test_enc_overlap = len(train_encs & test_encs)
    val_test_enc_overlap = len(val_encs & test_encs)

    is_isolated = (
        train_val_pt_overlap == 0 and
        train_test_pt_overlap == 0 and
        val_test_pt_overlap == 0 and
        train_val_enc_overlap == 0 and
        train_test_enc_overlap == 0 and
        val_test_enc_overlap == 0
    )

    return {
        "is_strictly_isolated": is_isolated,
        "patient_overlap": {
            "train_val": train_val_pt_overlap,
            "train_test": train_test_pt_overlap,
            "val_test": val_test_pt_overlap
        },
        "encounter_overlap": {
            "train_val": train_val_enc_overlap,
            "train_test": train_test_enc_overlap,
            "val_test": val_test_enc_overlap
        },
        "patient_counts": {
            "train": len(train_pts),
            "validation": len(val_pts),
            "locked_test": len(test_pts),
            "total_unique": len(train_pts | val_pts | test_pts)
        }
    }
