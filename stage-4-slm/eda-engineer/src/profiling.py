"""
Dataset Profiling Module for Stage 4 EDA.
Computes size statistics, cardinalities, per-column missingness, and duplicate detection.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def profile_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates overall structural and cardinality statistics for the dataset."""
    total_records = len(df)
    unique_patients = int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0
    unique_notes = int(df["note_id"].nunique()) if "note_id" in df.columns else 0

    records_per_patient = df["patient_id"].value_counts() if "patient_id" in df.columns else pd.Series([1])
    records_per_note = df["note_id"].value_counts() if "note_id" in df.columns else pd.Series([1])

    return {
        "total_records": int(total_records),
        "unique_patients": unique_patients,
        "unique_notes": unique_notes,
        "patient_cardinality_stats": {
            "min": int(records_per_patient.min()),
            "max": int(records_per_patient.max()),
            "mean": float(round(records_per_patient.mean(), 2)),
            "median": float(round(records_per_patient.median(), 2)),
            "std": float(round(records_per_patient.std(), 2))
        },
        "note_cardinality_stats": {
            "min": int(records_per_note.min()),
            "max": int(records_per_note.max()),
            "mean": float(round(records_per_note.mean(), 2))
        }
    }


def analyze_missingness(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes column-by-column missingness counts and percentages."""
    total = len(df)
    missing_by_col = {}
    completely_missing_cols = []
    high_missing_cols = []

    for col in df.columns:
        # For list/array columns, nulls might be None or NaN
        is_null_mask = df[col].isna()
        missing_count = int(is_null_mask.sum())
        missing_pct = float(round((missing_count / total) * 100.0, 4)) if total > 0 else 0.0

        missing_by_col[col] = {
            "missing_count": missing_count,
            "missing_percentage": missing_pct
        }

        if missing_count == total:
            completely_missing_cols.append(col)
        elif missing_pct > 5.0:
            high_missing_cols.append(col)

    return {
        "columns": missing_by_col,
        "completely_missing_columns": completely_missing_cols,
        "high_missing_columns": high_missing_cols,
        "total_missing_cells": int(sum(m["missing_count"] for m in missing_by_col.values()))
    }


def analyze_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyzes exact row duplicates, note duplicates, and target duplicates without dropping them."""
    total = len(df)

    # 1. Exact row duplicates across all columns (handling unhashable list columns)
    temp_df = df.copy()
    for col in temp_df.columns:
        if len(temp_df) > 0 and isinstance(temp_df[col].iloc[0], (list, np.ndarray)):
            temp_df[col] = temp_df[col].apply(lambda x: str(list(x)) if isinstance(x, (list, np.ndarray)) else str(x))
    exact_row_dups = int(temp_df.duplicated().sum())

    # 2. Duplicate clinical notes
    note_dups = int(df.duplicated(subset=["clinical_note"]).sum()) if "clinical_note" in df.columns else 0

    # 3. Duplicate patient_id + note_id pairs
    patient_doc_dups = (
        int(df.duplicated(subset=["patient_id", "note_id"]).sum())
        if "patient_id" in df.columns and "note_id" in df.columns
        else 0
    )

    # 4. Duplicate target outputs
    target_cols = [c for c in ["target_risk", "target_key_finding", "target_action"] if c in df.columns]
    target_dups = int(df.duplicated(subset=target_cols).sum()) if target_cols else 0

    return {
        "exact_row_duplicates": exact_row_dups,
        "duplicate_clinical_notes": note_dups,
        "duplicate_patient_doc_pairs": patient_doc_dups,
        "duplicate_target_triplets": target_dups,
        "exact_duplicate_rate": float(round((exact_row_dups / total), 6)) if total > 0 else 0.0,
        "note_duplicate_rate": float(round((note_dups / total), 6)) if total > 0 else 0.0
    }
