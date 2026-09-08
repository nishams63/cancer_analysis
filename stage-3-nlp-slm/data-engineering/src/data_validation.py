"""
Comprehensive Data Quality Validation & Audit Module for Stage 3 Clinical NLP.
Implements schema verification, record de-duplication, text health checks,
leakage detection, temporal validation, and patient split integrity.
"""

from typing import Dict, List, Any, Tuple, Set
import pandas as pd
import numpy as np

from config import (
    VALID_DOC_TYPES,
    VALID_URGENCY_LEVELS,
    VALID_HAZARD_TYPES,
    CFG
)
from de_identification import verify_no_direct_identifiers

class DataValidationError(Exception):
    """Raised when critical clinical data engineering checks fail."""
    pass

def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate presence and types of mandatory schema fields."""
    mandatory_cols = [
        "document_id",
        "patient_id",
        "encounter_id",
        "document_type",
        "document_date",
        "index_date",
        "text",
        "urgency_level",
        "hazard_type"
    ]
    missing = [c for c in mandatory_cols if c not in df.columns]
    if missing:
        raise DataValidationError(f"CRITICAL: Mandatory schema columns missing: {missing}")
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        raise DataValidationError("CRITICAL: Duplicate column names detected in dataframe!")

    # Check nulls in mandatory fields
    null_counts = {c: int(df[c].isnull().sum()) for c in mandatory_cols}
    has_nulls = {c: count for c, count in null_counts.items() if count > 0}
    
    return {
        "status": "PASSED" if not has_nulls else "FAILED",
        "total_columns": len(df.columns),
        "mandatory_columns_verified": mandatory_cols,
        "null_counts": null_counts,
        "critical_null_violations": has_nulls
    }

def audit_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Identify duplicate rows, duplicate texts, and composite key collisions."""
    exact_row_dups = int(df.duplicated().sum())
    exact_text_dups = int(df.duplicated(subset=["text"]).sum())
    key_dups = int(df.duplicated(subset=["patient_id", "encounter_id", "document_type"]).sum())
    id_dups = int(df.duplicated(subset=["document_id"]).sum())
    
    return {
        "exact_row_duplicates": exact_row_dups,
        "exact_text_duplicates": exact_text_dups,
        "composite_key_duplicates": key_dups,
        "document_id_duplicates": id_dups
    }

def audit_text_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Audit text cleanliness, lengths, and anomalies."""
    min_words = CFG["quality_thresholds"]["min_word_count"]
    max_words = CFG["quality_thresholds"]["max_word_count"]
    
    word_counts = df["text"].astype(str).apply(lambda t: len(t.split()))
    char_counts = df["text"].astype(str).apply(len)
    
    empty_count = int((df["text"].str.strip() == "").sum())
    null_count = int(df["text"].isnull().sum())
    short_count = int((word_counts < min_words).sum())
    long_count = int((word_counts > max_words).sum())
    
    # Unprintable characters
    has_control_chars = int(df["text"].astype(str).str.contains(r'[\x00-\x08\x0b\x0c\x0e-\x1f]').sum())
    
    return {
        "total_documents": len(df),
        "empty_text_count": empty_count,
        "null_text_count": null_count,
        "short_text_count": short_count,
        "long_text_count": long_count,
        "control_character_violations": has_control_chars,
        "word_count_stats": {
            "mean": float(word_counts.mean()),
            "median": float(word_counts.median()),
            "std": float(word_counts.std()),
            "min": int(word_counts.min()),
            "max": int(word_counts.max())
        },
        "char_count_stats": {
            "mean": float(char_counts.mean()),
            "median": float(char_counts.median()),
            "std": float(char_counts.std()),
            "min": int(char_counts.min()),
            "max": int(char_counts.max())
        }
    }

def audit_leakage_and_temporality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check for:
    1. Temporal violations: document_date > index_date
    2. Forbidden future outcome keywords in text
    """
    forbidden_terms = CFG["leakage_control"]["forbidden_outcome_terms"]
    
    # Temporal ordering check
    doc_dates = pd.to_datetime(df["document_date"], errors="coerce")
    idx_dates = pd.to_datetime(df["index_date"], errors="coerce")
    
    temporal_violations = int((doc_dates > idx_dates).sum())
    
    # Text keyword scan
    term_matches = {}
    total_leaking_notes = 0
    leaking_indices = set()
    
    for term in forbidden_terms:
        matches = df["text"].str.contains(term, case=False, na=False)
        m_count = int(matches.sum())
        term_matches[term] = m_count
        if m_count > 0:
            leaking_indices.update(df[matches].index.tolist())
            
    total_leaking_notes = len(leaking_indices)
    
    return {
        "temporal_order_violations": temporal_violations,
        "keyword_leakage_matches": term_matches,
        "total_leaking_notes_detected": total_leaking_notes,
        "leaking_indices": list(leaking_indices)
    }

def audit_privacy(df: pd.DataFrame) -> Dict[str, Any]:
    """Verify that no unmasked direct identifiers exist in text."""
    violations_found = []
    for idx, text in enumerate(df["text"]):
        v = verify_no_direct_identifiers(text)
        if v:
            violations_found.append({"index": idx, "violations": v})
            
    return {
        "status": "PASSED" if not violations_found else "FLAGGED",
        "total_violations": len(violations_found),
        "details": violations_found[:10]  # first 10 for review
    }

def audit_labels(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate target class labels and distributions."""
    invalid_doc_types = set(df["document_type"].unique()) - VALID_DOC_TYPES
    invalid_urgencies = set(df["urgency_level"].unique()) - VALID_URGENCY_LEVELS
    invalid_hazards = set(df["hazard_type"].unique()) - VALID_HAZARD_TYPES
    
    urgency_dist = df["urgency_level"].value_counts().to_dict()
    hazard_dist = df["hazard_type"].value_counts().to_dict()
    doc_type_dist = df["document_type"].value_counts().to_dict()
    
    return {
        "invalid_document_types": list(invalid_doc_types),
        "invalid_urgency_levels": list(invalid_urgencies),
        "invalid_hazard_types": list(invalid_hazards),
        "urgency_distribution": urgency_dist,
        "hazard_distribution": hazard_dist,
        "document_type_distribution": doc_type_dist
    }

def validate_split_integrity(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Strictly verify zero patient leakage and zero encounter leakage across data splits.
    """
    train_pats = set(train_df["patient_id"].unique())
    val_pats = set(val_df["patient_id"].unique())
    test_pats = set(test_df["patient_id"].unique())
    
    train_encs = set(train_df["encounter_id"].unique())
    val_encs = set(val_df["encounter_id"].unique())
    test_encs = set(test_df["encounter_id"].unique())
    
    train_val_pat_leak = train_pats.intersection(val_pats)
    train_test_pat_leak = train_pats.intersection(test_pats)
    val_test_pat_leak = val_pats.intersection(test_pats)
    
    train_val_enc_leak = train_encs.intersection(val_encs)
    train_test_enc_leak = train_encs.intersection(test_encs)
    val_test_enc_leak = val_encs.intersection(test_encs)
    
    total_pat_leak = len(train_val_pat_leak) + len(train_test_pat_leak) + len(val_test_pat_leak)
    total_enc_leak = len(train_val_enc_leak) + len(train_test_enc_leak) + len(val_test_enc_leak)
    
    if total_pat_leak > 0:
        raise DataValidationError(f"CRITICAL PATIENT LEAKAGE: Overlapping patients detected between splits! Count: {total_pat_leak}")
        
    if total_enc_leak > 0:
        raise DataValidationError(f"CRITICAL ENCOUNTER LEAKAGE: Overlapping encounters detected between splits! Count: {total_enc_leak}")
        
    return {
        "status": "PASSED (0% LEAKAGE)",
        "patient_overlap": {
            "train_val": len(train_val_pat_leak),
            "train_test": len(train_test_pat_leak),
            "val_test": len(val_test_pat_leak)
        },
        "encounter_overlap": {
            "train_val": len(train_val_enc_leak),
            "train_test": len(train_test_enc_leak),
            "val_test": len(val_test_enc_leak)
        },
        "patient_counts": {
            "train": len(train_pats),
            "val": len(val_pats),
            "test": len(test_pats),
            "total_unique": len(train_pats | val_pats | test_pats)
        },
        "document_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total_documents": len(train_df) + len(val_df) + len(test_df)
        }
    }
