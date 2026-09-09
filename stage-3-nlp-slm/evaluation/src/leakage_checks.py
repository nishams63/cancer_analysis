"""
Leakage Verification Module for Stage 3 Clinical NLP Evaluation.
Independently verifies patient isolation, encounter separation, document uniqueness,
temporal ordering, prospective phrase absence, and cryptographic file invariance.
"""

from typing import Dict, Any, Set, List
import hashlib
from pathlib import Path
import re
import pandas as pd

from config import (
    TRAIN_PARQUET_PATH,
    VAL_PARQUET_PATH,
    LOCKED_TEST_PARQUET_PATH,
    RAW_PARQUET_PATH,
    EXPECTED_SHA256
)


def compute_file_sha256(file_path: Path) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_upstream_file_hashes() -> Dict[str, Any]:
    """Verify that all upstream Data Engineering processed files match expected hashes."""
    results = {}
    all_matched = True

    files_to_check = {
        "clinical_nlp_dataset_v1.parquet": RAW_PARQUET_PATH,
        "train.parquet": TRAIN_PARQUET_PATH,
        "validation.parquet": VAL_PARQUET_PATH,
        "locked_test.parquet": LOCKED_TEST_PARQUET_PATH
    }

    for name, path in files_to_check.items():
        if not path.exists():
            results[name] = {"status": "MISSING", "matched": False}
            all_matched = False
            continue

        computed = compute_file_sha256(path)
        expected = EXPECTED_SHA256.get(name, "")
        is_match = (computed == expected)
        if not is_match:
            all_matched = False

        results[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "computed_sha256": computed,
            "matched": is_match
        }

    return {
        "all_files_invariant": all_matched,
        "details": results
    }


def verify_split_isolation(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame
) -> Dict[str, Any]:
    """
    Independently verify patient, encounter, and document ID disjointness across splits.
    """
    pts_tr = set(df_train["patient_id"])
    pts_val = set(df_val["patient_id"])
    pts_te = set(df_test["patient_id"])

    enc_tr = set(df_train["encounter_id"])
    enc_val = set(df_val["encounter_id"])
    enc_te = set(df_test["encounter_id"])

    doc_tr = set(df_train["document_id"])
    doc_val = set(df_val["document_id"])
    doc_te = set(df_test["document_id"])

    pt_tr_val = len(pts_tr & pts_val)
    pt_tr_te = len(pts_tr & pts_te)
    pt_val_te = len(pts_val & pts_te)

    enc_tr_val = len(enc_tr & enc_val)
    enc_tr_te = len(enc_tr & enc_te)
    enc_val_te = len(enc_val & enc_te)

    doc_tr_val = len(doc_tr & doc_val)
    doc_tr_te = len(doc_tr & doc_te)
    doc_val_te = len(doc_val & doc_te)

    is_completely_isolated = (
        pt_tr_val == 0 and pt_tr_te == 0 and pt_val_te == 0 and
        enc_tr_val == 0 and enc_tr_te == 0 and enc_val_te == 0 and
        doc_tr_val == 0 and doc_tr_te == 0 and doc_val_te == 0
    )

    return {
        "is_completely_isolated": is_completely_isolated,
        "patient_overlap": {
            "train_val": pt_tr_val,
            "train_test": pt_tr_te,
            "val_test": pt_val_te
        },
        "encounter_overlap": {
            "train_val": enc_tr_val,
            "train_test": enc_tr_te,
            "val_test": enc_val_te
        },
        "document_overlap": {
            "train_val": doc_tr_val,
            "train_test": doc_tr_te,
            "val_test": doc_val_te
        },
        "unique_patients": {
            "train": len(pts_tr),
            "validation": len(pts_val),
            "locked_test": len(pts_te),
            "total_distinct": len(pts_tr | pts_val | pts_te)
        }
    }


def verify_text_duplicates_across_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame
) -> Dict[str, Any]:
    """Check for identical clinical narratives appearing across split boundaries."""
    tr_hashes = {hashlib.sha256(t.encode("utf-8")).hexdigest() for t in df_train["text"]}
    val_hashes = {hashlib.sha256(t.encode("utf-8")).hexdigest() for t in df_val["text"]}
    te_hashes = {hashlib.sha256(t.encode("utf-8")).hexdigest() for t in df_test["text"]}

    tr_val_dups = len(tr_hashes & val_hashes)
    tr_te_dups = len(tr_hashes & te_hashes)
    val_te_dups = len(val_hashes & te_hashes)

    return {
        "cross_split_duplicates_detected": (tr_val_dups > 0 or tr_te_dups > 0 or val_te_dups > 0),
        "train_val_exact_text_duplicates": tr_val_dups,
        "train_test_exact_text_duplicates": tr_te_dups,
        "val_test_exact_text_duplicates": val_te_dups
    }


def verify_temporal_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """Verify document_date <= index_date across all documents in dataset."""
    doc_dates = pd.to_datetime(df["document_date"])
    idx_dates = pd.to_datetime(df["index_date"])

    future_inversions = int((doc_dates > idx_dates).sum())
    return {
        "total_documents_checked": len(df),
        "future_dated_inversions": future_inversions,
        "temporal_ordering_valid": (future_inversions == 0)
    }


def verify_target_leakage_terms(df: pd.DataFrame) -> Dict[str, Any]:
    """Scan texts for prospective outcome phrases that would indicate downstream target leakage."""
    forbidden_terms = [
        r"\bretrospective\s+survival\b",
        r"\bautopsy\b",
        r"\bpost-mortem\b",
        r"\bprogression\s+on\s+day\b",
        r"\bsubsequent\s+progression\b",
        r"\bfuture\s+relapse\b",
        r"\boverall\s+survival\s+reached\b"
    ]

    matches_found = 0
    match_details = []

    for _, row in df.iterrows():
        t = row["text"].lower()
        for pat in forbidden_terms:
            if re.search(pat, t):
                matches_found += 1
                match_details.append({
                    "document_id": row["document_id"],
                    "pattern": pat
                })

    return {
        "documents_scanned": len(df),
        "forbidden_outcome_matches": matches_found,
        "target_leakage_free": (matches_found == 0),
        "matches": match_details
    }
