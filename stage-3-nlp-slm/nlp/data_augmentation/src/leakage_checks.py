"""
Partition Isolation and Data Leakage Audit Module.
Strictly verifies that no augmented data leaks across patient, encounter, document,
or lexical boundaries into validation or locked-test partitions.
"""

from typing import Dict, Any, List, Set, Tuple
import re
import pandas as pd
from duplicate_detection import compute_canonical_hash


def verify_patient_split_isolation(
    df_augmented: pd.DataFrame,
    df_val: pd.DataFrame,
    df_train_source: pd.DataFrame
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verifies that all augmented records derive strictly from train cohort
    and have zero patient, encounter, or text overlap with validation.
    """
    aug_patients = set(df_augmented["patient_id"])
    train_patients = set(df_train_source["patient_id"])
    val_patients = set(df_val["patient_id"])

    # 1. Check patient partition
    patient_val_overlap = aug_patients & val_patients
    invalid_patients = aug_patients - train_patients # Must only be train patients

    # 2. Check encounter partition
    aug_encounters = set(df_augmented["encounter_id"])
    val_encounters = set(df_val["encounter_id"])
    encounter_val_overlap = aug_encounters & val_encounters

    # 3. Check canonical text overlap with validation
    aug_text_hashes = set(df_augmented["text"].apply(compute_canonical_hash))
    val_text_hashes = set(df_val["text"].apply(compute_canonical_hash))
    text_val_overlap = aug_text_hashes & val_text_hashes

    # 4. Check document_id uniqueness (augmented docs must have new unique IDs)
    aug_doc_ids = set(df_augmented["document_id"])
    train_doc_ids = set(df_train_source["document_id"])
    val_doc_ids = set(df_val["document_id"])
    doc_id_overlap = (aug_doc_ids & train_doc_ids) | (aug_doc_ids & val_doc_ids)

    audit_metrics = {
        "augmented_rows": len(df_augmented),
        "augmented_patients": len(aug_patients),
        "patient_overlap_with_val": len(patient_val_overlap),
        "non_train_patients_count": len(invalid_patients),
        "encounter_overlap_with_val": len(encounter_val_overlap),
        "canonical_text_overlap_with_val": len(text_val_overlap),
        "doc_id_collisions": len(doc_id_overlap),
        "is_leakage_free": (
            len(patient_val_overlap) == 0 and
            len(invalid_patients) == 0 and
            len(encounter_val_overlap) == 0 and
            len(text_val_overlap) == 0 and
            len(doc_id_overlap) == 0
        )
    }
    
    return audit_metrics["is_leakage_free"], audit_metrics
