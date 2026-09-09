"""
Split Verification and Independent Data Leakage Audit Module for Stage 4 EDA.
Verifies zero patient leakage, audits cross-split exact and near duplicates,
and checks for target/metadata contamination per Sections 15 & 16.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SplitAndLeakageAuditor:
    """Performs split integrity checks and comprehensive data leakage audits."""

    def __init__(self, near_dup_threshold: float = 0.85, max_tfidf_features: int = 5000):
        self.near_dup_threshold = near_dup_threshold
        self.max_tfidf_features = max_tfidf_features

    def verify_patient_isolation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Verifies that no patient appears across multiple splits:
        Patients(Train) ∩ Patients(Val) = ∅
        Patients(Train) ∩ Patients(Test) = ∅
        Patients(Val) ∩ Patients(Test) = ∅
        """
        splits = df["split"].unique()
        patient_sets: Dict[str, Set[str]] = {
            s: set(df[df["split"] == s]["patient_id"].dropna()) for s in splits
        }

        overlaps = {}
        split_names = list(patient_sets.keys())
        total_overlap_patients = 0

        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                inter = patient_sets[s1].intersection(patient_sets[s2])
                overlap_count = len(inter)
                total_overlap_patients += overlap_count
                overlaps[f"{s1}_vs_{s2}"] = {
                    "overlap_count": overlap_count,
                    "overlapping_patient_sample": list(inter)[:5]
                }

        is_isolated = total_overlap_patients == 0

        split_counts = df["split"].value_counts().to_dict()
        total_records = len(df)
        split_distribution = {
            s: {
                "records": cnt,
                "percentage": float(round((cnt / total_records) * 100.0, 2)),
                "unique_patients": len(patient_sets.get(s, set()))
            }
            for s, cnt in split_counts.items()
        }

        return {
            "is_patient_isolated": is_isolated,
            "total_patient_leakage": total_overlap_patients,
            "pairwise_overlaps": overlaps,
            "split_distribution": split_distribution
        }

    def audit_cross_split_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Audits exact note duplicates across splits using cryptographic SHA-256 hashes.
        """
        splits = df["split"].unique()
        note_hashes = {}
        for s in splits:
            subset = df[df["split"] == s]
            hashes = {}
            for idx, text in zip(subset["note_id"], subset["clinical_note"]):
                h = hashlib.sha256(str(text).strip().encode("utf-8")).hexdigest()
                hashes[idx] = h
            note_hashes[s] = hashes

        cross_split_exact_dups = []
        split_names = list(note_hashes.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                h1_map = {v: k for k, v in note_hashes[s1].items()}
                for doc2, h2 in note_hashes[s2].items():
                    if h2 in h1_map:
                        cross_split_exact_dups.append({
                            "split_1": s1,
                            "doc_1": h1_map[h2],
                            "split_2": s2,
                            "doc_2": doc2,
                            "hash": h2
                        })

        return {
            "cross_split_exact_duplicate_count": len(cross_split_exact_dups),
            "cross_split_exact_duplicates": cross_split_exact_dups
        }

    def audit_near_duplicates_cross_split(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Audits near-duplicate notes across Train vs Val, Train vs Test using TF-IDF cosine similarity.
        Scalable sampling-based audit across split boundaries.
        """
        train_df = df[df["split"] == "TRAIN"]
        val_df = df[df["split"] == "VALIDATION"]
        test_df = df[df["split"] == "TEST"]

        if len(train_df) == 0 or (len(val_df) == 0 and len(test_df) == 0):
            return {"near_duplicate_count": 0, "pairs": []}

        # Fit TF-IDF on corpus
        tfidf = TfidfVectorizer(max_features=self.max_tfidf_features, stop_words="english")
        tfidf.fit(df["clinical_note"].fillna(""))

        train_vecs = tfidf.transform(train_df["clinical_note"].fillna(""))
        train_ids = train_df["note_id"].values

        near_dup_pairs = []

        # Audit Train vs Validation
        if len(val_df) > 0:
            val_vecs = tfidf.transform(val_df["clinical_note"].fillna(""))
            val_ids = val_df["note_id"].values
            sim_matrix_val = cosine_similarity(val_vecs, train_vecs)
            val_dups = np.argwhere(sim_matrix_val >= self.near_dup_threshold)
            for r, c in val_dups:
                near_dup_pairs.append({
                    "split_pair": "VALIDATION_vs_TRAIN",
                    "doc_1": str(val_ids[r]),
                    "doc_2": str(train_ids[c]),
                    "similarity": float(round(float(sim_matrix_val[r, c]), 4))
                })

        # Audit Train vs Test
        if len(test_df) > 0:
            test_vecs = tfidf.transform(test_df["clinical_note"].fillna(""))
            test_ids = test_df["note_id"].values
            sim_matrix_test = cosine_similarity(test_vecs, train_vecs)
            test_dups = np.argwhere(sim_matrix_test >= self.near_dup_threshold)
            for r, c in test_dups:
                near_dup_pairs.append({
                    "split_pair": "TEST_vs_TRAIN",
                    "doc_1": str(test_ids[r]),
                    "doc_2": str(train_ids[c]),
                    "similarity": float(round(float(sim_matrix_test[r, c]), 4))
                })

        return {
            "similarity_threshold": self.near_dup_threshold,
            "near_duplicate_pair_count": len(near_dup_pairs),
            "sample_near_duplicate_pairs": near_dup_pairs[:10]
        }

    def audit_metadata_and_target_leakage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Audits whether internal identifiers (patient_id, note_id) leak into target text.
        """
        id_leakage_cases = []
        target_corpus = (
            df["target_risk"].fillna("")
            + " "
            + df["target_key_finding"].fillna("")
            + " "
            + df["target_action"].fillna("")
        )

        for idx, row in df.iterrows():
            pid = str(row.get("patient_id", ""))
            nid = str(row.get("note_id", ""))
            t_text = target_corpus.iloc[idx]

            if pid and pid in t_text:
                id_leakage_cases.append({"row": idx, "leaked_id": pid, "type": "PATIENT_ID"})
            if nid and nid in t_text:
                id_leakage_cases.append({"row": idx, "leaked_id": nid, "type": "NOTE_ID"})

        return {
            "metadata_id_leakage_count": len(id_leakage_cases),
            "leakage_cases": id_leakage_cases
        }

    def run_full_leakage_audit(self, df: pd.DataFrame, output_json_path: Optional[Path] = None) -> Dict[str, Any]:
        """Executes all leakage dimensions and optionally saves reports/leakage_eda.json."""
        patient_iso = self.verify_patient_isolation(df)
        exact_dups = self.audit_cross_split_duplicates(df)
        near_dups = self.audit_near_duplicates_cross_split(df)
        meta_leakage = self.audit_metadata_and_target_leakage(df)

        has_critical_leakage = (
            patient_iso["total_patient_leakage"] > 0
            or exact_dups["cross_split_exact_duplicate_count"] > 0
            or meta_leakage["metadata_id_leakage_count"] > 0
        )

        status = "CRITICAL" if has_critical_leakage else "PASS"

        report = {
            "status": status,
            "patient_isolation": patient_iso,
            "cross_split_exact_duplicates": exact_dups,
            "cross_split_near_duplicates": near_dups,
            "metadata_id_leakage": meta_leakage
        }

        if output_json_path:
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        return report
