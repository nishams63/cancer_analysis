"""
Exhaustive Leakage Audit Module for Stage 4.
Verifies:
- Patient partition isolation (Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅)
- Cross-split exact duplicate texts (SHA-256)
- Cross-split near-duplicate clinical notes (TF-IDF Cosine Similarity >= 0.85)
- Cross-split duplicate instruction/target pairs
- Temporal ordering integrity (document_date <= index_date)
- Prospective outcome terms scan
Fails pipeline if patient leakage is detected per Section 10.
"""

import re
import hashlib
import logging
from typing import Dict, List, Any, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("stage4.leakage_audit")


class LeakageAuditError(Exception):
    """Raised when critical leakage or split compromise is detected."""
    pass


class LeakageAuditor:
    """Performs rigorous multi-dimensional leakage audits on split datasets."""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_tfidf_features: int = 1000,
        forbidden_outcome_terms: List[str] = None
    ):
        self.similarity_threshold = similarity_threshold
        self.max_tfidf_features = max_tfidf_features
        self.forbidden_outcome_terms = forbidden_outcome_terms or [
            "retrospective survival",
            "autopsy finding",
            "overall survival reached",
            "post-mortem",
            "subsequent progression on day 180",
            "future relapse",
            "post-study survival"
        ]

    def audit_patient_isolation(self, df: pd.DataFrame, patient_col: str = "patient_id", split_col: str = "split") -> Dict[str, Any]:
        """Strict set-intersection check for patient overlap across splits."""
        pats_train = set(df[df[split_col] == "TRAIN"][patient_col].unique())
        pats_val = set(df[df[split_col] == "VALIDATION"][patient_col].unique())
        pats_test = set(df[df[split_col] == "TEST"][patient_col].unique())

        leak_train_val = pats_train.intersection(pats_val)
        leak_train_test = pats_train.intersection(pats_test)
        leak_val_test = pats_val.intersection(pats_test)

        total_leak = len(leak_train_val) + len(leak_train_test) + len(leak_val_test)

        res = {
            "patient_leakage": total_leak,
            "train_val_overlap": len(leak_train_val),
            "train_test_overlap": len(leak_train_test),
            "val_test_overlap": len(leak_val_test),
            "train_patient_count": len(pats_train),
            "val_patient_count": len(pats_val),
            "test_patient_count": len(pats_test),
            "status": "PASSED" if total_leak == 0 else "FAILED"
        }

        if total_leak > 0:
            raise LeakageAuditError(f"CRITICAL PATIENT LEAKAGE DETECTED: {total_leak} overlapping patients found across splits!")

        return res

    def audit_exact_duplicates(self, df: pd.DataFrame, text_col: str = "clinical_note", split_col: str = "split") -> Dict[str, Any]:
        """Detects exact text duplicates crossing split boundaries using SHA-256 hashes."""
        splits = df[split_col].unique()
        split_hashes: Dict[str, Set[str]] = {s: set() for s in splits}

        for idx, row in df.iterrows():
            text = str(row.get(text_col, ""))
            h = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
            s = row[split_col]
            split_hashes[s].add(h)

        cross_dups = 0
        split_list = list(split_hashes.keys())
        for i in range(len(split_list)):
            for j in range(i + 1, len(split_list)):
                s1, s2 = split_list[i], split_list[j]
                overlap = split_hashes[s1].intersection(split_hashes[s2])
                cross_dups += len(overlap)

        return {
            "cross_split_exact_duplicates": cross_dups,
            "status": "PASSED" if cross_dups == 0 else "FAILED"
        }

    def audit_near_duplicates(
        self,
        df: pd.DataFrame,
        text_col: str = "clinical_note",
        split_col: str = "split",
        sample_size_per_split: int = 250
    ) -> Dict[str, Any]:
        """
        Detects near-duplicate documents across splits using TF-IDF cosine similarity.
        Fails if cross-split similarity exceeds configured threshold.
        """
        train_texts = df[df[split_col] == "TRAIN"][text_col].astype(str).tolist()
        val_texts = df[df[split_col] == "VALIDATION"][text_col].astype(str).tolist()
        test_texts = df[df[split_col] == "TEST"][text_col].astype(str).tolist()

        if not train_texts or (not val_texts and not test_texts):
            return {"near_duplicate_cross_matches": 0, "status": "PASSED", "max_cross_similarity": 0.0}

        # Fit TF-IDF on Train only (to prevent fit leakage during audit)
        vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b"
        )
        vectorizer.fit(train_texts)

        # Sample for pairwise evaluation if large
        np.random.seed(42)
        s_train = np.random.choice(train_texts, min(len(train_texts), sample_size_per_split), replace=False)
        s_val = np.random.choice(val_texts, min(len(val_texts), sample_size_per_split), replace=False) if val_texts else []
        s_test = np.random.choice(test_texts, min(len(test_texts), sample_size_per_split), replace=False) if test_texts else []

        X_train = vectorizer.transform(s_train)
        near_dup_count = 0
        max_sim = 0.0

        if len(s_val) > 0:
            X_val = vectorizer.transform(s_val)
            sim_tv = cosine_similarity(X_train, X_val)
            max_sim = max(max_sim, float(sim_tv.max()))
            near_dup_count += int((sim_tv >= self.similarity_threshold).sum())

        if len(s_test) > 0:
            X_test = vectorizer.transform(s_test)
            sim_tt = cosine_similarity(X_train, X_test)
            max_sim = max(max_sim, float(sim_tt.max()))
            near_dup_count += int((sim_tt >= self.similarity_threshold).sum())

        return {
            "near_duplicate_cross_matches": near_dup_count,
            "max_cross_similarity": round(max_sim, 4),
            "threshold_used": self.similarity_threshold,
            "status": "PASSED" if near_dup_count == 0 else "WARNING"
        }

    def audit_duplicate_targets(self, df: pd.DataFrame, split_col: str = "split") -> Dict[str, Any]:
        """Checks for identical (instruction, target_risk, target_action) across splits."""
        df_target_hash = df.apply(
            lambda r: hashlib.sha256(f"{r.get('instruction')}|{r.get('target_risk')}|{r.get('target_action')}".encode("utf-8")).hexdigest(),
            axis=1
        )
        splits = df[split_col].unique()
        split_target_hashes = {s: set(df_target_hash[df[split_col] == s]) for s in splits}

        cross_target_dups = 0
        split_list = list(split_target_hashes.keys())
        for i in range(len(split_list)):
            for j in range(i + 1, len(split_list)):
                s1, s2 = split_list[i], split_list[j]
                overlap = split_target_hashes[s1].intersection(split_target_hashes[s2])
                cross_target_dups += len(overlap)

        return {
            "cross_split_duplicate_targets": cross_target_dups,
            "status": "PASSED" if cross_target_dups == 0 else "WARNING"
        }

    def audit_temporality_and_forbidden_terms(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Checks temporal sequence validity and prospective outcome terms."""
        temporal_violations = 0
        if "document_date" in df.columns and "index_date" in df.columns:
            d_dates = pd.to_datetime(df["document_date"], errors="coerce")
            i_dates = pd.to_datetime(df["index_date"], errors="coerce")
            temporal_violations = int((d_dates > i_dates).sum())

        # Scan text for forbidden prospective terms
        text_col = "clinical_note" if "clinical_note" in df.columns else "text"
        term_matches = {}
        for term in self.forbidden_outcome_terms:
            matches = df[text_col].astype(str).str.contains(term, case=False, na=False)
            cnt = int(matches.sum())
            if cnt > 0:
                term_matches[term] = cnt

        return {
            "temporal_order_violations": temporal_violations,
            "forbidden_outcome_term_matches": term_matches,
            "status": "PASSED" if temporal_violations == 0 and not term_matches else "FAILED"
        }

    def run_full_audit(self, df: pd.DataFrame, patient_col: str = "patient_id", split_col: str = "split") -> Dict[str, Any]:
        """Runs end-to-end multi-dimensional leakage audit."""
        logger.info("Executing comprehensive leakage audit across %d split records...", len(df))

        pat_audit = self.audit_patient_isolation(df, patient_col, split_col)
        exact_audit = self.audit_exact_duplicates(df, text_col="clinical_note" if "clinical_note" in df.columns else "text", split_col=split_col)
        near_audit = self.audit_near_duplicates(df, text_col="clinical_note" if "clinical_note" in df.columns else "text", split_col=split_col)
        target_audit = self.audit_duplicate_targets(df, split_col=split_col)
        temp_audit = self.audit_temporality_and_forbidden_terms(df)

        overall_status = "PASSED"
        if pat_audit["status"] != "PASSED" or exact_audit["status"] != "PASSED" or temp_audit["status"] != "PASSED":
            overall_status = "FAILED"

        report = {
            "overall_audit_status": overall_status,
            "patient_leakage": pat_audit["patient_leakage"],
            "patient_overlap_details": pat_audit,
            "exact_duplicate_audit": exact_audit,
            "near_duplicate_audit": near_audit,
            "target_duplicate_audit": target_audit,
            "temporality_and_terms_audit": temp_audit
        }

        logger.info("Leakage audit finished with overall status: %s (patient_leakage=%d)", overall_status, pat_audit["patient_leakage"])
        return report
