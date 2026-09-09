"""
Upgraded Feature Pipeline for Stage 3 Clinical NLP.
Combines whitespace-canonicalized text processing, contextual transformer embeddings,
and upgraded structured concept counts derived from the contextual NER model.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib

import sys
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from token_alignment import canonicalize_text
from contextual_concept_model import ContextualClinicalConceptModel


def extract_upgraded_concept_counts(
    df: pd.DataFrame,
    concept_model: ContextualClinicalConceptModel
) -> pd.DataFrame:
    """
    Extract deterministic structured counts from contextual transformer entity predictions:
    - word_count & char_count (on canonicalized text)
    - total_concepts, affirmed, negated, historical
    - drug, mutation, dosage, adverse event mentions
    - severity grades & high-risk symptom indicators
    """
    records = []
    for _, row in df.iterrows():
        raw_text = row.get("text", row.get("clean_narrative", ""))
        text = canonicalize_text(raw_text)

        # Extract entities using Contextual Transformer NER
        entities = concept_model.predict_entities(text, assign_polarity=True)

        affirmed = sum(1 for e in entities if e.get("polarity") == "AFFIRMED")
        negated = sum(1 for e in entities if e.get("polarity") == "NEGATED")
        historical = sum(1 for e in entities if e.get("polarity") == "HISTORICAL")

        drugs = sum(1 for e in entities if e["label"] == "DRUG_NAME")
        mutations = sum(1 for e in entities if e["label"] == "GENE_MUTATION")
        dosages = sum(1 for e in entities if e["label"] == "DOSAGE")
        adverse = sum(1 for e in entities if e["label"] == "ADVERSE_EVENT")

        t_lower = text.lower()
        has_grade_3_4 = int(bool(re.search(r"\bgrade\s*[34]\b", t_lower)))
        has_critical = int(bool(re.search(
            r"\b(?:dyspnea|shortness\s+of\s+breath|acute\s+adverse|nephrotoxicity|pneumonitis)\b",
            t_lower
        )))

        records.append({
            "document_id": row.get("document_id", f"doc_{_}"),
            "word_count": len(text.split()),
            "char_count": len(text),
            "total_concepts": len(entities),
            "affirmed_concepts": affirmed,
            "negated_concepts": negated,
            "historical_concepts": historical,
            "drug_mentions": drugs,
            "mutation_mentions": mutations,
            "dosage_mentions": dosages,
            "adverse_event_mentions": adverse,
            "has_grade_3_4": has_grade_3_4,
            "has_critical_symptom": has_critical
        })

    return pd.DataFrame(records)


class UpgradedClinicalFeaturePipeline:
    """
    Feature pipeline supporting Experiments A, B, C, D with whitespace canonicalization.
    """

    def __init__(self, max_tfidf_features: int = 1000):
        self.max_tfidf_features = max_tfidf_features
        self.tfidf = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95
        )
        self.scaler = StandardScaler()
        self.numeric_feature_cols = [
            "word_count", "char_count", "total_concepts",
            "affirmed_concepts", "negated_concepts", "historical_concepts",
            "drug_mentions", "mutation_mentions", "dosage_mentions",
            "adverse_event_mentions", "has_grade_3_4", "has_critical_symptom"
        ]
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame, struct_train_df: pd.DataFrame) -> "UpgradedClinicalFeaturePipeline":
        """Fit TF-IDF and Scaler strictly on canonicalized TRAIN texts and structured features."""
        canon_texts = [canonicalize_text(t) for t in df_train["text"]]
        if len(canon_texts) < 10:
            self.tfidf.set_params(min_df=1, max_df=1.0)
        self.tfidf.fit(canon_texts)

        X_num = struct_train_df[self.numeric_feature_cols].values
        self.scaler.fit(X_num)

        self.is_fitted = True
        return self

    def transform_experiment(
        self,
        df: pd.DataFrame,
        struct_df: pd.DataFrame,
        concept_model: ContextualClinicalConceptModel,
        experiment_type: str = "D"
    ) -> csr_matrix:
        """
        Generate feature matrices for Experiments A, B, C, D:
            - Exp B: 12-dim scaled structured features
            - Exp C: 396-dim (384-dim contextual embedding + 12-dim structured features)
            - Exp D: 1,396-dim Hybrid (1,000 TF-IDF + 384-dim contextual + 12-dim structured features)
        """
        if not self.is_fitted:
            raise RuntimeError("UpgradedClinicalFeaturePipeline must be fitted before transform!")

        # 1. Standardize numeric features
        X_num = struct_df[self.numeric_feature_cols].values
        X_num_scaled = self.scaler.transform(X_num)

        if experiment_type == "B":
            return csr_matrix(X_num_scaled)

        # Extract contextual document embeddings
        canon_texts = [canonicalize_text(t) for t in df["text"]]
        X_emb = concept_model.extract_contextual_embeddings(canon_texts)

        if experiment_type == "C":
            return hstack([csr_matrix(X_emb), csr_matrix(X_num_scaled)]).tocsr()

        elif experiment_type == "D":
            X_tfidf = self.tfidf.transform(canon_texts)
            return hstack([X_tfidf, csr_matrix(X_emb), csr_matrix(X_num_scaled)]).tocsr()

        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

    def save(self, output_dir: Path) -> None:
        """Save fitted TF-IDF vectorizer and scaler."""
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.tfidf, output_dir / "upgraded_tfidf_vectorizer.joblib")
        joblib.dump(self.scaler, output_dir / "upgraded_feature_scaler.joblib")
        joblib.dump(self.numeric_feature_cols, output_dir / "upgraded_numeric_cols.joblib")

    @classmethod
    def load(cls, input_dir: Path) -> "UpgradedClinicalFeaturePipeline":
        """Load fitted pipeline components."""
        pipeline = cls()
        pipeline.tfidf = joblib.load(input_dir / "upgraded_tfidf_vectorizer.joblib")
        pipeline.scaler = joblib.load(input_dir / "upgraded_feature_scaler.joblib")
        pipeline.numeric_feature_cols = joblib.load(input_dir / "upgraded_numeric_cols.joblib")
        pipeline.is_fitted = True
        return pipeline
