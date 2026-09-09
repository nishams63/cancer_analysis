"""
Clinical Feature Extraction Module for Stage 3 NLP.
Extracts negation-scoped TF-IDF representations and structured clinical concept counts.
Fitted EXCLUSIVELY on the TRAIN partition.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib

from config import TOKENIZERS_DIR, OUTPUTS_DIR
from sentence_processing import split_into_sentences
from negation_detection import resolve_concept_polarity
from clinical_concepts import extract_clinical_concepts


def create_negation_scoped_text(text: str) -> str:
    """
    Transform text so that tokens governed by negation receive prefix 'neg_',
    and tokens governed by historical context receive prefix 'hist_'.
    Prevents Bag-of-Words and TF-IDF models from treating 'no fever' as 'fever'.
    """
    sentences = split_into_sentences(text)
    transformed_sentences = []

    for sent_dict in sentences:
        sent_text = sent_dict["text"]
        words = sent_text.split()
        transformed_words = []
        for i, word in enumerate(words):
            # Check if this word is governed by negation
            w_start = sent_text.find(word)
            w_end = w_start + len(word) if w_start != -1 else 0
            polarity = resolve_concept_polarity(sent_text, w_start, w_end)
            
            clean_word = re.sub(r"[^\w\-]", "", word.lower())
            if not clean_word:
                continue
            if polarity == "NEGATED":
                transformed_words.append(f"neg_{clean_word}")
            elif polarity == "HISTORICAL":
                transformed_words.append(f"hist_{clean_word}")
            else:
                transformed_words.append(clean_word)
        transformed_sentences.append(" ".join(transformed_words))

    return " ".join(transformed_sentences)


def extract_structured_concept_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract deterministic structured counts and clinical indicators from each document:
    - total concepts, affirmed, negated, historical
    - drug count, mutation count, dosage count
    - severity grades, high-risk symptom indicators
    """
    records = []
    for _, row in df.iterrows():
        text = row["text"]
        concepts = extract_clinical_concepts(text, assign_polarity=True)

        affirmed = sum(1 for c in concepts if c.get("polarity") == "AFFIRMED")
        negated = sum(1 for c in concepts if c.get("polarity") == "NEGATED")
        historical = sum(1 for c in concepts if c.get("polarity") == "HISTORICAL")

        drugs = sum(1 for c in concepts if c["label"] == "DRUG_NAME")
        mutations = sum(1 for c in concepts if c["label"] == "GENE_MUTATION")
        dosages = sum(1 for c in concepts if c["label"] == "DOSAGE")
        adverse = sum(1 for c in concepts if c["label"] == "ADVERSE_EVENT")

        # Clinical rule indicators
        t_lower = text.lower()
        has_grade_3_4 = int(bool(re.search(r"\bgrade\s*[34]\b", t_lower)))
        has_critical = int(bool(re.search(r"\b(?:dyspnea|shortness\s+of\s+breath|acute\s+adverse|nephrotoxicity|pneumonitis)\b", t_lower)))

        records.append({
            "document_id": row["document_id"],
            "word_count": len(text.split()),
            "char_count": len(text),
            "total_concepts": len(concepts),
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


class ClinicalFeaturePipeline:
    """End-to-end feature extraction pipeline fitted ONLY on TRAIN data."""

    def __init__(self, max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
            max_df=0.95
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.numeric_feature_cols = [
            "word_count", "char_count", "total_concepts",
            "affirmed_concepts", "negated_concepts", "historical_concepts",
            "drug_mentions", "mutation_mentions", "dosage_mentions",
            "adverse_event_mentions", "has_grade_3_4", "has_critical_symptom"
        ]

    def fit(self, df_train: pd.DataFrame) -> "ClinicalFeaturePipeline":
        """Fit TF-IDF and Scaler strictly on TRAIN texts and structured features."""
        # 1. Prepare negation-scoped texts for TRAIN
        train_scoped = [create_negation_scoped_text(t) for t in df_train["text"]]
        if len(train_scoped) < 10:
            self.tfidf.set_params(min_df=1, max_df=1.0)
        self.tfidf.fit(train_scoped)

        # 2. Prepare structured count features for TRAIN
        struct_df = extract_structured_concept_features(df_train)
        X_num = struct_df[self.numeric_feature_cols].values
        self.scaler.fit(X_num)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[csr_matrix, pd.DataFrame]:
        """Transform text into combined (TF-IDF + Scaled Numeric) feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("ClinicalFeaturePipeline must be fitted on TRAIN before transform!")

        # 1. Transform negation-scoped text
        scoped_texts = [create_negation_scoped_text(t) for t in df["text"]]
        X_tfidf = self.tfidf.transform(scoped_texts)

        # 2. Extract and scale structured features
        struct_df = extract_structured_concept_features(df)
        X_num = struct_df[self.numeric_feature_cols].values
        X_num_scaled = self.scaler.transform(X_num)

        # 3. Stack sparse TF-IDF with numeric features
        X_combined = hstack([X_tfidf, csr_matrix(X_num_scaled)]).tocsr()
        return X_combined, struct_df

    def save(self, output_dir=TOKENIZERS_DIR) -> None:
        """Save fitted vectorizer, scaler, and column definitions."""
        joblib.dump(self.tfidf, output_dir / "tfidf_vectorizer.joblib")
        joblib.dump(self.scaler, output_dir / "feature_scaler.joblib")
        joblib.dump(self.numeric_feature_cols, output_dir / "numeric_feature_cols.joblib")

    @classmethod
    def load(cls, input_dir=TOKENIZERS_DIR) -> "ClinicalFeaturePipeline":
        """Load pre-fitted vectorizer and scaler."""
        pipeline = cls()
        pipeline.tfidf = joblib.load(input_dir / "tfidf_vectorizer.joblib")
        pipeline.scaler = joblib.load(input_dir / "feature_scaler.joblib")
        pipeline.numeric_feature_cols = joblib.load(input_dir / "numeric_feature_cols.joblib")
        pipeline.is_fitted = True
        return pipeline
