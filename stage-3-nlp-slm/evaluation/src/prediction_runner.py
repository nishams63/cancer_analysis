"""
Prediction Runner for Stage 3 Clinical NLP Evaluation.
Executes batch inference using frozen NLP models and artifacts,
producing structured predictions and probability arrays.
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from config import NLP_ROOT_DIR, RESULTS_PRED_DIR
from model_loader import FrozenNLPArtifacts, load_frozen_artifacts

# Ensure NLP src is importable for feature extraction logic
nlp_src_dir = (NLP_ROOT_DIR / "src").resolve()
if str(nlp_src_dir) not in sys.path:
    sys.path.append(str(nlp_src_dir))

from feature_extraction import create_negation_scoped_text, extract_structured_concept_features


def extract_features(df: pd.DataFrame, artifacts: FrozenNLPArtifacts) -> Tuple[csr_matrix, pd.DataFrame]:
    """Transform text using frozen TF-IDF vectorizer and frozen numeric scaler."""
    scoped_texts = [create_negation_scoped_text(t) for t in df["text"]]
    X_tfidf = artifacts.tfidf_vectorizer.transform(scoped_texts)

    struct_df = extract_structured_concept_features(df)
    X_num = struct_df[artifacts.numeric_feature_cols].values
    X_num_scaled = artifacts.feature_scaler.transform(X_num)

    X_combined = hstack([X_tfidf, csr_matrix(X_num_scaled)]).tocsr()
    return X_combined, struct_df


def run_predictions(
    df: pd.DataFrame,
    artifacts: Optional[FrozenNLPArtifacts] = None
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, csr_matrix, pd.DataFrame]:
    """
    Run end-to-end inference on a clinical dataframe using frozen artifacts.
    Returns:
        pred_df: DataFrame with predictions, confidences, and ground truth
        urg_probs: Array of urgency class probabilities (N x 4)
        haz_probs: Array of hazard class probabilities (N x 8)
        X_features: Combined feature matrix (N x 1012)
        struct_df: Extracted structured feature counts
    """
    if artifacts is None:
        artifacts = load_frozen_artifacts()

    X_features, struct_df = extract_features(df, artifacts)

    # 1. Urgency Predictions & Probabilities
    urg_encoded_preds = artifacts.urgency_model.predict(X_features)
    urg_probs = artifacts.urgency_model.predict_proba(X_features)
    urg_preds = artifacts.urgency_encoder.inverse_transform(urg_encoded_preds)
    urg_classes = list(artifacts.urgency_encoder.classes_)

    # 2. Hazard Predictions & Probabilities
    haz_encoded_preds = artifacts.hazard_model.predict(X_features)
    haz_probs = artifacts.hazard_model.predict_proba(X_features)
    haz_preds = artifacts.hazard_encoder.inverse_transform(haz_encoded_preds)
    haz_classes = list(artifacts.hazard_encoder.classes_)

    pred_records = {
        "document_id": df["document_id"].values,
        "patient_id": df["patient_id"].values,
        "encounter_id": df["encounter_id"].values,
        "document_type": df["document_type"].values,
        "ground_truth_urgency": df["urgency_level"].values,
        "predicted_urgency": urg_preds,
        "urgency_confidence": np.max(urg_probs, axis=1).round(4),
        "ground_truth_hazard": df["hazard_type"].values,
        "predicted_hazard": haz_preds,
        "hazard_confidence": np.max(haz_probs, axis=1).round(4)
    }

    # Add per-class probability columns
    for idx, cname in enumerate(urg_classes):
        pred_records[f"prob_urgency_{cname}"] = urg_probs[:, idx].round(4)

    for idx, cname in enumerate(haz_classes):
        pred_records[f"prob_hazard_{cname}"] = haz_probs[:, idx].round(4)

    pred_df = pd.DataFrame(pred_records)
    return pred_df, urg_probs, haz_probs, X_features, struct_df


def save_predictions_csv(pred_df: pd.DataFrame, file_name: str) -> Path:
    """Save predictions dataframe to CSV under results/predictions/."""
    out_path = RESULTS_PRED_DIR / file_name
    pred_df.to_csv(out_path, index=False)
    return out_path
