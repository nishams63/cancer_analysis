"""
Model and Artifact Loader for Stage 3 Clinical NLP Evaluation.
Deserializes frozen baseline models, vectorizers, scalers, and label encoders
strictly in read-only mode without altering any internal states.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import (
    URGENCY_MODEL_PATH,
    HAZARD_MODEL_PATH,
    TFIDF_VECTORIZER_PATH,
    FEATURE_SCALER_PATH,
    NUMERIC_COLS_PATH,
    URGENCY_ENCODER_PATH,
    HAZARD_ENCODER_PATH
)


@dataclass
class FrozenNLPArtifacts:
    """Container for deserialized frozen NLP components."""
    urgency_model: LogisticRegression
    hazard_model: LogisticRegression
    tfidf_vectorizer: TfidfVectorizer
    feature_scaler: StandardScaler
    numeric_feature_cols: List[str]
    urgency_encoder: LabelEncoder
    hazard_encoder: LabelEncoder


def load_frozen_artifacts() -> FrozenNLPArtifacts:
    """
    Load all frozen NLP models and artifacts from the upstream nlp/artifacts directory.
    Raises FileNotFoundError if any artifact is missing.
    """
    for path, name in [
        (URGENCY_MODEL_PATH, "Urgency Model"),
        (HAZARD_MODEL_PATH, "Hazard Model"),
        (TFIDF_VECTORIZER_PATH, "TF-IDF Vectorizer"),
        (FEATURE_SCALER_PATH, "Feature Scaler"),
        (NUMERIC_COLS_PATH, "Numeric Feature Columns"),
        (URGENCY_ENCODER_PATH, "Urgency Encoder"),
        (HAZARD_ENCODER_PATH, "Hazard Encoder")
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required frozen artifact: {name} at {path}")

    urgency_model = joblib.load(URGENCY_MODEL_PATH)
    hazard_model = joblib.load(HAZARD_MODEL_PATH)
    tfidf = joblib.load(TFIDF_VECTORIZER_PATH)
    scaler = joblib.load(FEATURE_SCALER_PATH)
    numeric_cols = joblib.load(NUMERIC_COLS_PATH)
    urgency_encoder = joblib.load(URGENCY_ENCODER_PATH)
    hazard_encoder = joblib.load(HAZARD_ENCODER_PATH)

    return FrozenNLPArtifacts(
        urgency_model=urgency_model,
        hazard_model=hazard_model,
        tfidf_vectorizer=tfidf,
        feature_scaler=scaler,
        numeric_feature_cols=numeric_cols,
        urgency_encoder=urgency_encoder,
        hazard_encoder=hazard_encoder
    )
