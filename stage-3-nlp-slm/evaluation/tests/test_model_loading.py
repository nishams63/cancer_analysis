"""
Unit tests for frozen artifact deserialization in evaluation module.
"""

from model_loader import load_frozen_artifacts
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder


def test_load_frozen_artifacts_types():
    artifacts = load_frozen_artifacts()
    assert isinstance(artifacts.urgency_model, LogisticRegression)
    assert isinstance(artifacts.hazard_model, LogisticRegression)
    assert isinstance(artifacts.tfidf_vectorizer, TfidfVectorizer)
    assert isinstance(artifacts.feature_scaler, StandardScaler)
    assert isinstance(artifacts.urgency_encoder, LabelEncoder)
    assert isinstance(artifacts.hazard_encoder, LabelEncoder)
    assert isinstance(artifacts.numeric_feature_cols, list)


def test_frozen_artifacts_properties():
    artifacts = load_frozen_artifacts()
    assert len(artifacts.numeric_feature_cols) == 12
    assert len(artifacts.urgency_encoder.classes_) == 4
    assert len(artifacts.hazard_encoder.classes_) == 8
    assert len(artifacts.tfidf_vectorizer.vocabulary_) <= 1000
    assert artifacts.feature_scaler.mean_.shape[0] == 12
