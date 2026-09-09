"""
Target Label Preparation and Class Weighting Module for Stage 3 NLP.
Encodes targets and computes inverse class frequency weights fitted ONLY on the TRAIN partition.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

from config import ENCODERS_DIR, VALID_URGENCY_LEVELS, VALID_HAZARD_TYPES


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights: w_c = N / (K * N_c)
    Counters the severe class imbalance documented in EDA (7.6:1 urgency, 138:1 hazard).
    """
    classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    n_classes = len(classes)
    weights = {}
    for cls, count in zip(classes, counts):
        weights[int(cls)] = float(total_samples / (n_classes * count))
    return weights


class TargetLabelManager:
    """Manages encoding and decoding of target labels strictly fitted on TRAIN."""

    def __init__(self):
        self.urgency_encoder = LabelEncoder()
        self.hazard_encoder = LabelEncoder()
        self.urgency_weights: Dict[int, float] = {}
        self.hazard_weights: Dict[int, float] = {}
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> "TargetLabelManager":
        """Fit encoders exclusively on the TRAIN split."""
        # Fit Urgency
        self.urgency_encoder.fit(df_train["urgency_level"])
        y_urg_train = self.urgency_encoder.transform(df_train["urgency_level"])
        self.urgency_weights = compute_class_weights(y_urg_train)

        # Fit Hazard
        self.hazard_encoder.fit(df_train["hazard_type"])
        y_haz_train = self.hazard_encoder.transform(df_train["hazard_type"])
        self.hazard_weights = compute_class_weights(y_haz_train)

        self.is_fitted = True
        return self

    def transform_urgency(self, series: pd.Series) -> np.ndarray:
        """Encode urgency labels."""
        if not self.is_fitted:
            raise RuntimeError("LabelManager must be fitted on TRAIN before transform!")
        return self.urgency_encoder.transform(series)

    def transform_hazard(self, series: pd.Series) -> np.ndarray:
        """Encode hazard labels."""
        if not self.is_fitted:
            raise RuntimeError("LabelManager must be fitted on TRAIN before transform!")
        return self.hazard_encoder.transform(series)

    def inverse_transform_urgency(self, y_pred: np.ndarray) -> np.ndarray:
        """Decode urgency integer predictions back to class strings."""
        return self.urgency_encoder.inverse_transform(y_pred)

    def inverse_transform_hazard(self, y_pred: np.ndarray) -> np.ndarray:
        """Decode hazard integer predictions back to class strings."""
        return self.hazard_encoder.inverse_transform(y_pred)

    def save(self, output_dir=ENCODERS_DIR) -> None:
        """Serialize fitted encoders and class weights."""
        joblib.dump(self.urgency_encoder, output_dir / "urgency_encoder.joblib")
        joblib.dump(self.hazard_encoder, output_dir / "hazard_encoder.joblib")
        joblib.dump(self.urgency_weights, output_dir / "urgency_weights.joblib")
        joblib.dump(self.hazard_weights, output_dir / "hazard_weights.joblib")

    @classmethod
    def load(cls, input_dir=ENCODERS_DIR) -> "TargetLabelManager":
        """Load pre-fitted encoders and class weights."""
        mgr = cls()
        mgr.urgency_encoder = joblib.load(input_dir / "urgency_encoder.joblib")
        mgr.hazard_encoder = joblib.load(input_dir / "hazard_encoder.joblib")
        mgr.urgency_weights = joblib.load(input_dir / "urgency_weights.joblib")
        mgr.hazard_weights = joblib.load(input_dir / "hazard_weights.joblib")
        mgr.is_fitted = True
        return mgr
