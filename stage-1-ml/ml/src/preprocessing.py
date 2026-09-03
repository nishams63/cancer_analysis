"""
Preprocessing Pipeline Module for Stage 1 ML Toxicity Risk Prediction.
"""

from typing import List, Tuple, Optional
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_preprocessing_pipeline(
    numerical_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer for numerical and categorical preprocessing.
    
    Numerical pipeline: SimpleImputer(median) -> StandardScaler
    Categorical pipeline: SimpleImputer(most_frequent) -> OneHotEncoder(drop='first', handle_unknown='ignore')
    """
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features)
        ],
        remainder="drop"
    )
    
    return preprocessor


class PreprocessingArtifactManager:
    """
    Manages fitting, transforming, saving, and loading of preprocessing pipelines and feature names.
    """
    def __init__(self, preprocessor: Optional[ColumnTransformer] = None):
        self.preprocessor = preprocessor
        self.feature_names_: List[str] = []

    def fit_transform(self, X: pd.DataFrame, numerical_features: List[str], categorical_features: List[str]) -> np.ndarray:
        """
        Fits preprocessor on X and transforms X into a 2D numpy array.
        MUST BE CALLED ONLY ON TRAINING DATA.
        """
        self.preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
        X_transformed = self.preprocessor.fit_transform(X)
        self.feature_names_ = self._extract_feature_names(numerical_features, categorical_features)
        return X_transformed

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms input X using the fitted preprocessor.
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self.preprocessor.transform(X)

    def _extract_feature_names(self, numerical_features: List[str], categorical_features: List[str]) -> List[str]:
        """
        Extracts output feature names after OneHotEncoding.
        """
        feature_names = list(numerical_features)
        if hasattr(self.preprocessor, "named_transformers_") and "cat" in self.preprocessor.named_transformers_:
            cat_trans = self.preprocessor.named_transformers_["cat"]
            if hasattr(cat_trans, "named_steps") and "onehot" in cat_trans.named_steps:
                ohe = cat_trans.named_steps["onehot"]
                if hasattr(ohe, "get_feature_names_out"):
                    cat_ohe_names = list(ohe.get_feature_names_out(categorical_features))
                    feature_names.extend(cat_ohe_names)
        return feature_names

    def save(self, artifact_path: str):
        """
        Saves fitted artifact manager to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(artifact_path)), exist_ok=True)
        joblib.dump(self, artifact_path)

    @classmethod
    def load(cls, artifact_path: str) -> "PreprocessingArtifactManager":
        """
        Loads fitted artifact manager from disk.
        """
        abs_path = os.path.abspath(artifact_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Preprocessing artifact not found at: {abs_path}")
        return joblib.load(abs_path)
