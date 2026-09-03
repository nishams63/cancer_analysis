"""
Data Loader & Validation Module for Stage 1 ML Toxicity Risk Prediction.
"""

import os
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

# Path relative to project root
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data-engineering", "data", "processed", "master_patient_dataset.csv"
)

EXPECTED_ROWS = 8754
EXPECTED_COLS = 35

EXPECTED_TARGET_CLASSES = {"Low", "Moderate", "High"}

TARGET_MAPPING: Dict[str, int] = {
    "Low": 0,
    "Moderate": 1,
    "High": 2
}

REVERSE_TARGET_MAPPING: Dict[int, str] = {
    0: "Low",
    1: "Moderate",
    2: "High"
}

EXCLUDED_FEATURES_MAP: Dict[str, str] = {
    "patient_id": "Patient identifier (Non-predictive metadata)",
    "encounter_id": "Encounter identifier (Non-predictive metadata)",
    "observation_date": "Encounter date (Non-predictive metadata)",
    "treatment_response": "Best overall response to current treatment (Concurrent/post-treatment outcome -> Target leakage)",
    "toxicity_risk": "ML Target variable"
}

NUMERICAL_FEATURES: List[str] = [
    "age",
    "mutation_burden",
    "gene_expression_score",
    "ctdna_level",
    "tumor_marker_level",
    "inflammation_marker",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "oxygen_saturation",
    "hemoglobin",
    "white_blood_cell_count",
    "platelet_count",
    "creatinine_level",
    "liver_function_marker",
    "drug_dose",
    "treatment_cycle",
    "previous_treatment_count",
    "previous_toxicity_grade",
    "comorbidity_count"
]

CATEGORICAL_FEATURES: List[str] = [
    "sex",
    "cancer_type",
    "cancer_stage",
    "smoking_history",
    "mutation_primary",
    "mutation_secondary",
    "biomarker_trend",
    "treatment_type",
    "drug_name",
    "previous_adverse_event"
]


def load_master_dataset(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Loads master patient dataset from CSV file.
    """
    abs_path = os.path.abspath(data_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Master dataset not found at path: {abs_path}")
    df = pd.read_csv(abs_path)
    return df


def validate_dataset(df: pd.DataFrame, expected_rows: Optional[int] = None) -> Dict[str, Any]:
    """
    Validates dataset integrity, dimensions, missing values, duplicates, and target classes.
    """
    issues = []
    
    # 1. Dimension validation
    if expected_rows is not None and df.shape[0] != expected_rows:
        issues.append(f"Row count mismatch: expected {expected_rows}, got {df.shape[0]}")
    if df.shape[1] != EXPECTED_COLS:
        issues.append(f"Column count mismatch: expected {EXPECTED_COLS}, got {df.shape[1]}")
        
    # 2. Missing values check
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        issues.append(f"Unhandled missing values found: {missing_count}")
        
    # 3. Duplicate check
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Duplicate rows found: {duplicate_count}")
        
    # 4. Target column validation
    if "toxicity_risk" not in df.columns:
        issues.append("Target column 'toxicity_risk' missing from dataset")
    else:
        unique_classes = set(df["toxicity_risk"].dropna().unique())
        if unique_classes != EXPECTED_TARGET_CLASSES:
            issues.append(f"Unexpected target classes found: {unique_classes} (Expected {EXPECTED_TARGET_CLASSES})")
            
    # 5. Identifier validation
    if "patient_id" not in df.columns or "encounter_id" not in df.columns:
        issues.append("Patient or Encounter ID missing")
        
    if issues:
        error_msg = "Dataset validation FAILED:\n" + "\n".join(f"- {iss}" for iss in issues)
        raise ValueError(error_msg)
        
    return {
        "status": "PASSED",
        "rows": df.shape[0],
        "columns": df.shape[1],
        "unique_patients": df["patient_id"].nunique(),
        "unique_encounters": df["encounter_id"].nunique(),
        "target_distribution": df["toxicity_risk"].value_counts().to_dict()
    }


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Separates predictor features (X), encoded target (y), and metadata (patient_id, encounter_id).
    
    Returns:
        X: Predictor features DataFrame (30 features)
        y: Target series encoded as integers (0=Low, 1=Moderate, 2=High)
        metadata: Identifiers DataFrame (patient_id, encounter_id, observation_date)
    """
    validate_dataset(df)
    
    # Extract metadata
    metadata = df[["patient_id", "encounter_id", "observation_date"]].copy()
    
    # Encode target
    y = df["toxicity_risk"].map(TARGET_MAPPING)
    if y.isnull().any():
        raise ValueError("Target mapping produced null values due to unmapped toxicity_risk labels.")
    y = y.astype(int)
    
    # Exclude non-predictive / leakage features
    excluded_cols = list(EXCLUDED_FEATURES_MAP.keys())
    X = df.drop(columns=[col for col in excluded_cols if col in df.columns]).copy()
    
    # Verify exact feature count (20 numerical + 10 categorical = 30 features)
    expected_features = len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)
    if X.shape[1] != expected_features:
        raise ValueError(f"Feature count mismatch after exclusion: expected {expected_features}, got {X.shape[1]}")
        
    return X, y, metadata


def get_feature_exclusion_report() -> Dict[str, Any]:
    """
    Returns structured documentation of features used, excluded, and reason for exclusion.
    """
    used_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    return {
        "used_features": used_features,
        "used_feature_count": len(used_features),
        "excluded_features": EXCLUDED_FEATURES_MAP,
        "retained_historical_features": {
            "previous_toxicity_grade": "Historical grade of prior toxicity (baseline record)",
            "previous_adverse_event": "Historical indicator of prior adverse events (baseline record)"
        }
    }
