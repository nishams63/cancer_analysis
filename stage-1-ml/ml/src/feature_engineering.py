"""
Feature Engineering Module for Stage 1 ML Toxicity Risk Prediction.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ENGINEERED_FEATURE_NAMES = [
    "blood_pressure_ratio",
    "pulse_pressure",
    "hematologic_risk_flag",
    "prior_toxicity_risk_flag",
    "comorbidity_age_interaction",
    "tumor_biomarker_index"
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that constructs domain-specific engineered features.
    
    Engineered Features:
    1. blood_pressure_ratio: Ratio of systolic to diastolic blood pressure.
    2. pulse_pressure: Difference between systolic and diastolic blood pressure.
    3. hematologic_risk_flag: Binary flag for low hemoglobin (< 11 g/dL) or low platelet count (< 150 k/µL).
    4. prior_toxicity_risk_flag: Binary flag for previous toxicity grade >= 2 AND prior adverse event True.
    5. comorbidity_age_interaction: Interaction term between comorbidity count and age.
    6. tumor_biomarker_index: Product of log1p(mutation_burden) and ctdna_level.
    """

    def __init__(self, include_engineered: bool = True):
        self.include_engineered = include_engineered

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        if not self.include_engineered:
            return X_out

        # 1. blood_pressure_ratio
        if "systolic_bp" in X_out.columns and "diastolic_bp" in X_out.columns:
            X_out["blood_pressure_ratio"] = (X_out["systolic_bp"] / (X_out["diastolic_bp"] + 1e-5)).astype(float)
            
        # 2. pulse_pressure
        if "systolic_bp" in X_out.columns and "diastolic_bp" in X_out.columns:
            X_out["pulse_pressure"] = (X_out["systolic_bp"] - X_out["diastolic_bp"]).astype(float)

        # 3. hematologic_risk_flag
        if "hemoglobin" in X_out.columns and "platelet_count" in X_out.columns:
            low_hb = X_out["hemoglobin"] < 11.0
            low_plt = X_out["platelet_count"] < 150.0
            X_out["hematologic_risk_flag"] = (low_hb | low_plt).astype(int)

        # 4. prior_toxicity_risk_flag
        if "previous_toxicity_grade" in X_out.columns and "previous_adverse_event" in X_out.columns:
            adv_event_bool = X_out["previous_adverse_event"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
            high_tox_grade = X_out["previous_toxicity_grade"] >= 2.0
            X_out["prior_toxicity_risk_flag"] = (adv_event_bool & high_tox_grade).astype(int)

        # 5. comorbidity_age_interaction
        if "comorbidity_count" in X_out.columns and "age" in X_out.columns:
            X_out["comorbidity_age_interaction"] = (X_out["comorbidity_count"] * X_out["age"]).astype(float)

        # 6. tumor_biomarker_index
        if "mutation_burden" in X_out.columns and "ctdna_level" in X_out.columns:
            log_mb = np.log1p(np.maximum(0, X_out["mutation_burden"]))
            X_out["tumor_biomarker_index"] = (log_mb * X_out["ctdna_level"]).astype(float)

        return X_out


def get_feature_engineering_documentation() -> List[Dict[str, str]]:
    """
    Returns documentation of all engineered features for training reports.
    """
    return [
        {
            "feature": "blood_pressure_ratio",
            "formula": "systolic_bp / (diastolic_bp + 1e-5)",
            "rationale": "Measures relative vascular pressure dynamics."
        },
        {
            "feature": "pulse_pressure",
            "formula": "systolic_bp - diastolic_bp",
            "rationale": "Surrogate marker for arterial stiffness and cardiovascular baseline risk."
        },
        {
            "feature": "hematologic_risk_flag",
            "formula": "(hemoglobin < 11.0) | (platelet_count < 150.0)",
            "rationale": "Binary flag indicating baseline hematologic impairment."
        },
        {
            "feature": "prior_toxicity_risk_flag",
            "formula": "(previous_toxicity_grade >= 2) & (previous_adverse_event == True)",
            "rationale": "Identifies patients with documented prior moderate-to-severe adverse event history."
        },
        {
            "feature": "comorbidity_age_interaction",
            "formula": "comorbidity_count * age",
            "rationale": "Captures multiplicative vulnerability of aging with multiple chronic conditions."
        },
        {
            "feature": "tumor_biomarker_index",
            "formula": "log1p(mutation_burden) * ctdna_level",
            "rationale": "Composite tumor load index integrating genomic alteration density and circulating tumor DNA."
        }
    ]
