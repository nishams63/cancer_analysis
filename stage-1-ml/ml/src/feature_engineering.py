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
    "tumor_biomarker_index",
    "cumulative_treatment_load",
    "organ_impairment_index",
    "vital_instability_score",
    "genomic_instability_score",
    "biomarker_severity_weight"
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that constructs domain-specific engineered features.
    
    Base Features (6):
    1. blood_pressure_ratio: Ratio of systolic to diastolic blood pressure.
    2. pulse_pressure: Difference between systolic and diastolic blood pressure.
    3. hematologic_risk_flag: Binary flag for low hemoglobin (< 11 g/dL) or low platelet count (< 150 k/µL).
    4. prior_toxicity_risk_flag: Binary flag for previous toxicity grade >= 2 AND prior adverse event True.
    5. comorbidity_age_interaction: Interaction term between comorbidity count and age.
    6. tumor_biomarker_index: Product of log1p(mutation_burden) and ctdna_level.

    Expanded Experimental Features (5):
    7. cumulative_treatment_load: drug_dose * treatment_cycle * (previous_treatment_count + 1).
    8. organ_impairment_index: creatinine_level + (liver_function_marker / 20.0).
    9. vital_instability_score: Count of abnormal vital signs (HR, BP, SpO2).
    10. genomic_instability_score: mutation_burden * (gene_expression_score / 50.0).
    11. biomarker_severity_weight: Numeric weight for biomarker_trend (Increasing=1.5, Stable=1.0, Decreasing=0.5).
    """

    def __init__(self, include_engineered: bool = True, include_expanded: bool = True):
        self.include_engineered = include_engineered
        self.include_expanded = include_expanded

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

        if not self.include_expanded:
            return X_out

        # 7. cumulative_treatment_load
        if "drug_dose" in X_out.columns and "treatment_cycle" in X_out.columns and "previous_treatment_count" in X_out.columns:
            X_out["cumulative_treatment_load"] = (X_out["drug_dose"] * X_out["treatment_cycle"] * (X_out["previous_treatment_count"] + 1.0)).astype(float)

        # 8. organ_impairment_index
        if "creatinine_level" in X_out.columns and "liver_function_marker" in X_out.columns:
            X_out["organ_impairment_index"] = (X_out["creatinine_level"] + (X_out["liver_function_marker"] / 20.0)).astype(float)

        # 9. vital_instability_score
        if "heart_rate" in X_out.columns and "systolic_bp" in X_out.columns and "oxygen_saturation" in X_out.columns:
            abnormal_hr = (X_out["heart_rate"] > 100.0) | (X_out["heart_rate"] < 60.0)
            abnormal_bp = (X_out["systolic_bp"] > 140.0) | (X_out["systolic_bp"] < 90.0)
            abnormal_spo2 = X_out["oxygen_saturation"] < 95.0
            X_out["vital_instability_score"] = (abnormal_hr.astype(int) + abnormal_bp.astype(int) + abnormal_spo2.astype(int)).astype(float)

        # 10. genomic_instability_score
        if "mutation_burden" in X_out.columns and "gene_expression_score" in X_out.columns:
            X_out["genomic_instability_score"] = (X_out["mutation_burden"] * (X_out["gene_expression_score"] / 50.0)).astype(float)

        # 11. biomarker_severity_weight
        if "biomarker_trend" in X_out.columns:
            trend_map = {"Increasing": 1.5, "Stable": 1.0, "Decreasing": 0.5}
            X_out["biomarker_severity_weight"] = X_out["biomarker_trend"].map(trend_map).fillna(1.0).astype(float)

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
        },
        {
            "feature": "cumulative_treatment_load",
            "formula": "drug_dose * treatment_cycle * (previous_treatment_count + 1)",
            "rationale": "Quantifies systemic drug exposure accumulation across treatment cycles."
        },
        {
            "feature": "organ_impairment_index",
            "formula": "creatinine_level + (liver_function_marker / 20.0)",
            "rationale": "Composite renal and hepatic clearance impairment score."
        },
        {
            "feature": "vital_instability_score",
            "formula": "(abnormal_HR) + (abnormal_BP) + (low_SpO2)",
            "rationale": "Composite index of baseline vital sign physiological instability."
        },
        {
            "feature": "genomic_instability_score",
            "formula": "mutation_burden * (gene_expression_score / 50.0)",
            "rationale": "Interaction between tumor mutational burden and gene expression activity."
        },
        {
            "feature": "biomarker_severity_weight",
            "formula": "biomarker_trend mapped (Increasing=1.5, Stable=1.0, Decreasing=0.5)",
            "rationale": "Quantitative scaling weight reflecting trajectory of biomarker progression."
        }
    ]
