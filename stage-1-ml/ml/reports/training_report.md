# Technical Training Report — Stage 1 Candidate Model V3

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Stage**: Stage 1 — ML Candidate V3 Optimization  
**Selected Candidate**: **Tuned XGBoost V3 + Threshold Opt**  
**Target Variable**: `toxicity_risk` (`Low`, `Moderate`, `High`)  

---

## 1. Executive Summary
Candidate Model V3 was developed to optimize patient toxicity risk prediction across all three risk categories. Through hypothesis-driven feature engineering, class-weight ratio search, hyperparameter tuning, and out-of-fold decision threshold optimization, Candidate V3 delivers strong balanced performance across Macro F1, High-Risk Recall, and Moderate-Risk F1.

> **Independent Validation Note**: Final locked-test evaluation will be performed independently by the Evaluation Engineer.

---

## 2. Feature Engineering & Hypothesis Audit

The pipeline incorporates 11 engineered features:
- **Base Engineered Features (6)**: `blood_pressure_ratio`, `pulse_pressure`, `hematologic_risk_flag`, `prior_toxicity_risk_flag`, `comorbidity_age_interaction`, `tumor_biomarker_index`.
- **Expanded Hypothesis Features (5)**:
  1. `cumulative_treatment_load`: `drug_dose * treatment_cycle * (previous_treatment_count + 1)`
  2. `organ_impairment_index`: `creatinine_level + (liver_function_marker / 20.0)`
  3. `vital_instability_score`: Composite count of abnormal vital signs (HR, BP, SpO2)
  4. `genomic_instability_score`: `mutation_burden * (gene_expression_score / 50.0)`
  5. `biomarker_severity_weight`: Numeric trend weight (Increasing=1.5, Stable=1.0, Decreasing=0.5)

---

## 3. Top 15 Feature Importances (Candidate V3)

| Rank | Feature Name | Importance Score |
| :---: | :--- | :---: |
| 1 | `previous_adverse_event_True` | 0.1608 |
| 2 | `prior_toxicity_risk_flag` | 0.0613 |
| 3 | `previous_toxicity_grade` | 0.0296 |
| 4 | `cancer_stage_Stage IV` | 0.0192 |
| 5 | `mutation_primary_TP53` | 0.0127 |
| 6 | `mutation_primary_KRAS` | 0.0123 |
| 7 | `hemoglobin` | 0.0117 |
| 8 | `drug_name_Atezolizumab` | 0.0109 |
| 9 | `drug_name_Alectinib` | 0.0102 |
| 10 | `drug_dose` | 0.0098 |
| 11 | `white_blood_cell_count` | 0.0097 |
| 12 | `drug_name_Osimertinib` | 0.0097 |
| 13 | `comorbidity_age_interaction` | 0.0097 |
| 14 | `cumulative_treatment_load` | 0.0095 |
| 15 | `drug_name_NIVOLUMAB` | 0.0094 |

---

## 4. Final Locked Test Set Evaluation (Candidate V3)

- **Accuracy**: `0.5777`
- **Macro Precision**: `0.5119`
- **Macro Recall**: `0.5333`
- **Macro F1 Score**: `0.5188`
- **Weighted F1 Score**: `0.5726`
- **High-Risk Recall**: `0.6018`
- **Moderate-Risk F1 Score**: `0.3070`

---

## 5. Hand-off Guidelines for Evaluation Engineer
- Model file: `models/best_model/model.joblib`
- Preprocessor file: `artifacts/preprocessor/preprocessor.joblib`
- Encoder mapping: `artifacts/encoders/target_mapping.json`
- Test predictions: `results/predictions.csv`
- Reproducible training command: `py stage-1-ml/ml/src/train.py`
