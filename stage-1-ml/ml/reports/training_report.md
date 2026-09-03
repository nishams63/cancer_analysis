# Stage 1 ML Training & System Architecture Report

**Project Title**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 1 Machine Learning (`stage-1-ml/ml`)  
**Target Variable**: `toxicity_risk` (`Low`, `Moderate`, `High`)  

---

## 1. Machine Learning Objective
The primary objective of Stage 1 ML is to construct a robust, reproducible, and leakage-safe multiclass classification model that predicts patient treatment toxicity risk from baseline patient clinical characteristics, laboratory counts, tumor biomarkers, and historical treatment events.

---

## 2. Dataset Overview & Data Quality
- **Source**: `stage-1-ml/data-engineering/data/processed/master_patient_dataset.csv`
- **Total Records**: 8754 encounter rows
- **Unique Patients**: 6000 unique patients
- **Total Features**: 35 original columns (30 input features + 3 metadata/identifiers + 1 leakage candidate + 1 target)
- **Data Quality**: 0 missing values, 0 duplicate rows, standardized physiological bounds verified by Data Engineering.

---

## 3. Target Variable Specification
- **Name**: `toxicity_risk`
- **Classes**:
  - `Low`: 4,678 encounters (53.4%)
  - `Moderate`: 2,409 encounters (27.5%)
  - `High`: 1,667 encounters (19.0%)
- **Target Encoding Mapping**:
  - `Low` $\rightarrow$ `0`
  - `Moderate` $\rightarrow$ `1`
  - `High` $\rightarrow$ `2`

---

## 4. Feature Selection & Data Leakage Prevention

### 4.1 Excluded Features

| Feature Name | Reason for Exclusion | Category |
| :--- | :--- | :--- |
| `patient_id` | Patient identifier (Non-predictive metadata) | Identifier / Metadata |
| `encounter_id` | Encounter identifier (Non-predictive metadata) | Identifier / Metadata |
| `observation_date` | Encounter date (Non-predictive metadata) | Identifier / Metadata |
| `treatment_response` | Best overall response to current treatment (Concurrent/post-treatment outcome -> Target leakage) | Target Leakage |
| `toxicity_risk` | ML Target variable | Target |

### 4.2 Retained Historical Features

| Feature Name | Baseline Rationale |
| :--- | :--- |
| `previous_toxicity_grade` | Historical grade of prior toxicity (baseline record) |
| `previous_adverse_event` | Historical indicator of prior adverse events (baseline record) |

---

## 5. Patient-Level Train/Test Split Strategy
- **Grouping Variable**: `patient_id`
- **Split Ratio**: 80% Training (~4,800 patients / ~7,003 encounters), 20% Locked Test (~1,200 patients / ~1,751 encounters)
- **Stratification**: Unique patients stratified by primary toxicity risk.
- **Overlap Check**: `set(train_patient_ids) ∩ set(test_patient_ids) == empty` (0 patient overlap verified).

---

## 6. Preprocessing & Encoding Pipeline
- **Numerical Features**: `SimpleImputer(strategy='median')` -> `StandardScaler()`
- **Categorical Features**: `SimpleImputer(strategy='most_frequent')` -> `OneHotEncoder(drop='first', handle_unknown='ignore')`
- **Leakage Safeguard**: Preprocessing pipeline fitted strictly on training data (`X_train`) and applied downstream to test data.

---

## 7. Feature Engineering Specification

| Feature Name | Formula / Logic | Clinical Rationale |
| :--- | :--- | :--- |
| `blood_pressure_ratio` | `systolic_bp / (diastolic_bp + 1e-5)` | Measures relative vascular pressure dynamics. |
| `pulse_pressure` | `systolic_bp - diastolic_bp` | Surrogate marker for arterial stiffness and cardiovascular baseline risk. |
| `hematologic_risk_flag` | `(hemoglobin < 11.0) | (platelet_count < 150.0)` | Binary flag indicating baseline hematologic impairment. |
| `prior_toxicity_risk_flag` | `(previous_toxicity_grade >= 2) & (previous_adverse_event == True)` | Identifies patients with documented prior moderate-to-severe adverse event history. |
| `comorbidity_age_interaction` | `comorbidity_count * age` | Captures multiplicative vulnerability of aging with multiple chronic conditions. |
| `tumor_biomarker_index` | `log1p(mutation_burden) * ctdna_level` | Composite tumor load index integrating genomic alteration density and circulating tumor DNA. |

---

## 8. Cross-Validation & Model Selection Results

Evaluated using 5-fold **StratifiedGroupKFold** on `patient_id` within the training split:

| Model Name | CV Macro F1 | CV Weighted F1 | CV High-Risk Recall | CV Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| `Logistic Regression (Baseline)` | **0.5279** | 0.5706 | **0.6324** | 0.5731 |
| `Decision Tree` | **0.4564** | 0.5043 | **0.4163** | 0.5014 |
| `Random Forest` | **0.4710** | 0.5382 | **0.4561** | 0.6098 |
| `XGBoost` | **0.5078** | 0.5673 | **0.4659** | 0.5949 |
| `LightGBM` | **0.5290** | 0.5769 | **0.5724** | 0.5865 |
| `Tuned Random Forest` | **0.5208** | 0.5771 | **0.6129** | 0.6142 |
| `Tuned LightGBM` | **0.5328** | 0.5799 | **0.6009** | 0.5877 |

---

## 9. Final Model Performance on Locked Test Set

Selected Model: **Tuned LightGBM**

- **Accuracy**: 0.5749
- **Macro Precision**: 0.5143
- **Macro Recall**: 0.5313
- **Macro F1 Score**: 0.5204
- **Weighted F1 Score**: 0.5721
- **High-Risk Recall**: 0.5808

---

## 10. Top 10 Predictive Feature Importances

| Rank | Feature | Importance Score |
| :---: | :--- | :---: |
| 1 | `drug_dose` | 505.000000 |
| 2 | `white_blood_cell_count` | 470.000000 |
| 3 | `hemoglobin` | 462.000000 |
| 4 | `creatinine_level` | 410.000000 |
| 5 | `gene_expression_score` | 406.000000 |
| 6 | `liver_function_marker` | 405.000000 |
| 7 | `platelet_count` | 387.000000 |
| 8 | `comorbidity_age_interaction` | 375.000000 |
| 9 | `previous_toxicity_grade` | 354.000000 |
| 10 | `tumor_marker_level` | 350.000000 |

> [!NOTE]
> **Interpretation Disclaimer**: Feature importance indicates predictive utility within the machine learning model. It does NOT establish biological or clinical causality.

---

## 11. Known Limitations & Recommendations
1. **Synthetic Dataset Structure**: The data is derived from synthetic oncology clinical trials; real-world clinical bio-distribution may require additional transfer learning or re-calibration.
2. **Research Decision-Support Prototype**: The model system is an educational prototype and MUST NOT be used for direct patient diagnosis or prescribing treatment without rigorous clinical trial validation.

---

## 12. Step-by-Step Reproduction Instructions

```powershell
# 1. Clone & navigate to repo
git clone https://github.com/nishams63/cancer_analysis.git
cd cancer_analysis

# 2. Run unit tests
py -m pytest stage-1-ml/ml/tests/ -v

# 3. Train ML pipeline and generate artifacts
py stage-1-ml/ml/src/train.py
```
