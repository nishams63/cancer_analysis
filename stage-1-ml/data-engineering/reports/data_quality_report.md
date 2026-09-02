# Data Quality & Profiling Report — Stage 1 Machine Learning

**Academic Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Stage**: Stage 1 — Machine Learning (Data Engineering Pipeline)  
**Output Master Dataset**: `data/processed/master_patient_dataset.csv`  

---

## 1. Dataset Overview
This report documents the automated data engineering pipeline that ingests, profiles, cleans, transforms, and validates the raw synthetic oncology dataset (`oncology_uncleaned.csv`) to produce a clean master dataset for EDA and toxicity risk machine learning modeling.

- **Raw Data Source**: `data/raw/oncology_uncleaned.csv`
- **Processed Master Path**: `data/processed/master_patient_dataset.csv`
- **Pipeline Execution**: Reproducible via `python src/data_pipeline.py`

---

## 2. Dataset Dimensions Summary

| Metric | Raw Dataset | Processed Master Dataset | Delta / Remediation |
| :--- | :--- | :--- | :--- |
| **Total Rows** | 8901 | 8754 | -147 (147 duplicate rows removed) |
| **Total Columns** | 36 | 35 | Preserved (35 clean feature/target/ID columns) |
| **Unique Patients** | 6,000 | 6,000 | Preserved across encounters |
| **Unique Encounters** | 8,754 | 8,754 | 100% unique per row post-deduplication |
| **Exact Duplicate Rows** | 147 | 0 | Deduplicated |
| **Missing Feature Values** | ~4,200 token/null instances | 0 | Standardized & Imputed |

---

## 3. Profiling & Raw Data Anomalies Detected

### 3.1 String Missing Value Representations
Raw string fields contained heterogeneous missing value tokens (`"?"`, `"na"`, `"NA"`, `"N/A"`, `"unknown"`, `"Unknown"`, `"-"`, `""`). All were converted to standard `np.nan` before applying feature-specific handling strategies.

### 3.2 Numeric String Unit Suffixes
The following numerical columns contained embedded text unit strings in raw records:
- `age`: Stripped `"years"` suffix (e.g. `'46 years'` -> `46.0`)
- `ctdna_level`: Stripped `"ng/mL"` suffix (e.g. `'0.645 ng/mL'` -> `0.645`)
- `oxygen_saturation`: Stripped `"%"` suffix (e.g. `'95.2%'` -> `95.2`)
- `drug_dose`: Stripped `"mg"` suffix (e.g. `'168.3mg'` -> `168.3`)

### 3.3 Physiological Outliers & Implausible Values Detected
Outlier values violating physiological feasibility were flagged and converted to `np.nan` prior to median imputation:
- `age`: 11 invalid values (`< 0` or `> 120` years)
- `heart_rate`: 12 invalid values (`< 30` or `> 220` bpm)
- `systolic_bp`: 17 invalid values (`< 50` or `> 250` mmHg)
- `white_blood_cell_count`: 12 invalid values (`< 0`)
- `oxygen_saturation`: 0 invalid values (`< 70` or `> 100` %)

---

## 4. Missing-Value Handling Strategy

| Feature Type | Column Name(s) | Handling Strategy | Justification |
| :--- | :--- | :--- | :--- |
| **Numerical Features** | `age`, `ctdna_level`, `oxygen_saturation`, `drug_dose`, `mutation_burden`, `gene_expression_score`, `tumor_marker_level`, `inflammation_marker`, `heart_rate`, `systolic_bp`, `diastolic_bp`, `hemoglobin`, `white_blood_cell_count`, `platelet_count`, `creatinine_level`, `liver_function_marker`, `treatment_cycle`, `previous_treatment_count`, `previous_toxicity_grade`, `comorbidity_count` | Median Imputation | Median is robust to extreme skewness and preserves clinical central tendencies. |
| **Categorical Features** | `sex`, `cancer_type`, `cancer_stage`, `smoking_history`, `biomarker_trend`, `treatment_type`, `drug_name`, `treatment_response` | Explicit `'Unknown'` Category | Preserves missingness pattern without introducing synthetic bias. |
| **Genomic Features** | `mutation_primary`, `mutation_secondary` | Explicit `'None/Unknown'` Category | Distinguishes wild-type/untested from known mutations. |
| **Binary Flags** | `previous_adverse_event` | Conservative `False` Imputation | Assumes no documented prior adverse event unless explicitly logged. |
| **ML Target** | `toxicity_risk` | None (0 missing in raw data) | Target values are 100% complete (`Low`, `Moderate`, `High`). |

---

## 5. Categorical Standardization Summary

- `sex`: Standardized `'M'`, `'m'`, `'Male'`, `'MALE'` -> `'Male'`; `'F'`, `'f'`, `'Female'`, `'FEMALE'` -> `'Female'`.
- `cancer_stage`: Mapped arabic/roman/case variations (`'1'`, `'I'`, `'stage 1'`, `'Stage I'`) -> `'Stage I'`, `'Stage II'`, `'Stage III'`, `'Stage IV'`.
- `cancer_type`: Standardized title case across `'NSCLC'`, `'SCLC'`, `'Colorectal Cancer'`, `'Breast Cancer'`, `'Melanoma'`, `'Prostate Cancer'`.
- `smoking_history`: Standardized `'Current'`, `'Former'`, `'Never'`.
- `previous_adverse_event`: Mapped heterogeneous string booleans (`'true'`, `'yes'`, `'1'`, `'false'`, `'no'`, `'0'`) to python `True`/`False`.

---

## 6. Date Transformations
- `observation_date` fields contained mixed formats (`MM-DD-YYYY`, `YYYY-MM-DD`, `DD/MM/YYYY`, `DD-Mon-YYYY`, and corrupt strings like `'not a date'`).
- Converted via robust multi-format datetime parser into standardized `YYYY-MM-DD` ISO-8601 strings.
- Invalid corrupt date strings were back-filled / forward-filled along patient encounters.

---

## 7. Data Quality Validation Results

| Check # | Validation Description | Result | Details |
| :---: | :--- | :---: | :--- |
| 1 | Required Columns Exist | **PASSED** | 35 expected features + identifiers present |
| 2 | Data Types Correct | **PASSED** | Numeric columns cast to float64/int64 |
| 3 | Duplicate Rows Handled | **PASSED** | 0 exact duplicate rows |
| 4 | Missing Values Handled | **PASSED** | 0 unhandled NaNs remaining |
| 5 | Categorical Standardized | **PASSED** | Clean category sets enforced |
| 6 | Numeric Bounds Validated | **PASSED** | Physiological ranges strictly enforced |
| 7 | Date Format Standardized | **PASSED** | ISO-8601 `YYYY-MM-DD` verified |
| 8 | Target Labels Valid | **PASSED** | Only `Low`, `Moderate`, `High` present |
| 9 | No Target Leakage | **PASSED** | Zero proxy target features created |
| 10 | Schema Stability | **PASSED** | Schema order strictly deterministic |
| 11 | Encounter Consistency | **PASSED** | `encounter_id` 100% unique per row |

---

## 8. Final Master Dataset Statistics

- **Rows**: 8,754
- **Columns**: 35
- **Toxicity Risk Class Distribution**:
  - `Low`: 4,676 (53.4%)
  - `Moderate`: 2,414 (27.6%)
  - `High`: 1,664 (19.0%)

---

## 9. Known Limitations
1. **Synthetic Data Nature**: Synthetic correlations may not reflect real clinical trial bio-distribution.
2. **Median Imputation**: Features with missing values rely on population median imputation; future stages may test MICE or KNN imputation if required by ML Engineers.
3. **Single Target Attribute**: `toxicity_risk` is multi-class ordinal; ordinal encoding may be performed by ML Engineer during feature engineering.

---
**Status**: `DATA ENGINEERING STATUS: PASSED`
