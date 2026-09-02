# Stage 1 — Machine Learning: Data Engineering Pipeline

## Project: Personalized Precision Medicine for Oncology Treatment Optimization

> **IMPORTANT DISCLAIMER**: This is an **EDUCATIONAL/RESEARCH prototype** using **SYNTHETIC data only**.
> - Do not use or generate real patient information.
> - Do not make clinical claims or treatment recommendations.

---

## Executive Summary

This repository contains the Stage-1 Data Engineering pipeline responsible for ingesting, profiling, cleaning, transforming, and validating raw synthetic oncology records (`oncology_uncleaned.csv`) into a clean, reproducible master patient dataset (`master_patient_dataset.csv`).

The validated master dataset is prepared specifically for downstream handoff to the **EDA Engineer** and **ML Engineer** to build an oncology treatment toxicity risk classification model (`Low`, `Moderate`, `High`).

---

## Project Structure

```
stage-1-ml/
└── data-engineering/
    │
    ├── data/
    │   ├── raw/
    │   │   └── oncology_uncleaned.csv       # Preserved raw dataset
    │   │
    │   └── processed/
    │       └── master_patient_dataset.csv   # Cleaned & validated master dataset
    │
    ├── src/
    │   └── data_pipeline.py                 # Reproducible end-to-end data pipeline
    │
    ├── reports/
    │   └── data_quality_report.md           # Automated Data Quality & Profiling Report
    │
    ├── docs/
    │   └── data_dictionary.md               # Detailed Data Dictionary (35 features + target)
    │
    ├── requirements.txt                     # Pipeline Python dependencies
    └── README.md                            # Pipeline documentation & handoff guide
```

---

## Pipeline Stages & Workflow

The pipeline is fully automated and reproducible via `src/data_pipeline.py`:

```
RAW DATA INGESTION -> DATA PROFILING -> DATA CLEANING -> DATA TRANSFORMATION -> TARGET VALIDATION -> QUALITY VALIDATION -> MASTER DATASET EXPORT
```

1. **Data Ingestion**: Loads raw CSV safely without altering source file; verifies encoding (`UTF-8`) and dimensions (8,901 rows × 36 columns).
2. **Data Profiling**: Identifies missing value tokens (`"?"`, `"na"`, `"N/A"`, `"unknown"`), string unit suffixes (`years`, `ng/mL`, `%`, `mg`), physiological outliers, and 147 exact duplicate rows.
3. **Data Cleaning**:
   - **Column Names**: Standardized to `lowercase_snake_case`.
   - **Whitespace**: Stripped leading/trailing whitespace across all string fields.
   - **Categoricals**: Standardized inconsistent categories (`sex`, `cancer_type`, `cancer_stage`, `smoking_history`, `mutation_primary`, `mutation_secondary`, `biomarker_trend`, `treatment_type`, `drug_name`, `previous_adverse_event`, `treatment_response`).
   - **Numeric Bounds**: Stripped text units and replaced physiological invalid values (`age < 0` or `> 120`, `heart_rate < 30` or `> 220`, `systolic_bp < 50` or `> 250`, `white_blood_cell_count < 0`, `oxygen_saturation < 70` or `> 100`) with feature medians.
   - **Dates**: Standardized `observation_date` into `YYYY-MM-DD` ISO-8601 strings.
   - **Deduplication**: Removed 147 exact duplicate rows (retaining 8,754 unique encounter records).
4. **Data Transformation**: Enforces consistent schema column ordering and deterministic sorting by `patient_id`, `observation_date`, and `encounter_id`.
5. **Target Validation**: Validates `toxicity_risk` multi-class target (`Low`, `Moderate`, `High`). Verifies 0 missing target values.
6. **Data Quality Validation**: Executes 11 automated quality assertions prior to export.
7. **Master Dataset Export**: Saves clean master dataset to `data/processed/master_patient_dataset.csv`.
8. **Report & Documentation Generation**: Automatically writes `reports/data_quality_report.md` and `docs/data_dictionary.md`.

---

## Reproducibility & Execution

To re-run the end-to-end data engineering pipeline:

```bash
# 1. Navigate to the data engineering directory
cd stage-1-ml/data-engineering

# 2. Install dependencies (if not already installed)
pip install -r requirements.txt

# 3. Execute the master pipeline
python src/data_pipeline.py
```

---

## Validation Summary

```
============================================================
DATA ENGINEERING STATUS: PASSED
============================================================
Final Dataset Dimensions: 8754 rows x 35 columns
Master CSV Location: data/processed/master_patient_dataset.csv
Quality Report Location: reports/data_quality_report.md
Data Dictionary Location: docs/data_dictionary.md
```

| Check # | Quality Validation Check | Status |
| :---: | :--- | :---: |
| 1 | Required Columns Exist | **PASSED** |
| 2 | Correct Data Types | **PASSED** |
| 3 | Duplicate Rows Handled | **PASSED** |
| 4 | Missing Values Handled | **PASSED** |
| 5 | Categorical Values Standardized | **PASSED** |
| 6 | Numeric Values Validated | **PASSED** |
| 7 | Dates Standardized | **PASSED** |
| 8 | Target Labels Valid (`Low`, `Moderate`, `High`) | **PASSED** |
| 9 | No Accidental Target Leakage | **PASSED** |
| 10 | Schema Stability & Deterministic Order | **PASSED** |
| 11 | Patient/Encounter Relationships Consistent | **PASSED** |

---

## Downstream Team Handoff

- **For EDA Engineer**: Use `data/processed/master_patient_dataset.csv` for exploratory data analysis, feature correlation plots, and distribution studies.
- **For ML Engineer**: Use `data/processed/master_patient_dataset.csv` as the clean feature matrix for encoding, train/test splitting, and training the `toxicity_risk` classifier.
- **Data Dictionary**: Refer to `docs/data_dictionary.md` for feature definitions, data types, and roles.
