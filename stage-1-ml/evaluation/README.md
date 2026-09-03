# Stage 1 — Evaluation Engineer Module

**Project Title**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 1 Machine Learning Evaluation (`stage-1-ml/evaluation`)  
**Target Variable**: `toxicity_risk` (`Low`, `Moderate`, `High`)  
**Target Mapping**: `Low` $\rightarrow$ `0`, `Moderate` $\rightarrow$ `1`, `High` $\rightarrow$ `2`  

---

## 1. Overview & Purpose
This directory contains the independent evaluation and verification suite for the Stage 1 Machine Learning system (**Tuned LightGBM**). As the Evaluation Engineer, our responsibility is to independently validate reported test set results, perform confusion matrix and error transition analysis, verify prediction probability behavior, execute pipeline robustness checks, maintain unit tests, and provide a comprehensive evaluation report for the Integration Engineer.

---

## 2. Directory Structure

```
stage-1-ml/
└── evaluation/
    ├── evaluation.py                     # Main independent evaluation script
    ├── error_analysis.py                 # Error transition & confidence analysis script
    │
    ├── results/
    │   ├── metrics.json                  # Overall & per-class metrics JSON
    │   ├── classification_report.csv     # Classification report CSV
    │   ├── confusion_matrix.png          # Test set confusion matrix heatmap plot
    │   └── error_analysis.csv            # Detailed row-by-row prediction error analysis
    │
    ├── reports/
    │   └── evaluation_report.md          # 12-section technical evaluation report
    │
    ├── tests/
    │   └── test_evaluation.py            # 10 automated unit tests (pytest)
    │
    └── README.md                         # Evaluation Engineer module documentation
```

---

## 3. Input Files & Dependencies
- **Master Dataset**: Read directly from `stage-1-ml/data-engineering/data/processed/master_patient_dataset.csv` (Not copied or modified).
- **Saved Model**: `stage-1-ml/ml/models/best_model/model.joblib` (Tuned LightGBM).
- **Saved Preprocessor**: `stage-1-ml/ml/artifacts/preprocessor/preprocessor.joblib`.
- **Target Encoder**: `stage-1-ml/ml/artifacts/encoders/target_mapping.json`.

---

## 4. Execution Commands

### 4.1 Run Main Evaluation Script
```powershell
py stage-1-ml/evaluation/evaluation.py
```
*Outputs `results/metrics.json`, `results/classification_report.csv`, and `results/confusion_matrix.png`.*

### 4.2 Run Error & Confidence Analysis Script
```powershell
py stage-1-ml/evaluation/error_analysis.py
```
*Outputs `results/error_analysis.csv` and prints error transition frequencies.*

### 4.3 Run Evaluation Unit Test Suite
```powershell
py -m pytest stage-1-ml/evaluation/tests/ -v
```
*Executes all 10 unit tests for metric logic, probability checks, and artifact loading.*

---

## 5. Summary of Independently Verified Results

- **Model Evaluated**: `Tuned LightGBM`
- **Locked Test Set**: 1,750 rows (1,200 unique patients, 0 patient overlap)
- **Overall Performance**:
  - **Accuracy**: `0.5749`
  - **Macro Precision**: `0.5143`
  - **Macro Recall**: `0.5313`
  - **Macro F1 Score**: `0.5204`
  - **Weighted F1 Score**: `0.5721`
  - **High-Risk Recall**: `0.5808`

---

## 6. Handoff to Integration Engineer

The evaluation package contains all required deliverables for Stage 1 Integration:
1. Main evaluation script: `evaluation.py`
2. Error analysis script: `error_analysis.py`
3. Metrics JSON: `results/metrics.json`
4. Classification report: `results/classification_report.csv`
5. Confusion matrix plot: `results/confusion_matrix.png`
6. Detailed error CSV: `results/error_analysis.csv`
7. Formal evaluation report: `reports/evaluation_report.md`
8. Unit test suite: `tests/test_evaluation.py`
9. Module README: `README.md`

> [!WARNING]
> **Educational Disclaimer**: This machine learning evaluation is part of an educational decision-support prototype and must not be used for direct clinical care or medical treatment decisions.
