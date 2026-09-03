# Stage 1 — Machine Learning Toxicity Risk System

**Project Title**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 1 Machine Learning (`stage-1-ml/ml`)  
**Target Variable**: `toxicity_risk` (`Low`, `Moderate`, `High`)  
**Target Mapping**: `Low` $\rightarrow$ `0`, `Moderate` $\rightarrow$ `1`, `High` $\rightarrow$ `2`  

---

## 1. Executive Summary
This directory contains the complete, reproducible, and testable Stage 1 Machine Learning pipeline for predicting patient treatment toxicity risk. The system processes patient clinical measurements, laboratory values, biomarker levels, baseline history, and treatment regimens to output class probabilities and predicted risk levels.

### Key Highlights & Innovations
- **Leakage Prevention**: Non-predictive metadata (`patient_id`, `encounter_id`, `observation_date`) and concurrent treatment outcomes (`treatment_response`) are excluded.
- **Patient-Level Data Split**: 80% train / 20% locked test split grouped by `patient_id` (4,800 train patients vs 1,200 test patients), ensuring **zero patient overlap** between splits.
- **Patient-Aware Cross-Validation**: 5-fold `StratifiedGroupKFold` cross-validation used strictly within training data for model comparison and hyperparameter tuning.
- **Locked Test Evaluation**: Final candidate model evaluated exactly **once** on the locked test set.

---

## 2. Directory Structure

```
stage-1-ml/ml/
│
├── notebooks/
│   └── stage1_ml_experiment.ipynb        # Interactive experiment notebook
│
├── src/
│   ├── __init__.py                       # Package initializer
│   ├── data_loader.py                    # Master dataset loader & validation
│   ├── preprocessing.py                  # Sklearn ColumnTransformer manager
│   ├── feature_engineering.py            # Clinically grounded feature transformer
│   ├── train.py                          # Full training & tuning pipeline
│   ├── predict.py                        # Standalone inference API
│   └── utils.py                          # Splitting, evaluation & plotting utils
│
├── models/
│   ├── baseline/                         # Fitted Logistic Regression baseline
│   └── best_model/                       # Fitted final LightGBM model
│
├── artifacts/
│   ├── preprocessor/                     # Fitted ColumnTransformer (joblib)
│   └── encoders/                         # Target mapping & metadata JSON/joblib
│
├── results/
│   ├── predictions.csv                   # Test set predictions with probabilities
│   ├── feature_importance.csv            # Feature importances with rank
│   ├── feature_importance.png            # Feature importance visualization plot
│   └── confusion_matrix.png              # Test set confusion matrix heatmap
│
├── reports/
│   ├── model_comparison.md               # CV & Test model benchmarking report
│   └── training_report.md                # Comprehensive technical training report
│
├── tests/
│   ├── test_preprocessing.py             # Preprocessing & loader unit tests
│   ├── test_model.py                     # Patient split & target mapping unit tests
│   └── test_prediction.py                # Inference API unit tests
│
├── requirements.txt                      # Module dependencies
└── README.md                             # Module documentation & handoff guide
```

---

## 3. Quick Start & Execution

### 3.1 Install Dependencies
```powershell
py -m pip install -r stage-1-ml/ml/requirements.txt
```

### 3.2 Run Unit Tests
```powershell
py -m pytest stage-1-ml/ml/tests/ -v
```

### 3.3 Run End-to-End Training & Evaluation Pipeline
```powershell
py stage-1-ml/ml/src/train.py
```

### 3.4 Perform Single Patient Inference
```python
from predict import predict_patient_toxicity

patient_record = {
    "age": 62.0, "sex": "Female", "cancer_type": "NSCLC", "cancer_stage": "Stage III",
    "smoking_history": "Former", "mutation_primary": "EGFR", "mutation_secondary": "TP53",
    "mutation_burden": 6.8, "gene_expression_score": 49.8, "ctdna_level": 1.4,
    "tumor_marker_level": 24.7, "inflammation_marker": 4.2, "biomarker_trend": "Stable",
    "heart_rate": 79.0, "systolic_bp": 128.0, "diastolic_bp": 80.0, "oxygen_saturation": 96.0,
    "hemoglobin": 12.5, "white_blood_cell_count": 7.41, "platelet_count": 231.0,
    "creatinine_level": 1.0, "liver_function_marker": 17.3, "treatment_type": "Targeted Therapy",
    "drug_name": "Erlotinib", "drug_dose": 150.0, "treatment_cycle": 4,
    "previous_treatment_count": 1, "previous_adverse_event": False, "previous_toxicity_grade": 0.0,
    "comorbidity_count": 1.0
}

result = predict_patient_toxicity(patient_record)
print(result)
```

---

## 4. Candidate V3 Performance Summary

- **Selected Candidate**: `Candidate V3` (`Tuned XGBoost V3 + Threshold Opt`)
- **Patient Grouped CV Metrics (5-Fold Stratified Group CV on patient_id)**:
  - **CV Macro F1**: `0.5427` (vs `0.5348` baseline V1)
  - **CV High-Risk Recall**: `0.6459` (vs `0.6009` baseline V1)
  - **CV Accuracy**: `0.5992` (vs `0.5877` baseline V1)
  - **CV Moderate-Risk F1**: `0.3284` (vs `0.3312` baseline V1)

> **Evaluation Boundary Note**: Final locked-test evaluation will be performed independently by the Evaluation Engineer.

---

## 5. Handoff to Evaluation Engineer

The following frozen Candidate V3 deliverables are ready for independent assessment:
1. Trained best model: `models/best_model/model.joblib`
2. Fitted preprocessor pipeline: `artifacts/preprocessor/preprocessor.joblib`
3. Encoded target mapping: `artifacts/encoders/target_mapping.json`
4. Test predictions artifact: `results/predictions.csv`
5. Model comparison report: `reports/model_comparison.md`
6. Training report: `reports/training_report.md`

> [!WARNING]
> **Prototype Disclaimer**: This machine learning system is an educational decision-support prototype. It does not provide medical diagnosis or treatment recommendations and must not be used for direct patient care.
