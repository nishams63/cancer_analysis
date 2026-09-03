# Stage 1: Machine Learning Toxicity Risk Prediction

This directory contains the complete end-to-end implementation of **Stage 1** for the Oncology Precision Medicine project.

---

## 📁 Directory Structure

```text
stage-1-ml/
├── data-engineering/    # Data cleaning, normalization, and master dataset generation
├── eda/                 # Exploratory data analysis, distributions, and leakage checks
├── ml/                  # Model training pipeline, feature engineering, and Candidate V4 model
├── evaluation/          # Independent locked-test evaluation, error analysis, and subgroup metrics
└── integration/         # FastAPI REST service serving the frozen V4 model for inference
```

---

## 🔄 End-to-End Workflow

1. **Data Engineering** (`data-engineering/`):
   - Ingests raw oncology encounters, resolves schema inconsistencies, removes data leakage columns, and creates `master_patient_dataset.csv`.
2. **Exploratory Data Analysis** (`eda/`):
   - Computes statistical summaries, verifies absence of target leakage, and generates biomarker distributions and correlation heatmaps.
3. **Machine Learning Pipeline** (`ml/`):
   - Performs patient-level stratified splitting (80% train / 20% locked test).
   - Trains Candidate Model V4: Regularized LightGBM (depth=3, 80 estimators, balanced class weights, gentle decision multipliers `W = [1.0, 1.05, 1.05]`).
   - Freezes model artifacts in `models/best_model/model.joblib`.
4. **Independent Evaluation** (`evaluation/`):
   - Evaluates the frozen V4 model on the locked test set (1,750 encounters / 1,200 patients).
   - Computes 95% bootstrap confidence intervals and performs error transition analysis.
5. **Inference Integration** (`integration/`):
   - Serves the frozen model via FastAPI endpoints (`/health`, `/predict`, `/predict/batch`).
   - Validates incoming patient payloads and handles unknown categorical values gracefully.

---

## 🧪 Testing

To run the full test suite across all Stage 1 modules (40 tests):
```bash
pytest stage-1-ml/ -v
```
