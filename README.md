# Personalized Precision Medicine for Oncology Treatment Optimization

Autonomous multi-agent AI system for personalized precision oncology, integrating Machine Learning, Pharmacogenomics, and Clinical Decision Support to optimize cancer patient treatment safety and mitigate adverse toxicity events.

---

## 📁 Repository Structure

The project is organized into modular stages, with **Stage 1 (Machine Learning Toxicity Risk Prediction)** fully completed:

```text
cancer_analysis/
└── stage-1-ml/
    ├── data-engineering/    # Data cleaning, schema validation, and master patient dataset
    ├── eda/                 # Exploratory data analysis, biomarker distributions, statistical reports
    ├── ml/                  # Model development pipeline, Candidate V4 artifacts, and training code
    ├── evaluation/          # Independent locked-test evaluation, error transition analysis, subgroups
    └── integration/         # Production-ready FastAPI REST service (/health, /predict, /predict/batch)
```

### Module Summary

| Module | Responsibility | Key Deliverables |
|:---|:---|:---|
| [`stage-1-ml/data-engineering`](stage-1-ml/data-engineering/) | Clean, validate, and curate raw oncology records | `master_patient_dataset.csv`, Data dictionary, Quality report |
| [`stage-1-ml/eda`](stage-1-ml/eda/) | Statistical analysis, leakage verification, and distributions | Statistical summaries, correlation heatmaps, biomarker plots |
| [`stage-1-ml/ml`](stage-1-ml/ml/) | Model training, regularization, and feature engineering | Frozen Candidate V4 model artifact (`model.joblib`), preprocessor |
| [`stage-1-ml/evaluation`](stage-1-ml/evaluation/) | Independent locked-test evaluation and error analysis | Confusion matrix, 95% bootstrap CIs, 26-cohort subgroup metrics |
| [`stage-1-ml/integration`](stage-1-ml/integration/) | REST API inference interface for downstream services | FastAPI application, Pydantic input schemas, sample payloads |

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the integration dependencies:
```bash
git clone https://github.com/nishams63/cancer_analysis.git
cd cancer_analysis/stage-1-ml/integration
pip install -r requirements.txt
```

### 2. Run the Prediction API
Start the FastAPI service:
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs will be available at: `http://localhost:8000/docs`.

### 3. Run Automated Tests
Execute the full test suite across the entire project (40 passed tests):
```bash
pytest stage-1-ml/ -v
```

---

## 📊 Stage 1 Final Model Benchmark (Candidate V4)

Candidate Model V4 was trained using regularized LightGBM with conservative decision rules and independently evaluated on the locked test set (1,750 encounters across 1,200 unique patients):

| Metric | Point Estimate | 95% Bootstrap Confidence Interval |
|:---|:---:|:---:|
| **Macro F1 Score (Primary)** | **0.5288** | [0.5035, 0.5521] |
| **High-Risk Recall (Safety)** | **0.6287** | [0.5759, 0.6783] |
| **Accuracy** | **0.5766** | [0.5520, 0.6000] |
| **Weighted F1 Score** | **0.5766** | — |

---

## ⚠️ Clinical Disclaimer
This system is a **machine learning research decision-support prototype**. It has not been approved by regulatory bodies for autonomous clinical diagnosis or treatment planning. All model outputs must be reviewed by qualified oncologists and medical professionals.
