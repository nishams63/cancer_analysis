# Personalized Precision Medicine for Oncology Treatment Optimization

Autonomous multi-agent AI system for personalized precision oncology, integrating Machine Learning, Pharmacogenomics, and Clinical Decision Support to optimize cancer patient treatment safety and mitigate adverse toxicity events.

---

## 📁 Repository Structure

The project is organized into modular engineering stages and documentation:

```text
cancer_analysis/
├── docs/
│   └── viva-preparation/     # Student-friendly Level 1 Viva Study PDF, HTML, & generators
├── stage-1-ml/               # Classical Machine Learning Toxicity Risk Prediction
│   ├── data-engineering/    # Data cleaning, schema validation, and master patient dataset
│   ├── eda/                 # Exploratory data analysis, biomarker distributions, statistical reports
│   ├── ml/                  # Model development pipeline, Candidate V4 artifacts, and training code
│   ├── evaluation/          # Independent locked-test evaluation, error transition analysis, subgroups
│   └── integration/         # Production-ready FastAPI REST service (/health, /predict, /predict/batch)
├── stage-2-dl/               # Deep Learning Multimodal Progression Prediction
│   ├── data-engineering/    # Synthetic pathology tiles and temporal biomarker sequences
│   ├── eda/                 # Spatial tissue and longitudinal trajectory analysis
│   ├── dl/                  # Vision CNNs, LSTMs, Transformers, MIL attention models
│   ├── evaluation/          # Locked test evaluation, calibration, and benchmark reports
│   └── integration/         # Multimodal Late Fusion API, clinical risk engine, and dashboard
├── stage-3-nlp-slm/          # NLP & Small Language Model (SLM) Decision Support
│   ├── data-engineering/    # Production-quality research NLP dataset, deduplication, PII scrub, splits
│   ├── eda/                 # Exploratory data analysis, vocabulary, negation profiling, 16 figures, reports
│   └── nlp/                 # Clinical NLP pipeline, negation scoping, feature extraction, baselines
├── stage-4-slm/              # SLM Fine-Tuning & Multi-Agent Decision Support
│   ├── data-engineer/        # Entity-verified instruction dataset pipeline, circuit breaker, zero-leakage splits
│   ├── eda-engineer/         # Statistical, linguistic & tokenization audit of instruction pairs (16 tests)
│   ├── slm-engineer/         # Parameter-efficient QLoRA fine-tuning & ablation study (BioMistral, LLaMA-3)
│   ├── evaluation-engineer/  # OOD, adversarial, empirical calibration (tau*), safety firewall & clinician review
│   └── integration-engineer/ # 100% offline local CPU FastAPI service, GGUF/llama.cpp, web UI, CLI (20 tests)
└── stage 5 Gen-AI/           # GenAI Synthetic Oncology Stress-Test Engine
    ├── Data Engineer/        # Reference distributions, bounds, constraints, rare-combination space, evidence chunks
    ├── Eda Engineer/         # Blind-spot analysis, scenario library (PROMPT-R01 to R15), retrieval intents, drift rules
    ├── Gen Ai Engineer/      # StructuredSampler, NVIDIA LLM client, TF-IDF RAG retriever, counterfactuals
    ├── Evaluation Engineer/  # 10-question evaluation harness, F01-F20 taxonomy, difficulty & impact scoring, 8 reports
    └── Integration Engineer/ # Unified CLI (run_stage5.py), SQLite store, Wildcard ranking & interactive dashboard
```

### Module Summary

| Module | Responsibility | Key Deliverables |
|:---|:---|:---|
| [`docs/viva-preparation`](docs/viva-preparation/) | Data Science Viva Level 1 Study Guide | [`Data_Science_Viva_Level_1_Study_Material.pdf`](docs/viva-preparation/Data_Science_Viva_Level_1_Study_Material.pdf), [`HTML Reference`](docs/viva-preparation/Data_Science_Viva_Level_1_Study_Material.html) |
| [`stage-1-ml/data-engineering`](stage-1-ml/data-engineering/) | Clean, validate, and curate raw oncology records | `master_patient_dataset.csv`, Data dictionary, Quality report |
| [`stage-1-ml/eda`](stage-1-ml/eda/) | Statistical analysis, leakage verification, and distributions | Statistical summaries, correlation heatmaps, biomarker plots |
| [`stage-1-ml/ml`](stage-1-ml/ml/) | Model training, regularization, and feature engineering | Frozen Candidate V4 model artifact (`model.joblib`), preprocessor |
| [`stage-1-ml/evaluation`](stage-1-ml/evaluation/) | Independent locked-test evaluation and error analysis | Confusion matrix, 95% bootstrap CIs, 26-cohort subgroup metrics |
| [`stage-1-ml/integration`](stage-1-ml/integration/) | REST API inference interface for downstream services | FastAPI application, Pydantic input schemas, sample payloads |
| [`stage-2-dl`](stage-2-dl/) | Multimodal Vision (CNN) & Sequence (LSTM/Transformer) | Tile classification, ctDNA forecasting, late fusion alert engine |
| [`stage-3-nlp-slm/data-engineering`](stage-3-nlp-slm/data-engineering/) | Curate leakage-free clinical NLP & SLM research dataset | `clinical_nlp_dataset_v1.parquet`, Train/Val/Test splits, audits |
| [`stage-3-nlp-slm/eda`](stage-3-nlp-slm/eda/) | Characterize clinical text, vocabulary, negation, and splits | 20-section EDA report, 16 figures, `eda_summary.json`, notebook |
| [`stage-3-nlp-slm/nlp`](stage-3-nlp-slm/nlp/) | Negation-aware clinical feature extraction & baselines | Baseline models (Macro F1 0.7557), feature outputs, 6 reports |
| [`stage-4-slm/data-engineer`](stage-4-slm/data-engineer/) | Entity-verified instruction dataset pipeline | `slm_finetune_dataset_v1.parquet`, 23 tests, zero-leakage reports |
| [`stage-4-slm/eda-engineer`](stage-4-slm/eda-engineer/) | Statistical, linguistic & tokenization audit | Readiness report, 16 unit tests, vocab & token length profiling |
| [`stage-4-slm/slm-engineer`](stage-4-slm/slm-engineer/) | QLoRA fine-tuning & 4-config ablation study | BioMistral-7B / Clinical-Llama-3 adapters, 13 unit tests |
| [`stage-4-slm/evaluation-engineer`](stage-4-slm/evaluation-engineer/) | Multi-axis clinical evaluation & safety firewall | OOD/adversarial benchmarks, tau* calibration, 6-gate firewall, 16 tests |
| [`stage-4-slm/integration-engineer`](stage-4-slm/integration-engineer/) | Offline local CPU application & decision support | GGUF Q4_K_M, llama.cpp, FastAPI (/summarize), Web UI, CLI, 20 tests |
| [`stage 5 Gen-AI/Data Engineer`](stage%205%20Gen-AI/Data%20Engineer/) | Reference distributions, constraints & evidence corpus | Validated parquets, constraint specs, 32 unit tests |
| [`stage 5 Gen-AI/Eda Engineer`](stage%205%20Gen-AI/Eda%20Engineer/) | Blind-spot analysis & scenario prompt library | 7 analysis modules, PROMPT-R01 to R15, 21 unit tests |
| [`stage 5 Gen-AI/Gen Ai Engineer`](stage%205%20Gen-AI/Gen%20Ai%20Engineer/) | GenAI synthetic patient & counterfactual generator | Structured sampler, RAG retriever, LLM pipeline, 25 unit tests |
| [`stage 5 Gen-AI/Evaluation Engineer`](stage%205%20Gen-AI/Evaluation%20Engineer/) | Multi-axis stress-test evaluation & failure audit | 10 DoD verifications, 8 markdown reports, 24 unit tests |
| [`stage 5 Gen-AI/Integration Engineer`](stage%205%20Gen-AI/Integration%20Engineer/) | End-to-end orchestration & operational layer | Unified CLI (`run_stage5.py`), SQLite, dashboard, 14 unit tests |

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

### 3. Run Stage 5 GenAI Synthetic Stress-Test Engine
Execute the unified Stage 5 pipeline to generate and stress-test 20 synthetic oncology scenarios:
```bash
python run_stage5.py --n 20 --seed 42
```
Launch the interactive Stage 5 evaluation dashboard:
```bash
uvicorn "stage 5 Gen-AI.Integration Engineer.src.dashboard.dashboard_api:create_app" --factory --port 8085
```

### 4. Run Automated Tests
Execute the test suites across the project (116 passed tests across Stage 5 roles):
```bash
pytest "stage 5 Gen-AI/Data Engineer/tests" -v
pytest "stage 5 Gen-AI/Eda Engineer/tests" -v
pytest "stage 5 Gen-AI/Gen Ai Engineer/tests" -v
pytest "stage 5 Gen-AI/Evaluation Engineer/tests" -v
pytest "stage 5 Gen-AI/Integration Engineer/tests" -v
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
