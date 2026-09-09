# Stage 4: Small Language Model (SLM) Fine-Tuning & Clinical Evaluation

This directory contains the complete end-to-end implementation of **Stage 4** for the Oncology Precision Medicine clinical NLP project.

---

## 📁 Directory Structure

```text
stage-4-slm/
├── data-engineer/
│   ├── src/
│   ├── tests/
│   ├── data/
│   ├── reports/
│   └── README.md
│
├── eda-engineer/
│   ├── src/
│   ├── tests/
│   ├── figures/
│   ├── reports/
│   └── README.md
│
├── slm-engineer/
│   ├── src/
│   ├── configs/
│   ├── tests/
│   ├── adapters/
│   ├── results/
│   └── README.md
│
├── evaluation-engineer/
│   ├── src/
│   ├── benchmarks/
│   ├── tests/
│   ├── figures/
│   ├── results/
│   ├── reports/
│   └── README.md
│
└── integration-engineer/
    ├── src/
    ├── frontend/
    ├── runtime/
    ├── scripts/
    ├── tests/
    ├── artifacts/
    ├── reports/
    ├── docs/
    ├── deployment/
    ├── cli.py
    ├── requirements.txt
    ├── README.md
    └── REPRODUCIBILITY.md
```

---

## 👥 Engineering Modules & Responsibilities

| Module | Engineering Role | Key Responsibility & Deliverables |
|:---|:---|:---|
| [`data-engineer/`](data-engineer/) | **Data Engineer** | Stage 3 ingestion, clinical entity normalization, target drafting with provenance logging, entity preservation quality gate, rejection circuit breaker, patient-level 70/15/15 split, multi-dimensional leakage audit (`patient_leakage = 0`), and `slm_finetune_dataset_v1.parquet`. |
| [`eda-engineer/`](eda-engineer/) | **EDA Engineer** | Statistical, linguistic, and clinical readiness audit of Stage 4 instruction pairs; tokenization profiling across target SLM tokenizers, vocabulary coverage, and partition uniformity analysis. |
| [`slm-engineer/`](slm-engineer/) | **SLM Engineer** | QLoRA fine-tuning (BioMistral-7B, Clinical-Llama-3-8B), 4-configuration ablation study, clinical entity retention evaluation, and checkpoint/adapter generation. |
| [`evaluation-engineer/`](evaluation-engineer/) | **Evaluation Engineer** | Rigorous multi-axis evaluation: In-distribution locked test, OOD testing (synthetic vs real provenance), adversarial & negation stress testing, empirical calibration ($\tau^*$), 6-gate safety firewall, blinded clinician review protocol, and production deployment gating. |
| [`integration-engineer/`](integration-engineer/) | **Integration Engineer** | 100% offline local CPU application: GGUF conversion & quantization (Q4_K_M, Q5_K_M), llama.cpp runtime, FastAPI service (`/summarize`, `/health`, `/metrics`), Stage 6 safety firewall, responsive offline web UI, terminal CLI, cryptographic provenance, and immutable audit logging. |

---

## 🧪 Testing & Execution

Run individual engineer test suites:
```bash
# Data Engineer (23 tests)
pytest stage-4-slm/data-engineer/tests/ -v

# EDA Engineer (16 tests)
pytest stage-4-slm/eda-engineer/tests/ -v

# SLM Engineer (13 tests)
pytest stage-4-slm/slm-engineer/tests/ -v

# Evaluation Engineer (16 tests)
pytest stage-4-slm/evaluation-engineer/tests/ -v

# Integration Engineer (20 tests)
pytest stage-4-slm/integration-engineer/tests/ -v
```

Run all Stage 4 test suites across all 5 engineering roles (88 tests total):
```bash
# Bash
for mod in data-engineer eda-engineer slm-engineer evaluation-engineer integration-engineer; do pytest "stage-4-slm/$mod/tests/" -v; done

# PowerShell
"data-engineer", "eda-engineer", "slm-engineer", "evaluation-engineer", "integration-engineer" | ForEach-Object { pytest "stage-4-slm/$_/tests" -v }
```

Run full end-to-end pipelines:
```bash
# 1. Data Engineering Dataset Synthesis
python stage-4-slm/data-engineer/src/pipeline.py

# 2. EDA Audit
python stage-4-slm/eda-engineer/src/pipeline.py

# 3. SLM Ablation Study (dry-run or full GPU execution)
python stage-4-slm/slm-engineer/src/pipeline.py --dry-run

# 4. Independent Clinical Evaluation Pipeline
python stage-4-slm/evaluation-engineer/src/pipeline.py

# 5. Integration Offline Validation & Service Launch
python stage-4-slm/integration-engineer/scripts/offline_validation.py
python stage-4-slm/integration-engineer/src/api.py
```
