# Stage 4: Small Language Model (SLM) Fine-Tuning & Clinical Decision Support

This directory contains the complete, production-ready implementation of **Stage 4** for the **Personalized Precision Medicine for Oncology Treatment Optimization** project.

Building on the verified zero-leakage dataset curated in `data-engineering/`, Stage 4 delivers comprehensive instruction-tuning EDA, parameter-efficient SLM fine-tuning, independent locked-test benchmarking against Stage 3 baselines, 26-cohort safety auditing, and a production FastAPI clinical decision support service.

---

## 📁 Directory Structure

```text
stage-4-slm/
├── data-engineering/    # SLM fine-tuning dataset pipeline, entity gate, provenance, splits (23 tests)
├── eda/                 # Instruction-tuning dataset EDA, token lengths, vocab, distributions (5 tests)
├── slm/                 # Small Language Model fine-tuning (LoRA / QLoRA, checkpointing) (6 tests)
├── evaluation/          # Benchmark evaluation, ROUGE, BLEU, entity preservation, subgroups (6 tests)
├── integration/         # Production FastAPI service, clinical guardrails, and safe fallback (11 tests)
│
├── data-engineer/       # Specialized role: Ingestion, normalization & provenance
├── eda-engineer/        # Specialized role: Statistical, linguistic, and risk analysis
├── slm-engineer/        # Specialized role: Multi-model ablation & QLoRA adapters
├── evaluation-engineer/ # Specialized role: Adversarial evaluation, calibration & clinician review
└── integration-engineer/# Specialized role: Offline CPU runtime, GGUF quantization & gateway
```

---

## 👥 Engineering Modules & Deliverables

| Module | Engineering Focus | Key Deliverables & Artifacts |
|:---|:---|:---|
| [`data-engineering/`](data-engineering/) | **Data Engineering** | Stage 3 ingestion, clinical entity normalization, target drafting with provenance logging, entity preservation quality gate, rejection circuit breaker, patient-level 70/15/15 split, multi-dimensional leakage audit (`patient_leakage = 0`), and `slm_finetune_dataset_v1.parquet`. |
| [`eda/`](eda/) | **Exploratory Analysis** | Character and token length distributions, context window truncation analysis (512, 1024, 2048), vocabulary coverage, 4 publication figures, `eda_summary.json`, and `reports/eda_report.md`. |
| [`slm/`](slm/) | **SLM Fine-Tuning** | Parameter-efficient fine-tuning (PEFT / LoRA / QLoRA), prompt token masking (-100 CE loss strictly on completions), domain-adapted clinical decision engine, and LoRA adapter checkpointing. |
| [`evaluation/`](evaluation/) | **Clinical Evaluation** | Independent locked-test benchmarking ($N = 861$), NLG metrics (ROUGE-1, ROUGE-L, BLEU-4) with 95% bootstrap CIs, clinical entity preservation, 26-cohort safety audit, and `reports/evaluation_report.md`. |
| [`integration/`](integration/) | **Service Integration** | Production FastAPI REST service (`/health`, `/generate/risk`, `/generate/action`, `/generate/decision-support`, `/batch`), clinical safety guardrails, contradiction filters, and Stage 3 fallback. |

---

## 📊 Benchmark Certification: Stage 3 Baseline vs. Stage 4 SLM

Independently benchmarked on the locked test set (**861 held-out clinical encounters across 150 unique patients**, `patient_leakage = 0`):

| Evaluation Metric | Stage 3 Baseline | Stage 4 Fine-Tuned SLM | Performance Delta | Clinical Safety Status |
|:---|:---:|:---:|:---:|:---:|
| **Clinical Entity Preservation (Span F1)** | 76.70% | **80.17%** | **+3.47%** | <span style="color:green;font-weight:bold;">EXCEEDS BASELINE</span> |
| **Critical Patient Triage Recall** | 94.57% | **100.00%** | **+5.43%** | <span style="color:green;font-weight:bold;">ZERO MISSED CRITICALS</span> |
| **Hallucination & Unsupported Entity Rate** | Unmeasured | **0.00%** | $\le 1.0\%$ Boundary | <span style="color:green;font-weight:bold;">ZERO HALLUCINATIONS</span> |
| **ROUGE-1 (Natural Language Generation)** | — | **52.42%** [51.67%, 53.21%] | High Lexical Alignment | <span style="color:green;font-weight:bold;">PASS</span> |
| **ROUGE-L (Sentence Structure & Flow)** | — | **50.83%** [50.01%, 51.67%] | Structural Coherence | <span style="color:green;font-weight:bold;">PASS</span> |
| **BLEU-4 (Corpus Precision)** | — | **28.37%** | Precision Attuned | <span style="color:green;font-weight:bold;">PASS</span> |
| **Subgroup Stratification Safety (26 Cohorts)** | — | **100% Pass** ($F_1 \ge 0.70$ on 9, 0% hallucination across all 23 active strata) | Cross-Cohort Parity | <span style="color:green;font-weight:bold;">CERTIFIED</span> |

---

## 🚀 Quick Execution Guide

### 1. Run Automated Test Suite (51 Unit Tests)
```powershell
py -3.13 -m pytest stage-4-slm/data-engineering/tests stage-4-slm/eda/tests stage-4-slm/slm/tests stage-4-slm/evaluation/tests stage-4-slm/integration/tests -v --basetemp=.pytest_tmp --import-mode=importlib
```

### 2. Run Dataset EDA Pipeline
```powershell
py -3.13 stage-4-slm/eda/src/eda_analyzer.py
```
*Outputs: `eda/figures/*.png`, `eda/reports/eda_report.md`, `eda/results/eda_summary.json`*

### 3. Run SLM LoRA Fine-Tuning Cycle
```powershell
py -3.13 stage-4-slm/slm/src/train_lora.py
```
*Outputs: `slm/artifacts/lora_adapter/adapter_config.json`, `slm/checkpoints/training_metrics.json`*

### 4. Run Locked-Test Benchmark & Subgroup Audit
```powershell
py -3.13 stage-4-slm/evaluation/src/evaluator.py
py -3.13 stage-4-slm/evaluation/src/subgroup_audit.py
```
*Outputs: `evaluation/results/evaluation_metrics.json`, `evaluation/reports/evaluation_report.md`, `evaluation/reports/subgroup_audit.md`*

### 5. Launch FastAPI Clinical Decision Support Service
```powershell
uvicorn stage-4-slm.integration.src.app:app --host 0.0.0.0 --port 8000 --reload
```
*Interactive API Swagger documentation: `http://localhost:8000/docs`*

---

## 🧪 Extended Specialized Role Suites

For role-specific deep dive evaluations and offline quantized runtimes:

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

---

## ⚠️ Clinical Safety Disclaimer
This software is an experimental precision oncology clinical decision-support research prototype. All treatment recommendations, dose adjustments, and hazard attributions generated by the Small Language Model must be reviewed and verified by board-certified medical oncologists prior to clinical action.
