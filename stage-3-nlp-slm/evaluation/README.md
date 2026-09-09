# Stage 3 — Clinical NLP Independent Evaluation

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: `stage-3-nlp-slm/evaluation`  
**Role**: Independent Evaluation Engineer  
**Status**: OFFICIAL, CERTIFIED & REPRODUCIBLE  
**Final Verdict**: **ACCEPTED WITH LIMITATIONS**  

---

## 1. Overview
This module provides an **independent, rigorous, and reproducible evaluation** of the frozen Stage 3 Clinical NLP pipeline and its baseline models.
The evaluation adheres to strict non-interventional principles:
- Upstream datasets (`data-engineering/`), exploratory data analysis (`eda/`), and NLP pipeline code (`nlp/`) are **strictly frozen and read-only**.
- No models were retrained, fine-tuned, or recalibrated.
- The **Locked Test Set** ($N=928$, 150 unique patients) was evaluated strictly once under an uncompromised holdout protocol.

---

## 2. Directory Architecture

```
stage-3-nlp-slm/evaluation/
├── configs/
│   └── evaluation_config.yaml         # Centralized configuration, schemas, paths, bootstrap settings
├── src/
│   ├── __init__.py                    # Module init
│   ├── config.py                      # Path resolver and vocabulary constants
│   ├── data_loader.py                 # Read-only data loader with schema validation
│   ├── model_loader.py                # Deserialization of frozen NLP models and vectorizers
│   ├── prediction_runner.py           # Deterministic batch inference on Val and Test
│   ├── classification_metrics.py      # Accuracy, Macro/Weighted F1, Recall, Precision, CM
│   ├── extraction_metrics.py          # Exact and relaxed span-level NER metrics
│   ├── negation_metrics.py            # Diagnostic suite for clinical negation scoping
│   ├── feature_metrics.py             # Feature dimensions, missingness, drift, and zero-variance
│   ├── calibration.py                 # Expected Calibration Error (ECE) and Brier score
│   ├── robustness.py                  # Textual perturbation stability (casing, spacing, punctuation)
│   ├── error_analysis.py              # Categorized failure taxonomy and high-confidence misses
│   ├── leakage_checks.py              # Cryptographic SHA-256 checks and split isolation proofs
│   ├── confidence_intervals.py        # 1,000-iteration patient-clustered bootstrap 95% CIs
│   ├── utils.py                       # Plotting helpers and JSON encoders
│   └── run_evaluation.py              # Master evaluation orchestrator script
├── notebooks/
│   └── stage3_evaluation.ipynb        # Interactive step-by-step evaluation walkthrough
├── results/
│   ├── validation/                    # validation_metrics.json
│   ├── locked_test/                   # locked_test_metrics.json
│   ├── metrics/                       # evaluation_summary.json, reproducibility_results.json
│   ├── predictions/                   # validation_predictions.csv, locked_test_predictions.csv
│   ├── confusion_matrices/            # Labeled CSV confusion matrices
│   ├── error_analysis/                # detailed_error_analysis.json
│   ├── robustness/                    # robustness_evaluation_results.json
│   └── calibration/                   # calibration_bins_*.json
├── reports/
│   ├── evaluation_protocol.md         # Formal methodology, isolation rules, and metrics definition
│   ├── validation_evaluation.md       # Validation partition benchmark (N=909)
│   ├── locked_test_evaluation.md      # Official locked test partition evaluation (N=928)
│   ├── error_analysis.md              # Systematic failure taxonomy
│   ├── robustness_report.md           # Perturbation stability and whitespace sensitivity
│   ├── leakage_verification.md        # Cryptographic verification and split isolation
│   ├── generalization_report.md       # Multi-partition comparison (Train -> Val -> Test)
│   └── final_evaluation_report.md     # Consolidated synthesis, scorecard, and handoff
├── figures/
│   ├── classification/                # Normalized confusion matrices and per-class F1 charts
│   ├── extraction/                    # Entity extraction metrics
│   ├── negation/                      # Negation diagnostics
│   ├── calibration/                   # Reliability diagrams
│   ├── robustness/                    # Robustness degradation
│   └── generalization/                # Validation vs. Locked Test comparison charts
├── tests/
│   ├── conftest.py                    # Pytest path isolation configuration
│   ├── test_data_loading.py           # Split loading and schema validation tests
│   ├── test_model_loading.py          # Artifact deserialization tests
│   ├── test_metrics.py                # Metric computation unit tests
│   ├── test_predictions.py            # Prediction runner and probability tests
│   ├── test_split_integrity.py        # Patient/encounter isolation tests
│   ├── test_leakage_checks.py         # Cryptographic hash and temporal order tests
│   ├── test_confidence_intervals.py   # Bootstrap bounds and determinism tests
│   └── test_reproducibility.py        # Dual-pass bit-exact reproducibility tests
├── requirements.txt                   # Module dependencies
└── README.md                          # This documentation file
```

---

## 3. Official Benchmark Results

### 3.1 Primary Task: Triage Urgency Level Classification
- **Classes**: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`
- **Validation**: Macro F1 = **0.7557**, Accuracy = **85.59%**, Critical Recall = **94.57%**
- **Locked Test (Official)**:
  - **Accuracy**: **87.07%** [95% CI: 84.40% – 89.68%]
  - **Macro F1**: **0.7814** [95% CI: 0.7444 – 0.8183]
  - **Weighted F1**: **0.8768** [95% CI: 0.8517 – 0.9014]
  - **Critical Class Recall**: **97.00%** (97 / 100 captured) [95% CI: 93.33% – 100.0%]
  - **Zero Critical-to-Low Misses**: 0 of 100 true `CRITICAL` records were misclassified as `LOW`.

### 3.2 Secondary Task: Toxicity Hazard Classification
- **Classes**: 8 organ/system toxicities (`NONE`, `HEPATIC`, `PULMONARY`, `HEMATOLOGIC`, `RENAL`, `NEUROPATHIC`, `DERMATOLOGIC`, `CARDIAC`)
- **Validation**: Macro F1 = **0.5214**, Accuracy = **77.56%**
- **Locked Test (Official)**:
  - **Accuracy**: **77.16%** [95% CI: 74.57% – 79.85%]
  - **Macro F1**: **0.5111** [95% CI: 0.4506 – 0.5649]
  - **Weighted F1**: **0.8228** [95% CI: 0.8020 – 0.8454]
  - **Hepatic Recall**: **88.31%** [95% CI: 80.68% – 94.94%]

### 3.3 Clinical Entity Extraction (NER)
- **Relaxed Overlap F1**: **0.7756** (Precision: 0.7145, Recall: 0.9037)
- **Exact Span F1**: **0.6439** (Precision: 0.6047, Recall: 0.7352)
- **Genomic Mutations**: Precision: **100.0%**, Recall: **81.13%**, F1: **0.8958**

---

## 4. Leakage & Reproducibility Audit
- **Patient Isolation**: **0 overlapping patients** across Train, Validation, and Locked Test ($1,000$ unique patients).
- **Encounter Isolation**: **0 overlapping encounters** ($2,038$ unique encounters).
- **Text Duplicate Isolation**: **0 identical clinical texts** across partitions.
- **Cryptographic Invariance**: 4 of 4 upstream processed datasets match exact SHA-256 hashes.
- **Deterministic Reproducibility**: Twin evaluation passes yielded bit-exact predictions ($\max|\Delta P| = 0.0$).

---

## 5. How to Run the Evaluation Suite

### 5.1 Execute Full Evaluation Pipeline
```powershell
py -3 stage-3-nlp-slm/evaluation/src/run_evaluation.py
```
Outputs all metrics JSONs, prediction CSVs, confusion matrices, calibration curves, and figures.

### 5.2 Execute Automated Test Suite
```powershell
py -3 -m pytest stage-3-nlp-slm/evaluation/tests/ -v
```
Runs all 20 evaluation unit tests (all passing).

### 5.3 Execute Full Stage 3 Test Suite
```powershell
py -3 -m pytest --import-mode=importlib stage-3-nlp-slm/ -v
```
Runs all 75 Stage 3 unit tests across Data Engineering (16), EDA (9), NLP (30), and Evaluation (20).

---

## 6. Evaluation Verdict & SLM Handoff
**Verdict**: **ACCEPTED WITH LIMITATIONS**

The baseline NLP pipeline provides an uncompromised, safety-conscious benchmark with 97.00% critical sensitivity and zero data leakage.
The upcoming **SLM Engineer** should use these locked-test metrics as the target comparison baseline, focusing specifically on resolving intermediate urgency ambiguity (`MEDIUM` vs `HIGH`) and contextually distinguishing drug dosages from physiological vital signs.
