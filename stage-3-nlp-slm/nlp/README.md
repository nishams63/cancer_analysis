# Stage 3 — Clinical NLP Engineering Module

## Overview
This module contains the reproducible, scientifically grounded **Clinical NLP Engineering Pipeline** for Stage 3 of **“Personalized Precision Medicine for Oncology Treatment Optimization”**.

It bridges raw clinical oncology narratives with structured, negation-aware feature representations, establishes transparent baseline classification benchmarks, and provides the foundational inputs for the downstream **Small Language Model (SLM) Engineer**.

> **Strict Operational Boundary**:
> - All upstream Data Engineering (`stage-3-nlp-slm/data-engineering/`) and EDA (`stage-3-nlp-slm/eda/`) files are **100% READ-ONLY** and were not modified.
> - The locked test set (`locked_test.parquet`, 928 documents) was **STRICTLY HELD OUT** and never touched for model training, feature selection, or hyperparameter tuning.

---

## Directory Architecture
```text
stage-3-nlp-slm/nlp/
├── configs/
│   └── nlp_config.yaml                # Master configuration (paths, vocabularies, model parameters)
├── src/
│   ├── __init__.py                    # Module declaration
│   ├── config.py                      # Dynamic path resolver and configuration loader
│   ├── data_loader.py                 # Read-only data loader with schema validation
│   ├── text_cleaning.py               # Conservative text cleaner (preserves numbers, units, negations)
│   ├── text_normalization.py          # Unicode NFKC decomposition, unit and acronym standardization
│   ├── sentence_processing.py         # Clinical sentence boundary detector and tokenizer
│   ├── negation_detection.py          # NegEx-inspired rule-based clinical negation & temporality detector
│   ├── clinical_concepts.py           # Concept extraction engine for drugs, mutations, dosages, adverse events
│   ├── feature_extraction.py          # Negation-aware TF-IDF vectorizer and structured count matrix
│   ├── label_preparation.py           # Target label encoders and balanced inverse class weights
│   ├── baseline.py                    # Class-weighted Logistic Regression baselines and NER evaluation
│   ├── inference.py                   # Production-ready inference engine returning structured JSON
│   └── utils.py                       # Checksum verifiers, seed handlers, serialization helpers
├── data/
│   ├── intermediate/                  # Tokenized caches and intermediate representations
│   └── outputs/                       # Extracted tabular feature matrices (nlp_features_*.parquet)
├── artifacts/
│   ├── encoders/                      # Serialized label encoders and class weights
│   └── tokenizers/                    # Serialized TF-IDF vectorizer and StandardScaler
├── results/
│   ├── feature_outputs/               # Summary distributions of extracted clinical features
│   ├── metrics/                       # Baseline validation classification reports and confusion matrices
│   └── predictions/                   # Document-level validation predictions with confidence scores
├── reports/
│   ├── nlp_task_definition.md         # Formal definition of classification and extraction tasks
│   ├── preprocessing_report.md        # Detailed text cleaning and normalization report
│   ├── baseline_report.md             # Baseline model architecture, training, and validation metrics
│   ├── error_analysis.md              # Systematic failure mode analysis and clinical examples
│   ├── leakage_audit.md               # Proof of zero patient, encounter, and temporal leakage
│   └── nlp_engineering_report.md      # Comprehensive end-to-end technical report
├── notebooks/
│   └── nlp_pipeline_experiment.ipynb  # Interactive step-by-step pipeline experiment notebook
├── tests/
│   ├── conftest.py                    # Pytest configuration and sys.path management
│   ├── test_data_loader.py            # Data loading, column verification, and split integrity
│   ├── test_text_cleaning.py          # Cleaning invariants, preservation of numbers and negations
│   ├── test_text_normalization.py     # NFKC, unit, and acronym normalization
│   ├── test_negation_detection.py     # Negation scoping (affirmative vs. negated vs. historical)
│   ├── test_feature_extraction.py     # Feature matrix shape, deterministic extraction, non-null assertions
│   ├── test_label_preparation.py      # Label encoding, inverse transforms, and class weights
│   ├── test_baseline.py               # Model training, inference, probability bounds, and metrics
│   └── test_split_integrity.py        # Strict patient/encounter isolation and read-only checksums
├── requirements.txt                   # Explicit dependency specifications
└── README.md                          # Module documentation and handoff directives
```

---

## Baseline Model Benchmarks (Official Validation Set, N = 909)

| Task Modality | Target Variable | Classes | Primary Metric | Validation Performance |
| :--- | :--- | :---: | :--- | :---: |
| **Multi-Class Classification (Primary)** | `urgency_level` | 4 classes | **Macro F1 Score** | **0.7557** (Accuracy: 85.59%) |
| | | | **Critical Patient Recall** | **94.57%** (`CRITICAL` F1 = 0.9560) |
| **Multi-Class Classification (Secondary)**| `hazard_type` | 8 classes | **Macro F1 Score** | **0.5214** (Weighted F1: 0.8230) |
| **Named Entity Recognition (NER)** | `ner_entities` | 4 classes | **Mean Span F1** | **76.70%** (Precision: 75.84%) |

---

## Quick Start & Execution

### 1. Run Automated Tests
```powershell
py -3 -m pytest stage-3-nlp-slm/nlp/tests/ -v
```
*Expected: 30 passed in ~2.8s.*

### 2. Run End-to-End Baseline Training & Validation
```powershell
py -3 -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('stage-3-nlp-slm/nlp/src').resolve())); from baseline import run_baseline_pipeline; run_baseline_pipeline()"
```

### 3. Run Inference on a Clinical Note
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("stage-3-nlp-slm/nlp/src").resolve()))
from inference import get_inference_engine

engine = get_inference_engine()
note = (
    "ONCOLOGY CONSULTATION: 62yo female with Stage III NSCLC, EGFR L858R positive. "
    "Patient received Cisplatin 75 mg/m2. Denies acute chest pain or dyspnea. "
    "Reports manageable mild fatigue. ECOG PS: 1."
)
response = engine.analyze_document(note, document_id="NOTE-001")
print(response)
```

### 4. Launch the Interactive Experiment Notebook
```powershell
jupyter notebook stage-3-nlp-slm/nlp/notebooks/nlp_pipeline_experiment.ipynb
```

---

## Handoff Directives for the SLM Engineer

1. **Benchmark Standard**: Fine-tuned Small Language Models (1B–3B parameters) must match or exceed the linear baseline benchmarks:
   - Urgency Classification: **Macro F1 $\ge 0.7557$**
   - Critical Safety Triage Recall: **$\ge 94.57\%$**
2. **Conditioning Format**: Inject `document_type` as an instruction prefix:
   ```text
   [INST] Document Type: {document_type}
   Clinical Note: {text}
   Generate a 2-sentence bedside clinical handoff briefing. [/INST]
   ```
3. **Length Bounds**: Ground-truth `slm_summary` targets average $34.2 \pm 4.1$ words. Bound generation to `max_new_tokens = 64` and `min_new_tokens = 20`.
4. **Data Isolation**: Never train or evaluate the SLM on `locked_test.parquet`. All development, tuning, and prompt iteration must be performed strictly on `TRAIN` and evaluated on `VALIDATION`.
