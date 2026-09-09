# Stage 3 Clinical NLP: Train-Only Data Augmentation & Benchmark System

This module implements a mathematically verifiable, entity-preserving, train-only clinical data augmentation pipeline and benchmarking suite for Stage 3 Precision Oncology NLP and SLM workflows.

---

## Key Features & Invariants

1. **Zero Fact Fabrication**: Non-entity carrier phrases are varied using clinically verified synonyms; independent lab measurements and vitals are permuted safely; clinical framing is adjusted without altering clinical meaning.
2. **Entity Span Invariance**: Character offsets `[start, end]` are dynamically re-aligned with 100% mathematical precision:
   $$\forall e \in \text{ner\_entities}: \quad \text{text}[e[\text{"start"}]:e[\text{"end"}]] \equiv e[\text{"text"}]$$
3. **Strict Polarity Protection**: Negation cues (`no`, `not`, `without`, `denies`, `denied`, `resolved`) are strictly locked to prevent polarity reversal.
4. **Split Isolation & Locked-Test Protection**:
   - Only `train.parquet` receives augmentation.
   - `validation.parquet` remains 100% frozen for benchmarking.
   - `locked_test.parquet` is completely sealed and never accessed.
   - 0.0% cross-split patient or encounter leakage.
5. **10-Point Quality Control Gate**: Automated gatekeeper enforcing length bounds, text non-emptiness, label preservation, duplicate rejection, and Jaccard similarity bounds (`0.50 <= similarity <= 0.98`).
6. **Interactive Localhost Dashboard**: Real-time visual explorer on `http://localhost:8505` with clinical diffs, entity highlights, and metrics.

---

## Directory Architecture

```
data_augmentation/
├── configs/
│   └── augmentation_config.yaml         # Configuration parameters and safety thresholds
├── src/
│   ├── __init__.py
│   ├── annotation_propagation.py        # Exact character offset recalculation
│   ├── terminology_augmentation.py      # Non-entity carrier phrase synonym replacement
│   ├── sentence_augmentation.py         # Vitals & chemistry panel measurement permutation
│   ├── context_augmentation.py          # Symptom log framing variation
│   ├── clinical_paraphrasing.py         # Multi-strategy paraphrasing coordinator
│   ├── entity_preserving_augmentation.py# Top-level entity-preservation invariant controller
│   ├── duplicate_detection.py           # Exact & canonical duplicate detection
│   ├── leakage_checks.py                # Patient, encounter & text overlap audit
│   ├── quality_control.py               # 10-point QC filter
│   ├── augmentation_pipeline.py         # End-to-end dataset generation pipeline
│   └── run_augmentation_experiments.py # Benchmark execution across dataset variants
├── dashboard/
│   ├── index.html                       # Modern glassmorphic web dashboard
│   ├── styles.css                       # Premium dark-mode styling
│   ├── app.js                           # Interactive diff viewer and client logic
│   └── server.py                        # Localhost HTTP server (Port 8505)
├── data/
│   ├── intermediate/                    # Disk-cached MiniLM embeddings
│   └── augmented_train/                 # Parquet and JSONL datasets
├── results/
│   ├── diversity_metrics.json           # Lexical & distribution statistics
│   ├── qc_summary.json                  # QC filter acceptance/rejection log
│   ├── leakage_audit.json               # Zero-leakage verification log
│   └── summary_comparison.csv           # Benchmark comparison matrix
├── reports/
│   ├── augmentation_report.md           # Methodology & concrete clinical examples
│   ├── quality_report.md                # 10-point QC gate audit
│   ├── leakage_report.md                # Mathematical proof of partition isolation
│   ├── class_balance_report.md          # Imbalance mitigation analysis
│   └── final_data_upgrade_report.md     # Final evaluative synthesis & recommendation
├── tests/
│   └── test_augmentation.py             # Automated unit & regression test suite
└── requirements.txt
```

---

## Quick Start & Verification

### 1. Launch Interactive Localhost Dashboard
```bash
py -3.14 stage-3-nlp-slm/nlp/data_augmentation/dashboard/server.py
```
Open your browser at: **`http://localhost:8505`**

### 2. Run Test Suite
```bash
py -3.14 -m pytest stage-3-nlp-slm/nlp/data_augmentation/tests/
```

### 3. Run End-to-End Augmentation Pipeline
```bash
py -3.14 stage-3-nlp-slm/nlp/data_augmentation/src/augmentation_pipeline.py
```

### 4. Run Benchmark Experiments
```bash
py -3.14 stage-3-nlp-slm/nlp/data_augmentation/src/run_augmentation_experiments.py
```
