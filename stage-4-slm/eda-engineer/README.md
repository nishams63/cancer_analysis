# Stage 4 — EDA Engineer: Production-Grade Clinical SLM Data Audit

## 1. Purpose & Mandate
This package implements an end-to-end, production-grade **Exploratory Data Analysis (EDA) and Training-Data Readiness Audit** for the instruction-tuning dataset produced by Stage 4 Data Engineering (`slm_finetune_dataset_v1.parquet`).

The audit answers the central engineering and clinical question:
> **"Is the Stage 4 dataset statistically, linguistically, structurally, and clinically suitable for Stage 5 Small Language Model (SLM) fine-tuning, and what risks could cause the model to learn incorrect behavior?"**

In strict accordance with audit guidelines:
- **Zero In-Place Modifications**: The audit operates strictly in read-only mode (`DETECT -> MEASURE -> REPORT -> RECOMMEND`).
- **No SLM Training**: Focuses exclusively on data readiness, bias detection, and safety auditing.
- **Empirical Real Data**: All distributions and evaluations are computed from the real 5,706 clinical instruction records.

---

## 2. Architecture & File Layout

```text
stage-4-slm/eda/
├── README.md                      # Comprehensive package documentation and audit guide
├── requirements.txt               # Pinned dependencies (pandas, scipy, tiktoken, etc.)
├── config.yaml                    # Global audit configuration and quality thresholds
├── config/
│   └── critical_terms.yaml        # Curated oncology lexicon (drugs, biomarkers, regimens, toxicities)
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_loader.py             # Read-only ingestion, SHA-256 verification, Stage 3 provenance
│   ├── profiling.py               # Cardinality, row-level missingness, duplicate analysis
│   ├── token_analysis.py          # BPE token distributions, context overflow & tail truncation risk
│   ├── vocabulary_analysis.py     # Vocabulary profiling, hapax legomena, subword fragmentation
│   ├── entity_analysis.py         # Entity density per note & reference retention rates
│   ├── negation_analysis.py       # Clinical polarity preservation & negation-flip audit
│   ├── risk_analysis.py           # Risk tier distribution (Low/Mod/High) & stratified loss
│   ├── split_analysis.py          # Independent patient isolation & cross-split leakage audit
│   ├── drift_analysis.py          # Temporal distributions & Kolmogorov-Smirnov/Jensen-Shannon drift
│   ├── quality_flags.py           # Configurable threshold evaluation (PASS/WARNING/CRITICAL)
│   ├── visualization.py           # 12 publication-quality PNG charts generator (300 DPI)
│   ├── report_generator.py        # Markdown and JSON report compiler
│   └── pipeline.py                # End-to-end CLI orchestrator
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Test environment path fixtures
│   ├── test_data_loader.py        # Schema enforcement, hashing, missingness tests
│   ├── test_token_analysis.py     # Token distribution, context overflow & truncation risk tests
│   ├── test_entity_analysis.py    # Entity density & retention tests
│   ├── test_negation_analysis.py  # Negation detection & flip detection tests
│   ├── test_split_analysis.py     # Zero leakage & injected leakage tests
│   └── test_quality_flags.py      # Quality threshold evaluation & status decision tests
├── reports/
│   ├── eda_report.md              # Human-readable clinical readiness audit report
│   ├── eda_results.json           # Machine-readable audit metrics payload
│   ├── negation_review_cases.parquet # Granular audit cases for flagged negation flips
│   └── leakage_eda.json           # Independent cross-split leakage audit results
└── figures/                       # 12 high-resolution audit visualization figures
    ├── source_token_length.png
    ├── target_token_length.png
    ├── token_length_comparison.png
    ├── risk_distribution.png
    ├── entity_density_distribution.png
    ├── entity_density_by_risk.png
    ├── entity_retention_by_type.png
    ├── entity_retention_by_risk.png
    ├── critical_term_frequency.png
    ├── critical_term_fragmentation.png
    ├── negation_analysis.png
    └── split_distribution.png
```

---

## 3. Inputs & Upstream Data Provenance

1. **Stage 4 Target Dataset**: `stage-4-slm/data-engineering/data/slm_finetune_dataset_v1.parquet`
   - Row count: 5,706 accepted instruction-tuning pairs.
   - Verified SHA-256: `0cf7f9ffd6b8413f94bcb58f62696a055d3848c6fe7385972f66b742297e9f02`.
   - Columns: `patient_id`, `note_id`, `clinical_note`, `instruction`, `target_risk`, `target_key_finding`, `target_action`, `ner_genes`, `ner_drugs`, `ner_dosages`, `ner_adverse_events`, `entity_check_status`, `entity_coverage`, `missing_entities`, `invented_entities`, `generation_model_version`, `split`.
2. **Stage 3 Reference NLP Dataset**: `stage-3-nlp-slm/data-engineering/data/processed/clinical_nlp_dataset_v1.parquet`
   - Provides reference metadata (`urgency_level`, `hazard_type`, `document_date`).

---

## 4. Installation & Environment

The EDA audit utilizes Python 3.11 with locked virtual environment dependencies:

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install EDA dependencies
pip install -r stage-4-slm/eda/requirements.txt
```

---

## 5. Configuration (`config.yaml`)

Audit thresholds, paths, and tokenizer configurations are managed via `stage-4-slm/eda/config.yaml`:
```yaml
paths:
  input_dataset: "stage-4-slm/data-engineering/data/slm_finetune_dataset_v1.parquet"
  reports_dir: "stage-4-slm/eda/reports"
  figures_dir: "stage-4-slm/eda/figures"

token_analysis:
  tokenizer_model: "cl100k_base"
  context_limits: [512, 1024, 2048, 4096]
  target_context_limit: 4096

quality_thresholds:
  context_overflow_rate:
    warning: 0.01
    critical: 0.05
  negation_flip_rate:
    warning: 0.01
    critical: 0.05
  entity_retention_rate:
    warning: 0.95
    critical: 0.90
  patient_leakage:
    critical: 0
  risk_class_imbalance:
    warning: 0.20
```

---

## 6. Running the Audit & Tests

### Automated Test Suite
```bash
.venv\Scripts\pytest -v stage-4-slm/eda/tests
```

### Complete Pipeline Execution
```bash
.venv\Scripts\python stage-4-slm/eda/src/pipeline.py --config stage-4-slm/eda/config.yaml
```

CLI Options:
- `--input <path>`: Override input parquet path.
- `--output <path>`: Override reports output directory.
- `--figures-dir <path>`: Override figures directory.
- `--generate-figures`: Enable rendering of 12 publication charts (default: True).
- `--strict`: Terminate with non-zero exit code if status is NOT READY or WARNING.

---

## 7. Audit Modules & Key Findings

| Module | Purpose | Key Metric / Finding | Status |
| :--- | :--- | :--- | :--- |
| `data_loader` & `profiling` | Schema enforcement & missingness | 5,706 rows, 0 missing cells, 0 duplicate rows | **PASS** |
| `token_analysis` | BPE subword length & overflow | Max sequence: 404 tokens; 0.0% overflow at 4,096 | **PASS** |
| `vocabulary_analysis` | Lexicon & BPE subword fragmentation | 65 oncology terms audited; 43 multi-token terms flagged | **PASS / ADVISORY** |
| `entity_analysis` | Stage 3 NER reference retention | 18,541 total entities; 100.00% retention in targets | **PASS** |
| `split_analysis` | Cross-split patient isolation | 0 patient leakage across Train (70%), Val (15%), Test (15%) | **PASS** |
| `drift_analysis` | Two-sample KS & JS tests | Minimal drift (KS $p > 0.60$, JS $< 0.03$) | **PASS** |
| `risk_analysis` | Risk tier representation | Low (67.8%), High (23.0%), Moderate (9.2%) | **WARNING** |
| `negation_analysis` | Clinical polarity preservation | 1,026 cases (14.88%) of contradictory risk assertions | **CRITICAL** |

---

## 8. Final Readiness Decision & Rationale

```text
========================================
EDA ENGINEER — DATA READINESS AUDIT
========================================
Records:              5706
Patients:             1000
Context overflow:     0.00%
Entity retention:     100.00%
Negation flips:       14.88%
Patient leakage:      0
Critical issues:      1
Warnings:             1

FINAL STATUS:
NOT READY
========================================
```

### Blocking Finding:
- **Negation Polarity Collision (14.88% Flip Rate)**:
  In notes where patients had zero acute toxicities (`ner_adverse_events = ['no acute adverse toxicities']`), the draft target generator generated:
  `"Increased no acute adverse toxicities and systemic toxicity hazard associated with [Drug] therapy."`
  and `"developed no acute adverse toxicities"`.
  This contradicts clinical logic by asserting an absence of toxicities as an "Increased hazard".
- **Recommended Action Before Stage 5**:
  Patch `stage-4-slm/data-engineering/src/summary_generator.py` to route notes with `no acute adverse toxicities` to `"Baseline toxicity risk..."` and `"exhibits stable tolerance with no acute toxicities"`, re-export the fine-tuning parquet, and re-certify readiness.

---

## 9. Privacy, Security & Reproducibility
- **PHI Scrubbing**: No unhashed patient identifiers or raw note excerpts are logged in public reports.
- **Deterministic Evaluation**: Cryptographic hashes (`SHA-256: 0cf7f9ffd6b8413f94bcb58f62696a055d3848c6fe7385972f66b742297e9f02`), pinned seeds (`42`), and deterministic tokenizers ensure 100% bit-for-bit reproducibility.
