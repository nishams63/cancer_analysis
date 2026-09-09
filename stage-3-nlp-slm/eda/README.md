# Stage 3 — Clinical NLP + SLM: Exploratory Data Analysis (EDA)

## Overview
This module contains the rigorous, evidence-based **Exploratory Data Analysis (EDA)** for Stage 3 (Clinical NLP & Small Language Models) of the project **“Personalized Precision Medicine for Oncology Treatment Optimization”**.

> **Strict Operational Boundary**: The EDA module operates in a **100% READ-ONLY** capacity. No source records were added, dropped, modified, re-labeled, or re-balanced. The locked test set was inspected solely for descriptive distribution alignment and was not used for model selection.

---

## Directory Architecture
```text
stage-3-nlp-slm/
└── eda/
    ├── notebooks/
    │   └── nlp_eda.ipynb              # Executable Jupyter notebook with interactive analysis & tables
    ├── src/
    │   ├── __init__.py
    │   └── eda_utils.py               # Pure analysis & visualization utility functions (deterministic)
    ├── figures/
    │   ├── text_length/               # 4 high-DPI plots (word/char distributions, length by doc/class)
    │   ├── vocabulary/                # 3 high-DPI plots (Zipf curve, top 20 terms, antineoplastic agents)
    │   ├── document_types/            # 2 high-DPI plots (composition, breakdown across splits)
    │   ├── labels/                    # 3 high-DPI plots (urgency classes, hazard types, heatmap)
    │   ├── temporal/                  # 2 high-DPI plots (monthly timeline, inter-encounter days)
    │   └── patients/                  # 2 high-DPI plots (documents/patient, encounters/patient)
    ├── reports/
    │   └── eda_report.md              # Exhaustive 20-section formal EDA report
    ├── results/
    │   └── eda_summary.json           # Machine-readable summary of all computed statistics & metrics
    ├── tests/
    │   └── test_eda.py                # Automated pytest suite enforcing read-only invariants & split integrity
    └── README.md
```

---

## Key Dataset Findings at a Glance

| Metric | Empirical Value | Context / Meaning for Modeling |
| :--- | :---: | :--- |
| **Total Validated Records** | **6,098** | Full canonical dataset (`clinical_nlp_dataset_v1.parquet`) |
| **Unique Patients** | **1,000** | Cohort identifiers `PT-000001` through `PT-001000` |
| **Unique Encounters** | **2,038** | Average 2.99 documents per encounter |
| **Mean Word Count** | **105.90** | Max 143 words; comfortably fits within `max_length = 256` tokens |
| **Corpus Vocabulary** | **3,181 terms** | Rich oncology entities, Type-Token Ratio = 0.0049 |
| **Clinical Negation Rate** | **100.0%** | Mean 2.30 negation cues/doc; **requires contextual transformer heads** |
| **Urgency Imbalance** | **7.60:1** | `LOW` (67.9%), `HIGH` (13.7%), `CRITICAL` (9.5%), `MEDIUM` (8.9%) |
| **Hazard Imbalance** | **138.43:1** | `NONE` (79.5%), `HEPATIC` (7.8%), ..., `CARDIAC` (0.57%) |
| **Cross-Split Patient Overlap** | **0.0%** | Strict patient-level grouping (`GroupKFold`, 700 Train / 150 Val / 150 Test) |
| **Cross-Split Encounter Overlap**| **0.0%** | Complete isolation between all splits |
| **Prospective Outcome Leakage** | **0.0%** | 0 forbidden cues detected; $\text{document\_date} \le \text{index\_date}$ strictly enforced |
| **Read-Only SHA256 Invariance** | **PASSED** | Source parquet hashes identical before and after EDA execution |

---

## Quick Execution

### 1. Run Automated Tests
```powershell
py -3 -m pytest stage-3-nlp-slm/eda/tests/ -v
```
*Expected Result: 9 passed in ~7.5s with zero errors or warnings.*

### 2. Re-run End-to-End Pipeline & Generate Figures
```powershell
py -3 -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('stage-3-nlp-slm/eda/src').resolve())); from eda_utils import run_full_eda_pipeline; run_full_eda_pipeline()"
```

### 3. Launch Interactive Jupyter Notebook
```powershell
jupyter notebook stage-3-nlp-slm/eda/notebooks/nlp_eda.ipynb
```

---

## Handoff Directives for Downstream Engineers

### For the NLP Engineer:
1. **Model Backbone**: Do **NOT** use TF-IDF or Bag-of-Words. Because 100% of notes contain negation cues, use `Bio_ClinicalBERT` or `PubMedBERT` to preserve semantic negation scope.
2. **Class Imbalance**: Apply class-weighted cross-entropy loss for 4-class `urgency_level` and Focal Loss ($\gamma = 2.0$) for 8-class `hazard_type`.
3. **Context Length**: Set `max_length = 256` tokens. This covers 100% of documents without truncating clinical content while minimizing GPU memory.

### For the SLM Engineer:
1. **Structured Prompt Anchors**: Leverage `document_type` as an instruction prefix (`[INST] Document Type: {document_type} ... [/INST]`).
2. **Target Length**: Target summaries are strictly 2 sentences (mean 34 words). Cap generation at `max_new_tokens = 64` to prevent repetitive hallucination.
3. **Training Method**: Use 4-bit QLoRA on a 1B–3B parameter foundation model (e.g., `Llama-3.2-1B-Instruct` or `Qwen2.5-1.5B-Instruct`).
