# Stage 3 — NLP + SLM: Data Engineering Module

## Overview
This module contains the research data engineering pipeline for **Stage 3 (NLP + SLM)** of the project **“Personalized Precision Medicine for Oncology Treatment Optimization”**.

Its sole purpose is to build the highest-quality, reliable, reproducible, leakage-free, and de-identified NLP dataset for downstream NLP modeling (Urgency Triage, Toxicity Hazard Classification, Clinical NER) and Small Language Model (SLM) fine-tuning (2-sentence bedside briefings).

> **Disclaimer**: This dataset and pipeline are built for oncology research and prototyping. It is NOT clinically validated and MUST NOT be used for direct patient treatment decisions.

---

## Directory Structure
```
stage-3-nlp-slm/
└── data-engineering/
    ├── configs/
    │   └── data_config.yaml            # Master configuration (cohort, taxonomy, thresholds, splits)
    ├── data/
    │   ├── raw/
    │   │   ├── raw_clinical_notes_v1.jsonl
    │   │   └── raw_clinical_notes_metadata.csv
    │   └── processed/
    │       ├── clinical_nlp_dataset_v1.parquet   # Full clean canonical dataset (6,098 docs)
    │       ├── clinical_nlp_dataset_v1.jsonl
    │       ├── clinical_nlp_dataset_v1.csv
    │       ├── train.parquet / train.jsonl       # Train split (700 patients, 4,261 docs)
    │       ├── validation.parquet / .jsonl       # Val split (150 patients, 909 docs)
    │       ├── locked_test.parquet / .jsonl      # Locked test split (150 patients, 928 docs)
    │       └── data_dictionary.md                # Schema and column definitions
    ├── reports/
    │   ├── data_quality_report.md      # Full audit ledger, text statistics, label distributions
    │   ├── data_provenance.md          # Upstream lineage from Stage 1 master patient cohort
    │   ├── leakage_audit.md            # Temporal cutoff and outcome leak exclusion certification
    │   ├── privacy_audit.md            # HIPAA Safe Harbor direct identifier scrubbing report
    │   └── dataset_card.md             # Standard research dataset card and limitations
    ├── src/
    │   ├── config.py                   # Dynamic environment path resolver and vocabulary loader
    │   ├── de_identification.py        # Regex-based PII scrubber & validation scanner
    │   ├── text_preprocessing.py       # NFKC normalization, whitespace cleanup, metric calculations
    │   ├── data_generator.py           # Anchored clinical note generator (with injection testing)
    │   ├── data_validation.py          # Quality gate validator & test assertions
    │   └── data_pipeline.py            # Master end-to-end execution pipeline
    ├── tests/
    │   ├── test_data_validation.py     # Schema, mandatory non-null, ID uniqueness, thresholds
    │   ├── test_split_integrity.py     # Zero patient & encounter leakage, split proportions
    │   └── test_text_preprocessing.py  # Normalization, negation, dosage, and PII masking tests
    └── README.md
```

---

## Dataset Summary
- **Total Processed Documents**: 6,098
- **Unique Patients**: 1,000 (`PT-000001` to `PT-001000`)
- **Unique Encounters**: 2,038
- **Document Types**:
  - `oncology_consultation` (2,033 docs)
  - `nurse_intake_note` (1,577 docs)
  - `patient_symptom_log` (1,487 docs)
  - `pathology_report` (1,001 docs)
- **Target Tasks & Labels**:
  - `urgency_level`: `LOW` (4,140), `HIGH` (836), `CRITICAL` (577), `MEDIUM` (545)
  - `hazard_type`: `NONE` (4,845), `HEPATIC` (474), `PULMONARY` (367), `HEMATOLOGIC` (144), `RENAL` (127), `NEUROPATHIC` (60), `DERMATOLOGIC` (46), `CARDIAC` (35)
  - `ner_entities`: JSON span list `[{"start", "end", "label", "text"}]` for `GENE_MUTATION`, `DRUG_NAME`, `DOSAGE`, `ADVERSE_EVENT`
  - `slm_summary`: Exactly 2-sentence clinical bedside briefing

---

## Leakage Prevention & Privacy
1. **Zero Patient / Encounter Leakage**:
   - Split strictly by `patient_id` using Group K-Split (seed = 42).
   - Train = 700 patients, Validation = 150 patients, Locked Test = 150 patients.
   - Cross-split patient overlap: **0.0%**.
   - Cross-split encounter overlap: **0.0%**.
2. **Temporal & Outcome Leakage Guardrails**:
   - Index time established as the encounter observation date.
   - Hard constraint: `document_date <= index_date`.
   - Post-treatment outcome statements (e.g., progression day 180, retrospective survival) were audited and filtered out (12 records dropped).
3. **De-Identification**:
   - Direct identifiers (patient names, phone numbers, emails, MRNs, provider names) are scrubbed via HIPAA Safe Harbor regex patterns into token placeholders (`[NAME]`, `[PHONE]`, `[EMAIL]`, `[MRN]`).
   - 16 direct identifier occurrences sanitized.
   - Post-cleaning scan verified zero remaining direct PII.

---

## Execution & Verification
To regenerate the dataset from scratch:
```powershell
py -3 stage-3-nlp-slm/data-engineering/src/data_pipeline.py
```

To run the automated validation test suite:
```powershell
py -3 -m pytest stage-3-nlp-slm/data-engineering/tests/ -v
```

All 16 test cases verify schema completeness, categorical vocabularies, length bounds, temporal ordering, negation retention, dosage preservation, PII scrubbing, and split isolation.
