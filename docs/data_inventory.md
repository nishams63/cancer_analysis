# Pre-EDA Data Inventory & Upstream Provenance Catalog

**Stage**: Stage 4 — Exploratory Data Analysis (EDA) & Data Readiness Audit  
**Author**: EDA Engineer  
**Date**: September 2026  
**Status**: COMPLETE / VERIFIED  

---

## 1. Overview & Purpose

This data inventory documents all upstream clinical datasets, named entity recognition (NER) outputs, negation modules, and fine-tuning datasets discovered and inspected prior to executing the Stage 4 Exploratory Data Analysis.

In accordance with **Section 3** of the Clinical SLM Data Audit specification:
1. Complete repository inspection has been executed.
2. The Stage 4 dataset schema is empirically discovered from parquet metadata rather than assumed.
3. Upstream relationships between Stage 3 NLP assets and Stage 4 Data Engineering outputs are explicitly cataloged.
4. Cryptographic SHA-256 hashes and row counts are recorded for end-to-end reproducibility.

---

## 2. Discovered Datasets & Provenance

### 2.1 Stage 4 Primary Target Dataset (Audit Ingestion Target)

| Property | Value |
| :--- | :--- |
| **File Path** | `stage-4-slm/data-engineering/data/slm_finetune_dataset_v1.parquet` |
| **Format** | Apache Parquet (Snappy compressed, columnar) |
| **Row Count** | **5,706** accepted instruction-tuning pairs |
| **Column Count** | 17 columns |
| **Unique Patients** | 1,000 (`PT-000001` to `PT-001000`) |
| **Unique Notes** | 5,706 (`DOC-000001` to `DOC-006098`) |
| **SHA-256 Hash** | `0cf7f9ffd6b8413f94bcb58f62696a055d3848c6fe7385972f66b742297e9f02` |
| **Missingness** | 0 nulls across all 17 columns (100% complete) |
| **Read/Write Access** | **STRICTLY READ-ONLY** (no modification, filtering, or in-place rewriting) |

#### Empirical Schema: `slm_finetune_dataset_v1.parquet`

| Column Name | Data Type | Null Count | Description / Role | Example Value |
| :--- | :--- | :--- | :--- | :--- |
| `patient_id` | `string` | 0 | Canonical synthetic patient identifier | `PT-000001` |
| `note_id` | `string` | 0 | Unique clinical encounter document ID (maps to Stage 3 `document_id`) | `DOC-000001` |
| `clinical_note` | `string` | 0 | Full raw clinical consultation progress note text | `"ONCOLOGY CONSULTATION PROGRESS NOTE\nPatient ID: PT-000001..."` |
| `instruction` | `string` | 0 | Pinned instruction-tuning prompt prefix | `"Analyze the clinical note and provide the patient's Risk, Key Finding, and Action."` |
| `target_risk` | `string` | 0 | Draft target field: Toxicity hazard assessment | `"Increased no acute adverse toxicities and systemic toxicity hazard associated with Docetaxel (265.4 mg) therapy."` |
| `target_key_finding` | `string` | 0 | Draft target field: Molecular findings, drug administration & adverse events | `"Kras variant was identified; patient received docetaxel at 265.4 mg; developed no acute adverse toxicities."` |
| `target_action` | `string` | 0 | Draft target field: Recommended clinical action / monitoring | `"Continue standard clinical monitoring and maintain current Docetaxel regimen as tolerated."` |
| `ner_genes` | `object (list)` | 0 | Array of verified gene/biomarker entities from Stage 3 NER | `['KRAS']` |
| `ner_drugs` | `object (list)` | 0 | Array of verified antineoplastic drug entities from Stage 3 NER | `['Docetaxel']` |
| `ner_dosages` | `object (list)` | 0 | Array of verified dosage expressions from Stage 3 NER | `['265.4 mg']` |
| `ner_adverse_events` | `object (list)` | 0 | Array of verified adverse toxicities from Stage 3 NER | `['no acute adverse toxicities']` |
| `entity_check_status`| `string` | 0 | Entity preservation gate status | `'PASS'` |
| `entity_coverage` | `float64` | 0 | Proportion of reference entities preserved in target fields | `1.0` (100% across all accepted rows) |
| `missing_entities` | `object (list)` | 0 | List of entities omitted from target | `[]` (empty list across accepted rows) |
| `invented_entities`| `object (list)` | 0 | List of entities hallucinated into target | `[]` (empty list across accepted rows) |
| `generation_model_version` | `string` | 0 | Generator model identifier and version | `'clinical-draft-target-generator-v1.0.0'` |
| `split` | `string` | 0 | Patient-partitioned split assignment | `'TRAIN'` (3,996), `'TEST'` (861), `'VALIDATION'` (849) |

---

### 2.2 Stage 4 Secondary Data Engineering Outputs

| File Path | Rows | Description |
| :--- | :--- | :--- |
| `stage-4-slm/data-engineering/data/rejected_pairs.parquet` | 392 | Rejected note-target pairs with detailed failure reasons and coverage scores |
| `stage-4-slm/data-engineering/data/generation_log.parquet` | 6,098 | Complete audit log capturing raw generation attempts, prompts, hyperparameters, and execution timing |
| `stage-4-slm/data-engineering/reports/data_quality_report.json` | - | Summary statistics, circuit breaker metrics (6.43% rejection rate), and patient leakage audit |
| `stage-4-slm/data-engineering/reports/manual_review_audit.md` | - | Human-in-the-loop qualitative review of 30 PASS and 30 REJECT cases |

---

### 2.3 Stage 3 Upstream NLP & NER Reference Assets

| File Path | Content & Purpose |
| :--- | :--- |
| `stage-3-nlp-slm/data-engineering/data/processed/clinical_nlp_dataset_v1.parquet` | 6,098 total clinical notes with document metadata (`urgency_level`, `hazard_type`, `document_date`, `index_date`). Used to verify joins, temporal ordering, and reference risk annotations. |
| `stage-3-nlp-slm/nlp/data/outputs/nlp_features_train.parquet` | 4,261 training records containing extracted concept counts (`total_concepts`, `affirmed_concepts`, `negated_concepts`, `historical_concepts`). |
| `stage-3-nlp-slm/nlp/data/outputs/nlp_features_val.parquet` | 909 validation records containing concept counts and polarity extractions. |
| `stage-3-nlp-slm/nlp/results/metrics/baseline_validation_metrics.json` | Upstream NER performance baseline (`precision: 0.7191`, `recall: 0.8797`, `f1: 0.7670`). Serves as calibration baseline. |
| `stage-3-nlp-slm/nlp/src/negation_detection.py` | Production-grade NegEx rule-based negation detector with `resolve_concept_polarity`, `PRE_NEGATION_TRIGGERS`, `POST_NEGATION_TRIGGERS`, `PSEUDO_NEGATIONS`, and `SCOPE_TERMINATORS`. Reused for negation-scope audit. |

---

## 3. Relationships & Foreign Key Mapping

- **Primary Document Key**: `stage-4-slm/data-engineering/data/slm_finetune_dataset_v1.parquet.note_id` maps identically to `stage-3-nlp-slm/.../clinical_nlp_dataset_v1.parquet.document_id`. Exactly 5,706 of 5,706 rows resolve to Stage 3 parent records.
- **Patient Identifier**: `patient_id` matches across all stages (`PT-000001` through `PT-001000`).

---

## 4. Software Dependencies & Runtime Environment

The EDA audit utilizes the locked Python virtual environment at `.venv` (Python 3.11.16):

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `pandas` | 2.2.3 | Parquet ingestion, structured slicing, grouping |
| `numpy` | 2.2.3 | Vectorized statistical calculations, percentile arrays |
| `pyarrow` | 19.0.1 | Columnar binary reading, metadata serialization |
| `pyyaml` | 6.0.2 | Parsing `config.yaml` and `critical_terms.yaml` |
| `scipy` | 1.15.2 | Kolmogorov-Smirnov 2-sample tests, Jensen-Shannon divergence, Chi-square |
| `scikit-learn` | 1.6.1 | TF-IDF vectorization for near-duplicate cosine similarity audit |
| `tiktoken` | 0.8.0 | BPE subword tokenization (`cl100k_base` / 100,277 vocabulary) |
| `transformers` | 5.17.0 | HuggingFace tokenizer utilities and fallback verification |
| `matplotlib` | 3.10.0 | High-resolution rendering for 12 publication charts |
| `seaborn` | 0.13.2 | Statistical distribution plots and density heatmaps |
| `pytest` | 8.3.4 | Automated unit and integration test framework |

---

## 5. Pre-Audit Data Verification Sign-Off

- [x] Primary Stage 4 Parquet file exists and is intact (`slm_finetune_dataset_v1.parquet`).
- [x] Cryptographic SHA-256 hash verified: `0cf7f9ffd6b8413f94bcb58f62696a055d3848c6fe7385972f66b742297e9f02`.
- [x] Dataset row count matches: exactly **5,706** accepted records.
- [x] Zero missingness across all 17 schema fields.
- [x] Upstream Stage 3 files and negation code located and verified accessible.
- [x] Python environment equipped with all necessary statistical, tokenization, and plotting libraries.
- [x] Ready to commence Stage 4 EDA audit module development in `stage-4-slm/eda/`.
