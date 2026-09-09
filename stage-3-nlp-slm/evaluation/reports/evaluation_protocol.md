# Clinical NLP Independent Evaluation Protocol — Stage 3

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 3 — Clinical NLP & Baseline Systems Independent Evaluation  
**Protocol Version**: `1.0.0`  
**Evaluator Role**: Independent Evaluation Engineer  
**Date**: 2026-09-09  
**Status**: APPROVED & LOCKED  

---

## 1. Primary Evaluation Objective
The objective of this evaluation is to provide an **independent, unbiased, and statistically defensible benchmark** of the frozen Stage 3 Clinical NLP pipeline and its baseline models.

This protocol enforces strict non-interventional evaluation:
- Zero retraining, fine-tuning, or hyperparameter adjustment.
- Zero modification of upstream datasets (`data-engineering/`), exploratory data analysis (`eda/`), or NLP engineering pipeline source code (`nlp/`).
- Independent verification of all claimed metrics, leakage invariants, and reproducibility properties.
- Strictly isolated, single-pass evaluation of the virgin **Locked Test Set** ($N=928$, 150 unique patients).

---

## 2. System & Artifacts Evaluated

### 2.1 Software & Dataset Versions
- **Dataset Version**: `clinical_nlp_dataset_v1.parquet` (SHA-256: `426ea0c51f354a5fa9e1177ec0d8fabc0ed26ed708f748ee8f8f4b0f18554230`)
- **NLP System Version**: `stage-3-nlp-slm/nlp` v1.0.0
- **Random Seed**: Fixed globally to `42`.

### 2.2 Frozen Artifacts Under Evaluation
All artifacts evaluated reside under `stage-3-nlp-slm/nlp/artifacts/`:
1. **Urgency Baseline Model**: `urgency_baseline_model.joblib` (Class-weighted `LogisticRegression`, $C=1.0$, `lbfgs`, 1,000 iterations).
2. **Hazard Baseline Model**: `hazard_baseline_model.joblib` (Class-weighted `LogisticRegression`, $C=0.5$, `lbfgs`, 1,000 iterations).
3. **TF-IDF Vectorizer**: `tokenizers/tfidf_vectorizer.joblib` (1,000 unigram/bigram features with `neg_` and `hist_` prefixes).
4. **Feature Scaler**: `tokenizers/feature_scaler.joblib` (`StandardScaler` fitted on 12 structured count features).
5. **Numeric Feature Schema**: `tokenizers/numeric_feature_cols.joblib` (12 numerical columns).
6. **Urgency Label Encoder**: `encoders/urgency_encoder.joblib` (Classes: `CRITICAL`, `HIGH`, `LOW`, `MEDIUM`).
7. **Hazard Label Encoder**: `encoders/hazard_encoder.joblib` (Classes: `CARDIAC`, `DERMATOLOGIC`, `HEMATOLOGIC`, `HEPATIC`, `NEUROPATHIC`, `NONE`, `PULMONARY`, `RENAL`).

---

## 3. Scope of Evaluated NLP Tasks

The evaluation assesses the four distinct tasks actually implemented by the NLP Engineer:

| Task Modality | Task Name | Vocabulary / Domain | Ground Truth Type | Primary Target Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Multi-class Classification** | Triage Urgency Level | 4 classes: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` | Document label (`urgency_level`) | Macro F1 & Critical Class Recall |
| **Multi-class Classification** | Toxicity Hazard Type | 8 classes: `NONE`, `HEPATIC`, `PULMONARY`, ..., `CARDIAC` | Document label (`hazard_type`) | Macro F1 & Weighted F1 |
| **Named Entity Extraction** | Clinical Concept Recognition | 4 types: `GENE_MUTATION`, `DRUG_NAME`, `DOSAGE`, `ADVERSE_EVENT` | Character span offsets (`ner_entities`) | Exact & Relaxed Span F1 |
| **Contextual Attribution** | Clinical Negation Scoping | 4 polarities: `AFFIRMED`, `NEGATED`, `HISTORICAL`, `RESOLVED` | Diagnostic Rule Suite & Synthetic Cases | Rule Consistency Accuracy |
| **Tabular Representation** | Structured NLP Features | 1,012 tabular dimensions (1,000 TF-IDF + 12 concept counts) | Schema invariants & variance checks | Dimensionality & Missingness Rate |

---

## 4. Split Structure & Patient Partitioning Policy

The canonical research dataset ($N=6,098$ documents across 1,000 unique patients and 2,038 clinical encounters) is partitioned strictly at the **patient level**:

| Partition | Patient Count | Encounter Count | Document Count | Proportion | Role in Evaluation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TRAIN** | 700 (`PT-000001` to `PT-000700`) | 1,425 | 4,261 | 69.88% | **Development Only**: Baseline fitting and vocabulary selection. (Read-only during evaluation). |
| **VALIDATION** | 150 (`PT-000701` to `PT-000850`) | 303 | 909 | 14.91% | **Development Benchmark**: Re-evaluated to verify baseline claims and calibrate error taxonomies. |
| **LOCKED TEST** | 150 (`PT-000851` to `PT-001000`) | 310 | 928 | 15.22% | **Final Independent Evaluation**: Evaluated strictly once under uncompromised hold-out protocol. |

### Patient Isolation Guardrail
$$\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Val}) \cap \text{Patients}(\text{Test}) = \emptyset$$
$$\text{Encounters}(\text{Train}) \cap \text{Encounters}(\text{Val}) \cap \text{Encounters}(\text{Test}) = \emptyset$$

---

## 5. Statistical & Confidence Interval Methodology

### 5.1 Patient-Clustered Non-Parametric Bootstrap
Because clinical documents from the same patient share underlying oncological histories, treating documents as independent identically distributed (i.i.d.) observations artificially narrows confidence intervals.
To account for within-patient correlation:
1. Resampling is performed at the **patient cluster level**:
   $$\text{Patients}^* \sim \text{Uniform}(\text{UniquePatients}, N_{\text{patients}})$$
2. All clinical documents associated with sampled patients are retrieved.
3. Target metrics ($\text{Accuracy}$, $\text{Macro F1}$, $\text{Macro Recall}$, $\text{Weighted F1}$, $\text{Critical Recall}$) are calculated on the pooled sample.
4. The process is repeated across $B = 1,000$ iterations with fixed seed `42`.
5. 95% empirical confidence intervals are computed using the $2.5\text{th}$ and $97.5\text{th}$ percentiles:
   $$\text{CI}_{95\%} = [\hat{\theta}^*_{0.025}, \hat{\theta}^*_{0.975}]$$

---

## 6. Leakage Controls & Isolation Rules

1. **Model & Preprocessing Freeze**: No model, scaler, encoder, or vectorizer will be refit on validation or test text.
2. **Locked Test Isolation**: The locked test dataset must never be used to guide threshold selection, vocabulary pruning, or error-correcting heuristics.
3. **Cryptographic Checksums**: Pre- and post-evaluation SHA-256 hashes of all four processed parquet files are checked and verified.
4. **Target Leakage Scan**: Clinical narratives are scanned for 7 forbidden prospective outcome phrases.
5. **Temporal Consistency**: Observation dates must satisfy $\text{document\_date} \le \text{index\_date}$ across all records.

---

## 7. Deterministic Reproducibility Standards

To certify reproducibility:
- Dual-pass batch inference is executed on the test partition.
- Predictions, predicted probability vectors, and extracted features must achieve bit-level equality ($\max|\Delta P| < 10^{-6}$).
- Reproducibility status is logged in machine-readable JSON (`reproducibility_results.json`).

---

## 8. Explicit Non-Claims
This evaluation certifies the engineering integrity and research performance of a prototype NLP baseline. It explicitly does **NOT** claim:
- Clinical diagnostic validation.
- Regulatory clearance or production deployment readiness.
- Autonomous medical triage capability without human oncologist oversight.
