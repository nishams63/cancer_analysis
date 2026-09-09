# Clinical NLP Engineering Report — Stage 3

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 3 — Clinical NLP Engineering  
**Version**: `1.0.0`  
**Execution Date**: 2026-09-09  
**Handoff Target**: Downstream SLM Engineer (Stage 3) & Multimodal Integration Engineer (Stage 4)  

---

## 1. Executive Summary

This report documents the completion of the Clinical NLP Engineering layer for Stage 3. Built upon the validated, leakage-free dataset produced by Data Engineering and characterized by EDA, the NLP pipeline converts unstructured oncology clinical text into structured, negation-aware clinical features and establishes robust baseline benchmarks.

### Key Engineering Accomplishments
1. **Conservative Clinical Preprocessing**: Developed semantic-preserving text cleaning and NFKC normalization modules that maintain 100% of numerical dosages, lab units, severity grades, and negation terms.
2. **NegEx-Inspired Rule-Based Negation Scoping**: Implemented a clinical negation engine resolving concepts into `AFFIRMED`, `NEGATED`, `HISTORICAL`, and `RESOLVED` states across forward and backward trigger windows.
3. **Negation-Aware Feature Pipeline**: Designed a hybrid feature extractor combining a 1,000-dimensional negation-scoped TF-IDF representation with 12 structured clinical entity and severity indicators, fitted **exclusively** on the training partition.
4. **Strong Baseline Performance**:
   - **Primary Urgency Triage (4-class)**: Achieved a **Macro F1 of 0.7557** and **85.59% accuracy** on the official validation partition, with **94.57% recall on critical safety patients** (`CRITICAL` F1 = 0.9560).
   - **Secondary Toxicity Hazard (8-class)**: Achieved a **Macro F1 of 0.5214** and **82.30% weighted F1**.
   - **Clinical Named Entity Recognition**: Achieved a **Mean Span F1 of 76.70%** against ground-truth spans.
5. **Rigorous Leakage Control & Test Isolation**: Verified 0.0% patient leakage, 0.0% encounter leakage, and certified that the locked test set (`locked_test.parquet`, 928 docs) was **strictly held out** with zero hyperparameter or threshold tuning.
6. **Automated Verification**: Comprehensive test suite consisting of 30 dedicated NLP unit tests passing in 2.77s (55 total passing tests across Stage 3).

---

## 2. Dataset Used

The pipeline operates on the official Stage 3 research dataset:
- **Full Canonical Dataset**: `clinical_nlp_dataset_v1.parquet` (6,098 documents, 18 columns, 1,000 patients, 2,038 encounters).
- **TRAIN Partition**: `train.parquet` (4,261 documents, 700 patients, 1,425 encounters).
- **VALIDATION Partition**: `validation.parquet` (909 documents, 150 patients, 303 encounters).
- **LOCKED_TEST Partition**: `locked_test.parquet` (928 documents, 150 patients, 310 encounters).

---

## 3. EDA Findings Incorporated

Our NLP engineering decisions directly implemented the key recommendations from the EDA report:
1. **Negation Awareness Mandate**: EDA established that 100% of notes contain clinical negations (mean 2.3 cues/doc). We implemented token-level negation prefixing (`neg_fever`) in `feature_extraction.py` to prevent linear models from misinterpreting denied symptoms as acute toxicities.
2. **Class Imbalance Mitigation**: EDA documented a 7.60:1 urgency imbalance and 138.4:1 hazard imbalance. We integrated balanced inverse-class frequency weights into baseline loss functions.
3. **Context Length Sizing**: EDA revealed maximum document lengths of 143 words. Our sequence segmentation and windowing configurations were capped at 256 tokens, guaranteeing zero truncation without VRAM waste.
4. **Patient-Level Isolation**: Preserved the Group K-Split boundaries, guaranteeing that no patient features or vocabulary leaked between splits.

---

## 4. NLP Task Definition

Grounded in the available data fields, the pipeline operationalizes three primary tasks:
1. **Triage Urgency Level Classification (4-class)**: Predicts patient acuity (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) from the clinical intake narrative.
2. **Organ Toxicity Hazard Attribution (8-class)**: Identifies the primary organ system under acute adverse toxicity risk (`NONE`, `HEPATIC`, `PULMONARY`, `HEMATOLOGIC`, `RENAL`, `NEUROPATHIC`, `DERMATOLOGIC`, `CARDIAC`).
3. **Clinical Named Entity Recognition (NER)**: Locates character spans and canonical types for `GENE_MUTATION`, `DRUG_NAME`, `DOSAGE`, and `ADVERSE_EVENT`.

---

## 5. Preprocessing & Normalization Protocol

- **Module**: `src/text_cleaning.py` and `src/text_normalization.py`.
- **Invariants**: Strict preservation of decimals, percentages, dosages, and medical units (`mg/m2`, `mg/dL`, `mmHg`).
- **Standardization**: Canonical casing for driver mutations (`EGFR`, `KRAS`, `TP53`, `ALK`) and oncology acronyms (`NSCLC`, `ECOG`).

---

## 6. Clinical Negation & Context Attribution

- **Module**: `src/negation_detection.py`.
- **Logic**: Rule-based NegEx engine with scope termination.
  - Recognizes pre-negations (`no`, `denies`, `negative for`, `ruled out`, `without`).
  - Recognizes post-negations (`unlikely`, `absent`, `resolved`, `subsided`).
  - Recognizes historical cues (`history of`, `previously treated with`, `prior`).
  - Filters pseudo-negations (`no change`, `not only`).
  - Scopes up to 6 words forward, terminating upon punctuation boundaries (`;`, `:`) or contrasting conjunctions (`but`, `however`, `although`).

---

## 7. Clinical Concept Extraction

- **Module**: `src/clinical_concepts.py`.
- **Taxonomy**:
  - `DRUG_NAME`: 26 targeted antineoplastic agents (Cisplatin, Osimertinib, Pembrolizumab, Paclitaxel, etc.).
  - `GENE_MUTATION`: Driver alterations and specific alleles (EGFR, KRAS, TP53, BRAF, ALK, T790M, G12D).
  - `DOSAGE`: Regimen concentrations and numerical administrations (`\d+(\.\d+)?\s*(mg/m2|mg/dL|mg)`).
  - `ADVERSE_EVENT`: High-risk toxicities and symptoms (neutropenia, dyspnea, fatigue, elevated transaminases, diarrhea, neuropathy, etc.).
- **Output**: Full span dictionaries with exact character start/end offsets and clinical polarity tags.

---

## 8. Feature Extraction Layer

- **Module**: `src/feature_extraction.py`.
- **Components**:
  1. `create_negation_scoped_text`: Modifies vocabulary so that negated terms are distinct tokens (e.g., `neg_fatigue` $\ne$ `fatigue`).
  2. `TfidfVectorizer`: 1,000 sublinear n-gram features (unigrams + bigrams), fitted only on TRAIN.
  3. `StandardScaler`: 12 numeric counts (affirmed concepts, negated concepts, historical concepts, drug count, mutation count, dosage count, Grade 3/4 indicator, critical symptom indicator).
- **Data Products**: Saved as Parquet feature matrices (`nlp_features_train.parquet`, `nlp_features_val.parquet`).

---

## 9. Baseline Models & Evaluation Methodology

- **Module**: `src/baseline.py`.
- **Urgency Model**: `LogisticRegression(class_weight='balanced', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)`.
- **Hazard Model**: `LogisticRegression(class_weight='balanced', C=0.5, solver='lbfgs', max_iter=1000, random_state=42)`.
- **Partitioning**: Strictly trained on `TRAIN` ($N = 4,261$) and validated on `VALIDATION` ($N = 909$).

---

## 10. Empirical Validation Results

### Primary Urgency Level Classification (Validation Set, N = 909)
- **Macro F1 Score**: **0.7557**
- **Weighted F1 Score**: **0.8605**
- **Overall Accuracy**: **85.59%**
- **Per-Class Breakdown**:
  - `CRITICAL`: Precision = 0.9667, Recall = **0.9457**, F1 = **0.9560** (Support = 92)
  - `HIGH`: Precision = 0.6408, Recall = **0.7109**, F1 = **0.6741** (Support = 128)
  - `LOW`: Precision = 0.9654, Recall = **0.9316**, F1 = **0.9482** (Support = 599)
  - `MEDIUM`: Precision = 0.4242, Recall = **0.4667**, F1 = **0.4444** (Support = 90)

### Secondary Toxicity Hazard Classification (Validation Set, N = 909)
- **Macro F1 Score**: **0.5214**
- **Weighted F1 Score**: **0.8230**
- **Overall Accuracy**: **77.56%**
- **Key System F1 Scores**: `PULMONARY` (0.9541), `NONE` (0.8675), `HEPATIC` (0.7665), `CARDIAC` (0.5333).

### Entity Span Extraction (Validation Set, N = 909)
- **Mean Span Precision**: **75.84%**
- **Mean Span Recall**: **78.31%**
- **Mean Span F1 Score**: **76.70%**

---

## 11. Error Analysis Summary

Systematic analysis identified three primary error modes:
1. **Mild vs. Moderate Boundary Confusion**: Documents presenting with Grade 1 vs Grade 2 fatigue share substantial lexical tokens. Linear models lack the self-attention depth required to distinguish subtle clinical modifiers.
2. **Minority Hazard Over-Guessing**: Rigid inverse-frequency weights caused false positives for rare classes (`DERMATOLOGIC`, support = 4) when non-toxic cosmetic mentions appeared.
3. **Complex Clause Boundaries**: Nested clauses without punctuation boundaries occasionally exceeded the 6-word forward negation scope.

---

## 12. Leakage Audit Summary

- **Patient Overlap**: **0.0%** (0 overlapping patients between Train, Val, and Locked Test).
- **Encounter Overlap**: **0.0%** (0 overlapping encounters).
- **Preprocessing Invariance**: Vectorizers and scalers were fitted strictly on `TRAIN`.
- **Temporal Check**: $\text{document\_date} \le \text{index\_date}$ across 100% of records.
- **Forbidden Phrases**: 0 prospective outcome terms detected.
- **Locked Test Set**: Strictly isolated; zero parameter tuning conducted.

---

## 13. Limitations

1. **Rule-Based Extraction Boundaries**: Entity span extraction uses curated regular expressions. While achieving 76.70% Span F1, multi-token entity boundaries can vary compared to deep neural sequence taggers.
2. **Class Imbalance in Rare Hazards**: Classes with support $< 15$ (`CARDIAC`, `DERMATOLOGIC`, `NEUROPATHIC`) suffer precision degradation under linear balanced loss functions.
3. **Research Prototype Boundary**: This system is designed for engineering experimentation and decision-support research. It is **NOT clinically validated** and must not be used for autonomous medical decision-making.

---

## 14. Reproducibility

- **Random Seeds**: Enforced at `seed = 42` across all modules.
- **Configuration**: All parameters and paths centralized in `configs/nlp_config.yaml`.
- **Self-Contained Artifacts**: Pre-trained baselines, tokenizers, scalers, and encoders saved to `artifacts/`.
- **Automated Testing**: Fully reproducible via `pytest stage-3-nlp-slm/nlp/tests/ -v`.

---

## 15. Handoff to SLM Engineer

The SLM Engineer can immediately proceed with fine-tuning 1B–3B Small Language Models using the deliverables provided in this module:
- **Baseline Benchmark**: SLM models must exceed **Macro F1 $\ge 0.7557$** on Urgency and **F1 $\ge 0.9560$** on Critical safety triage.
- **Prompt Format**: Incorporate `document_type` as an instruction prefix:
  `[INST] Document Type: {document_type}\nClinical Note: {text}\nGenerate 2-sentence bedside clinical handoff briefing. [/INST]`
- **Target Summary**: The ground-truth `slm_summary` target is strictly 2 sentences (mean $34.2 \pm 4.1$ words). Bound generation to `max_new_tokens = 64`.
- **Feature Artifacts**: The extracted feature datasets (`nlp_features_train.parquet`, `nlp_features_val.parquet`) and validation predictions (`validation_predictions.csv`) are available for direct comparison.

---

## 16. Recommended Next Steps for Stage 3

1. **SLM Fine-Tuning**: Initialize 4-bit QLoRA fine-tuning on `Llama-3.2-1B-Instruct` or `Qwen2.5-1.5B-Instruct` targeting bedside briefing summarization and triage instruction following.
2. **Comparative Evaluation**: Benchmark the SLM against our linear baseline on the validation set using ROUGE, BERTScore, and Urgency classification accuracy.
3. **Multimodal Late Fusion**: Pass the tabular NLP features (`nlp_features_*.parquet`) to the Integration Engineer to combine with Stage 1 ML toxicity predictions.
