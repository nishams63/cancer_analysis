# Investigation Report: Diagnosis of Frozen NER Metric Across Augmented Training Regimes

**Diagnostic Status:** Completed via empirical runtime instrumentation  
**Timestamp of Investigation:** 2026-09-09 13:16:19 UTC  
**Target Component:** `TrainSpanLexicon` entity extractor (`stage-3-nlp-slm/nlp/benchmarking/src/protocol.py`)  
**Evaluated On:** Frozen validation cohort (`validation.parquet`, 909 documents, 150 patients)  

---

## 1. Prior Hypotheses vs Empirical Measurement Plan

Prior to running this diagnostic probe, two competing hypotheses existed for why the reported NER Exact F1 was identically `0.7186` across all 5 dataset configurations (Configs A&ndash;E):
1. **Hypothesis 1 (Pipeline Caching / Config Pointer Bug):** The evaluation runner was reading a stale cached prediction file or reusing a single frozen model instance without retraining per config.
2. **Hypothesis 2 (Lexicon Vocabulary Invariance by Design):** The entity augmentation rules strictly preserved gold entity text without introducing synthetic entity terms, causing the induced phrase dictionary to be identical and producing invariant exact regex extractions.

To resolve this empirically without assumptions, the diagnostic probe instrumented:
- The exact dataset artifact loaded per configuration.
- The total entity span occurrences parsed from each training split.
- The count and SHA-256 cryptographic hash of unique learned lexicon terms.
- The runtime execution timestamp of prediction generation on the validation set.
- The cryptographic SHA-256 hash of the complete predicted span list for all 909 validation documents.

---

## 2. Empirical Measurements & Instrumentation Results

| Dataset Configuration | Training Rows | Entity Span Mentions in Train | Unique Lexicon Terms | Terms Dict SHA-256 (Prefix) | Prediction Timestamp (UTC) | Predictions SHA-256 (Prefix) | Exact Match F1 | Relaxed Match F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A (Original)** | 4,261 | 15,290 | 1,127 | `50421a2de4b3dae8...` | `2026-09-09T13:15:56.897572+00:00` | `eefa299c8b2ef57d...` | 0.718568 | 0.718568 |
| **Config B (+25% Aug)** | 5,326 | 19,237 | 1,127 | `50421a2de4b3dae8...` | `2026-09-09T13:16:01.340502+00:00` | `eefa299c8b2ef57d...` | 0.718568 | 0.718568 |
| **Config C (+50% Aug)** | 6,391 | 23,160 | 1,127 | `50421a2de4b3dae8...` | `2026-09-09T13:16:05.891101+00:00` | `eefa299c8b2ef57d...` | 0.718568 | 0.718568 |
| **Config D (+100% Aug)** | 8,522 | 31,074 | 1,127 | `50421a2de4b3dae8...` | `2026-09-09T13:16:10.502176+00:00` | `eefa299c8b2ef57d...` | 0.718568 | 0.718568 |
| **Config E (Targeted Balanced)** | 5,761 | 21,248 | 1,127 | `50421a2de4b3dae8...` | `2026-09-09T13:16:15.015894+00:00` | `eefa299c8b2ef57d...` | 0.718568 | 0.718568 |

### Per-Entity Exact Match F1 Breakdown

| Entity Category | Config A | Config B | Config C | Config D | Config E |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GENE_MUTATION** | 0.9486 | 0.9486 | 0.9486 | 0.9486 | 0.9486 |
| **DRUG_NAME** | 0.7390 | 0.7390 | 0.7390 | 0.7390 | 0.7390 |
| **DOSAGE** | 0.4207 | 0.4207 | 0.4207 | 0.4207 | 0.4207 |
| **ADVERSE_EVENT** | 0.7516 | 0.7516 | 0.7516 | 0.7516 | 0.7516 |

---

## 3. Root Cause Analysis & Empirical Findings

### Finding 1: Predictions are genuinely recomputed per config, NOT read from stale disk caches
The timestamp logging proves that each configuration's training dataset was independently opened, parsed, and its lexicon extracted at runtime. Separate inference passes over all 909 validation notes occurred sequentially with unique timestamps.

### Finding 2: The learned lexicon vocabulary is 100% identical across all 5 configurations
As measured, `len(terms)` is **exactly 1127 unique phrases** across all 5 datasets. Furthermore, the cryptographic SHA-256 hash of the sorted `(phrase, label)` term dictionary is identical:
- Lexicon Terms SHA-256: `50421a2de4b3dae808abacd3a4a8343e69d9985fdd0f2354e0346ea0b29c5c6d`

### Finding 3: Why did total entity mentions expand while unique lexicon terms remained fixed?
- In Config A (Original): 4,261 documents yielded **15,290** total entity mentions, mapping to **1,127** unique phrases.
- In Config D (+100% Aug): 8,522 documents yielded **31,074** total entity mentions (+100.0% volume), yet mapped to the **exact same 1,127** unique phrases.

This occurred because the data augmentation pipeline was strictly designed under clinical preservation constraints:
1. **Entity-Preserving Augmentation**: Under constraints 4 and 5 of the design specification (*'DO NOT change drug names, dosages, units, gene mutations, adverse events'*), augmented instances transformed only non-entity carrier text (vitals phrasing, symptom framing, laboratory ordering).
2. **Zero Entity Synthesis**: The augmentation pipeline deliberately did NOT introduce synthetic drug names, novel dosages, or hypothetical mutations that were absent from the training partition.
3. **Non-Statistical Matching Architecture**: `TrainSpanLexicon` is an unweighted exact longest-match regex dictionary. It matches any phrase present in `self.lookup = {t['phrase']: t['label']}`. Unlike a probabilistic sequence tagger (e.g., BiLSTM-CRF, BERT token classifier) whose transition weights shift with token frequency, an unweighted dictionary regex is mathematically identical whether a term appears 1 time or 100 times in the training data.

Consequently, the prediction output hash across all 909 validation documents was identical (`eefa299c8b2ef57d...`), producing the exact same Exact Match F1 (`0.718568`).

---

## 4. Formal Resolution & Technical Qualification

1. **No Pipeline Bug Found**: The evaluation script correctly loads each dataset and recomputes predictions. The invariant F1 is not due to a stale cache or broken pointer.
2. **Architectural Limitation Identified**: `TrainSpanLexicon` is fundamentally a non-trainable, deterministic lookup table on unique training spans. It cannot leverage increased training data frequency to adjust transition probabilities, resolve ambiguous boundaries, or generalize to unseen context.
3. **Required Report Qualification**: The final Stage 3 report must explicitly document:  
   > *'Train Span Lexicon NER Exact F1 (0.7186) remained identical across all 5 configurations because the augmentation pipeline strictly preserved gold entity text without introducing new entity vocabulary, and the unweighted regex dictionary matcher does not update boundary probabilities based on token frequency.'*
4. **Roadmap Recommendation**: To realize gains from data augmentation in NER, Stage 3 must adopt a trainable token-level classifier (such as MiniLM-BIO or PubMedBERT-BIO) whose contextual representations learn boundary features from diverse carrier text.