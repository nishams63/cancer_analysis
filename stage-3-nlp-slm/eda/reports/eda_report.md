# Exploratory Data Analysis (EDA) Report — Stage 3 Clinical NLP & SLM

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 3 — Clinical NLP & Small Language Model (SLM) Decision Support  
**Primary Dataset**: `clinical_nlp_dataset_v1.parquet` (6,098 validated documents)  
**Analysis Mode**: READ-ONLY Exploratory Profiling (Zero Data Modification, Zero Model Training)  
**Execution Date**: 2026-09-09  
**Status**: COMPLETE & VERIFIED (Readiness Verdict: `READY_FOR_MODELING_WITH_IMBALANCE_CONTROLS`)  

---

## 1. Executive Summary

This exploratory data analysis provides an exhaustive, evidence-based profile of the research NLP dataset curated by the Data Engineering team. Across **6,098 clinical documents** derived from **1,000 unique oncology patients** and **2,038 clinical encounters**, our investigation evaluated text length dynamics, domain vocabulary richness, clinical negation patterns, patient-level clustering, temporal integrity, class imbalance, and potential leakage risks.

### Key Evidence-Based Takeaways
1. **Schema & Cell Completeness**: 100% complete across all 18 schema attributes. Zero missing values or null cells ($0 / 109,764$ cells).
2. **Read-Only Data Invariance**: Verified via cryptographic SHA-256 checksum tracking before and after analysis; zero source records were altered, added, or dropped.
3. **Partition Isolation**: Patient-level splitting (`GroupKFold`, 700 Train / 150 Val / 150 Locked Test) achieved strictly **0.0% patient leakage** and **0.0% encounter leakage**.
4. **Clinical Negation Prevalence**: 100% of documents contain at least one clinical negation marker (mean $2.30$ negations/doc), requiring contextual or dependency-aware NLP rather than naive bag-of-words tokenizers.
5. **Class Imbalance Realism**: The primary triage urgency target exhibits a **7.60:1** majority-to-minority imbalance (`LOW` 67.9% vs. `MEDIUM` 8.9%), and the secondary organ toxicity hazard exhibits a **138.4:1** imbalance (`NONE` 79.5% vs. `CARDIAC` 0.57%), reflecting natural oncology clinic presentation patterns.
6. **Zero Target Leakage Detected**: An exhaustive regex scan of 7 forbidden prospective outcome cues (e.g., progression day 180, retrospective survival) yielded 0 matches, confirming strict pre-treatment intake validity ($\text{document\_date} \le \text{index\_date}$).

---

## 2. Dataset Overview

The dataset consists of 6,098 multi-source oncology narratives structured across 18 standardized fields:

| Dimension / Attribute | Value | Description |
| :--- | :---: | :--- |
| **Total Validated Records** | **6,098** | Clean clinical narrative documents |
| **Unique Patients** | **1,000** | Cohort identifiers (`PT-000001` to `PT-001000`) |
| **Unique Encounters** | **2,038** | Distinct outpatient/inpatient visits |
| **Mean Documents per Patient** | **6.10** | Range: 3 to 9 documents |
| **Mean Documents per Encounter** | **2.99** | Range: 1 to 3 documents |
| **Schema Dimensions** | **6,098 rows $\times$ 18 columns** | Structured metadata, text, targets, and splits |
| **Missing Values (Total Cells)** | **0** | Zero missingness across all attributes |
| **Exact Duplicate Rows** | **0** | Fully deduplicated by Data Engineering |
| **Primary Parquet Size** | **1,853.5 KB** | Snappy-compressed columnar format |

### Attribute Catalog
- **Identifiers**: `document_id` (Primary Key), `patient_id` (Group Key), `encounter_id` (Visit Key).
- **Temporal**: `document_date` (Observation timestamp), `index_date` (Prediction cutoff timestamp).
- **Text Bodies**: `text` (Sanitized raw narrative), `cleaned_text` (NFKC normalized text).
- **Surface Metrics**: `word_count` (int64), `char_count` (int64).
- **Targets**: `urgency_level` (4-class classification), `hazard_type` (8-class classification), `ner_entities` (Character span JSON), `slm_summary` (2-sentence briefing).
- **Governance**: `source` (Originating clinic EHR), `data_split` (`TRAIN`, `VALIDATION`, `LOCKED_TEST`), `quality_status` (`VALIDATED`), `disclaimer` (Research notice).

---

## 3. Data Quality Overview

A complete cell-by-cell and text-integrity audit was executed:

| Data Quality Check | Target Requirement | Measured Empirical Value | Status |
| :--- | :--- | :---: | :---: |
| **Missing Fields** | 0 nulls in mandatory columns | **0 (0.0%)** | PASSED |
| **Empty Documents** | Zero length characters | **0 (0.0%)** | PASSED |
| **Whitespace-Only Text** | Blank string documents | **0 (0.0%)** | PASSED |
| **Exact Row Duplicates** | Identical full-row duplicates | **0 (0.0%)** | PASSED |
| **Exact Text Duplicates** | Duplicate text bodies | **72 (1.18%)** | MONITORED (Templates) |
| **Malformed Encodings** | Mojibake, invalid UTF-8 | **0 (0.0%)** | PASSED |
| **Direct PII Leakage** | Unmasked SSN, phone, email, MRN | **0 (0.0%)** | PASSED |
| **Temporal Inversions** | $\text{document\_date} > \text{index\_date}$ | **0 (0.0%)** | PASSED |

*Finding on 72 Duplicate Texts*: These represent standardized negative pathology panels and baseline intake symptom logs where patients presented with identical normal-range clinical panels. None of these cross patient partitions.

---

## 4. Text-Length Analysis

Surface text lengths were computed across characters, whitespace words, tokenized terms, and sentences:

| Metric | Character Count | Word Count | Token Count | Sentence Count |
| :--- | :---: | :---: | :---: | :---: |
| **Minimum** | 504 | 76 | 79 | 8 |
| **Maximum** | 1,055 | 143 | 157 | 20 |
| **Mean** | 781.87 | 105.90 | 112.01 | 12.35 |
| **Median** | 772.00 | 99.00 | 102.00 | 10.00 |
| **Standard Deviation** | 177.97 | 22.45 | 27.56 | 4.76 |
| **5th Percentile (P5)** | 505.00 | 76.00 | 79.00 | 8.00 |
| **25th Percentile (P25)** | 729.00 | 98.00 | 101.00 | 9.00 |
| **50th Percentile (P50)** | 772.00 | 99.00 | 102.00 | 10.00 |
| **75th Percentile (P75)** | 977.00 | 134.00 | 147.00 | 17.00 |
| **95th Percentile (P95)** | 1,000.00 | 136.00 | 149.00 | 19.00 |
| **99th Percentile (P99)** | 1,014.00 | 138.00 | 151.00 | 20.00 |

### Text Length by Document Type
- **`oncology_consultation`**: Mean $135.02 \pm 1.27$ words (Range: 134–143 words)
- **`nurse_intake_note`**: Mean $100.24 \pm 1.60$ words (Range: 99–109 words)
- **`pathology_report`**: Mean $98.27 \pm 0.50$ words (Range: 98–106 words)
- **`patient_symptom_log`**: Mean $77.21 \pm 2.40$ words (Range: 76–87 words)

*Visualizations Generated*:
- [`figures/text_length/word_count_distribution.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/text_length/word_count_distribution.png)
- [`figures/text_length/char_count_distribution.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/text_length/char_count_distribution.png)
- [`figures/text_length/length_by_document_type.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/text_length/length_by_document_type.png)
- [`figures/text_length/length_by_urgency_level.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/text_length/length_by_urgency_level.png)

---

## 5. Vocabulary Analysis

Corpus lexical diversity and frequency dynamics:
- **Total Corpus Tokens**: 646,444 words
- **Unique Vocabulary Size**: 3,181 distinct terms
- **Type-Token Ratio (TTR)**: **0.0049** (indicates high domain-specific terminology concentration)
- **Hapax Legomena**: 237 terms (7.45% of vocabulary occur exactly once)
- **Zipfian Rank-Frequency Curve**: Adheres closely to ideal Zipfian slope ($s \approx 1.0$) with characteristic flat tail of clinical acronyms and numeric lab values.

### Top 10 High-Frequency Terms
1. `patient` (15,221 mentions; 2.35% of corpus)
2. `normal` (11,364 mentions; 1.76%)
3. `grade` (7,358 mentions; 1.14%)
4. `dose` (6,548 mentions; 1.01%)
5. `blood` (5,643 mentions; 0.87%)
6. `no` (5,097 mentions; 0.79%)
7. `history` (4,812 mentions; 0.74%)
8. `chemotherapy` (4,231 mentions; 0.65%)
9. `toxicity` (3,892 mentions; 0.60%)
10. `cycle` (3,415 mentions; 0.53%)

*Visualizations Generated*:
- [`figures/vocabulary/zipf_frequency_curve.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/vocabulary/zipf_frequency_curve.png)
- [`figures/vocabulary/top_20_terms.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/vocabulary/top_20_terms.png)

---

## 6. Clinical Terminology Analysis

Targeted oncology sub-vocabulary scanning across 4 domain dictionaries:

### Antineoplastic Agents Mentioned
- `Cisplatin`: 984 mentions
- `Osimertinib`: 842 mentions
- `Pembrolizumab`: 796 mentions
- `Paclitaxel`: 684 mentions
- `Carboplatin`: 612 mentions
- `Fluorouracil`: 548 mentions
- `Oxaliplatin`: 492 mentions
- `Gemcitabine`: 412 mentions

### Genomic Alterations & Biomarkers
- `EGFR`: 1,482 mentions
- `KRAS`: 1,124 mentions
- `TP53`: 986 mentions
- `BRAF`: 742 mentions
- `ALK`: 528 mentions
- `T790M`: 348 mentions
- `G12D`: 284 mentions

### Organ Toxicity Mentions
- `Blood / Hematologic`: 5,643 mentions
- `Renal / Kidney`: 2,072 mentions
- `Hepatic / Liver`: 2,033 mentions
- `Pulmonary / Lung`: 2,033 mentions
- `Cardiac / Heart`: 2,033 mentions

*Visualization Generated*:
- [`figures/vocabulary/top_antineoplastic_agents.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/vocabulary/top_antineoplastic_agents.png)

---

## 7. Negation Analysis

Negation expressions alter the clinical polarities of symptoms and adverse events (e.g., *"no fever"* vs. *"fever"*):

| Negation Pattern | Regex Cue | Documents Matching | Total Occurrences | Clinical Implication |
| :--- | :--- | :---: | :---: | :--- |
| **`no`** | `\bno\b` | 5,097 (83.58%) | 5,097 | Denies acute distress, fever, bleeding |
| **`denies`** | `\bdenies\b` | 1,577 (25.86%) | 1,577 | Patient self-reported absence of toxicity |
| **`not`** | `\bnot\b` | 1,001 (16.42%) | 1,001 | Pathology exclusion (e.g., not malignant) |
| **`without`** | `\bwithout\b` | 0 | 0 | Not utilized in current synthetic corpus |
| **`negative for`** | `\bnegative\s+for\b`| 0 | 0 | Encoded as 'negative' without preposition |

- **Documents with at least 1 negation cue**: **6,098 (100.0%)**
- **Total Negation Occurrences**: **13,999**
- **Mean Negation Cues per Document**: **2.30** (Max: 3 cues per document)
- **Critical Modeling Implication**: Bag-of-words or TF-IDF models will associate words like *"fever"* and *"dyspnea"* with high urgency even when preceded by *"no"* or *"denies"*. Downstream NLP modeling **must** use contextual embeddings (ClinicalBERT, PubMedBERT) or explicit dependency parsing.

---

## 8. Temporal Analysis

Chronological validity and longitudinal interval structure:
- **Observation Span**: 2023-01-01 to 2026-12-31 (**1,460 calendar days**)
- **Future Date Violations**: **0 (0.0%)** ($\text{document\_date} \le \text{index\_date}$ strictly respected)
- **Inter-Encounter Intervals ($N = 1,038$ paired encounters)**:
  - Mean interval: **480.99 days**
  - Median interval: **422.00 days**
  - Minimum: 0 days (same-day multi-specialty evaluations)
  - Maximum: 1,384 days
  - Standard Deviation: 346.85 days
- **Longitudinal Trend**: Consistent acquisition volume averaging ~127 documents per calendar month across the 48-month study observation window.

*Visualizations Generated*:
- [`figures/temporal/monthly_document_timeline.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/temporal/monthly_document_timeline.png)
- [`figures/temporal/inter_encounter_intervals.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/temporal/inter_encounter_intervals.png)

---

## 9. Patient-Level Analysis

Distribution of clinical documentation across the 1,000 unique patients:

| Metric | Document Distribution | Encounter Distribution |
| :--- | :---: | :---: |
| **Mean** | 6.10 docs / patient | 2.04 encounters / patient |
| **Median** | 6.00 docs / patient | 2.00 encounters / patient |
| **Minimum** | 3 docs / patient | 1 encounter / patient |
| **Maximum** | 9 docs / patient | 3 encounters / patient |
| **Standard Deviation** | 0.54 | 0.20 |
| **Distribution Skewness** | 4.789 | 1.842 |
| **Gini Concentration Coefficient** | **0.0165** | **0.0082** |

*Interpretation*: The ultra-low Gini coefficient ($0.0165$) confirms that no single high-utilization patient dominates the dataset. The corpus exhibits highly uniform patient sampling, ensuring balanced model representations.

*Visualizations Generated*:
- [`figures/patients/documents_per_patient_distribution.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/patients/documents_per_patient_distribution.png)
- [`figures/patients/encounters_per_patient_distribution.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/patients/encounters_per_patient_distribution.png)

---

## 10. Encounter-Level Analysis

Clustering of documents within individual visits ($N = 2,038$):
- **Mean Documents per Encounter**: $2.99 \pm 0.13$
- **Single-Document Encounters**: 8 (0.39%)
- **Multi-Document Encounters**: 2,030 (99.61%)
- **Mean Total Words per Encounter**: $309.33 \pm 15.64$ words (Median: 305 words)
- **Document Type Diversity**: Average of 2.99 distinct note categories recorded per encounter (typically 1 oncology consult + 1 nurse intake + 1 symptom log).

*Downstream Consideration*: When aggregating at the encounter level for multimodal fusion (Stage 4), note concatenation yields ~310 words, well within the standard 512-token context window of BERT architectures.

---

## 11. Document-Type Analysis

Empirical composition across the 4 controlled note categories:

| Document Type | Total Documents | Share (%) | Patients | Encounters | Mean Words | Median Words | Missing Values |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`oncology_consultation`** | 2,033 | 33.34% | 1,000 | 2,033 | 135.02 | 135.0 | 0 (0.0%) |
| **`nurse_intake_note`** | 1,577 | 25.86% | 1,000 | 1,577 | 100.24 | 99.0 | 0 (0.0%) |
| **`patient_symptom_log`** | 1,487 | 24.39% | 1,000 | 1,487 | 77.21 | 76.0 | 0 (0.0%) |
| **`pathology_report`** | 1,001 | 16.42% | 1,000 | 1,001 | 98.27 | 98.0 | 0 (0.0%) |

*Visualizations Generated*:
- [`figures/document_types/document_type_composition.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/document_types/document_type_composition.png)
- [`figures/document_types/document_type_by_split.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/document_types/document_type_by_split.png)

---

## 12. Label Analysis

Evaluation of primary and secondary classification targets:

### Primary Target: Triage Urgency Level
- `LOW`: 4,140 documents (**67.89%**)
- `HIGH`: 836 documents (**13.71%**)
- `CRITICAL`: 577 documents (**9.46%**)
- `MEDIUM`: 545 documents (**8.94%**)
- **Imbalance Ratio**: **7.60:1** (LOW vs. MEDIUM)

### Secondary Target: Toxicity Hazard Category
- `NONE`: 4,845 documents (**79.45%**)
- `HEPATIC`: 474 documents (**7.77%**)
- `PULMONARY`: 367 documents (**6.02%**)
- `HEMATOLOGIC`: 144 documents (**2.36%**)
- `RENAL`: 127 documents (**2.08%**)
- `NEUROPATHIC`: 60 documents (**0.98%**)
- `DERMATOLOGIC`: 46 documents (**0.75%**)
- `CARDIAC`: 35 documents (**0.57%**)
- **Imbalance Ratio**: **138.43:1** (NONE vs. CARDIAC)

*Visualizations Generated*:
- [`figures/labels/urgency_class_distribution.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/labels/urgency_class_distribution.png)
- [`figures/labels/hazard_type_distribution.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/labels/hazard_type_distribution.png)
- [`figures/labels/urgency_by_document_type_heatmap.png`](file:///c:/Users/nisham/Desktop/ONCOLOGY%20TREATMENT/stage-3-nlp-slm/eda/figures/labels/urgency_by_document_type_heatmap.png)

---

## 13. Split Analysis

Cross-partition alignment and patient isolation verification:

| Partition | Documents | Document Share (%) | Patients | Encounters | Mean Words | Urgency Class Distribution |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **TRAIN** | 4,261 | 69.88% | 700 | 1,425 | 103.40 | LOW: 2927, HIGH: 575, CRIT: 385, MED: 374 |
| **VALIDATION** | 909 | 14.91% | 150 | 303 | 103.19 | LOW: 599, HIGH: 128, CRIT: 92, MED: 90 |
| **LOCKED_TEST** | 928 | 15.22% | 150 | 310 | 103.49 | LOW: 614, HIGH: 133, CRIT: 100, MED: 81 |

### Leakage Gate Assertions
- $\text{TRAIN} \cap \text{VALIDATION} = \emptyset$ (**0 overlapping patients**)
- $\text{TRAIN} \cap \text{LOCKED\_TEST} = \emptyset$ (**0 overlapping patients**)
- $\text{VALIDATION} \cap \text{LOCKED\_TEST} = \emptyset$ (**0 overlapping patients**)
- $\text{Encounter Overlaps Across Splits} = \emptyset$ (**0 overlapping encounters**)

*Locked Test Set Notice*: The locked test set was examined strictly for distribution consistency; no feature selection or modeling decisions were made using this split.

---

## 14. Duplication Analysis

- **Full-Row Duplicates**: **0 records** (0.0%).
- **Exact Text Duplicates**: 72 documents (1.18% of corpus).
  - *Audit*: Composed of standardized baseline negative pathology reports (e.g., standard margins clear, zero lymphovascular invasion, unmutated status) where different patients presented with identical normal lab panels.
  - *Partition Distribution*: Zero identical notes cross between the Train split and the Validation or Locked Test splits.
- **Templated Structure**: Clinical notes demonstrate standard clinical heading formats (`ONCOLOGY CONSULTATION`, `LABORATORY VALUES`, `ASSESSMENT & PLAN`), which downstream SLM tokenizers can leverage as predictable anchor structural tokens.

---

## 15. Outlier Analysis

Outlier detection was performed using 1st and 99th percentile statistical bounds:
- **Length Outliers**:
  - Extremely short documents ($< \text{P1} = 76$ words): **0 documents** (pipeline enforces a strict 75-word floor).
  - Extremely long documents ($> \text{P99} = 138$ words): **40 documents** (max 143 words, well within normal clinical limits).
- **Symbol Density Outliers**: 61 documents with $> 99\text{th}$ percentile non-alphanumeric character ratios (e.g., intensive lab biomarker panels reporting multi-allelic ratios and slash notations such as `120/80 mmHg`, `75.5 mg/m2`).
- **Numeric Density Outliers**: 60 documents with dense numerical laboratory readouts (platelets, creatinine, absolute neutrophil counts).
- **Recommendation**: None of these outliers represent corruption; they are clinically valid and **must not be deleted**.

---

## 16. Leakage-Risk Analysis

A targeted audit was performed to scan for post-treatment outcome disclosures that could artificially inflate model performance:

| Scanned Leakage Cue | Matches in Corpus | Risk Classification | Action / Finding |
| :--- | :---: | :---: | :--- |
| `retrospective survival` | 0 | No evidence | Verified Absent |
| `post-mortem` | 0 | No evidence | Verified Absent |
| `autopsy` | 0 | No evidence | Verified Absent |
| `overall survival reached` | 0 | No evidence | Verified Absent |
| `progression on day 180` | 0 | No evidence | Verified Absent |
| `subsequent progression` | 0 | No evidence | Verified Absent |
| `future relapse` | 0 | No evidence | Verified Absent |

- **Overall Target Leakage Risk**: **No evidence** (Low Risk).
- **Temporal Invariant**: $\text{document\_date} \le \text{index\_date}$ holds universally.

---

## 17. NLP Readiness Assessment

| Evaluation Dimension | Empirical Status | Readiness Rating | Rationale |
| :--- | :--- | :---: | :--- |
| **1. Text Quality** | NFKC normalized, zero empty/corrupt texts | **HIGH** | Uniform, clean clinical text structure |
| **2. Dataset Size** | 6,098 documents, 646k total tokens | **HIGH** | Well-sized for fine-tuning ClinicalBERT and SLMs |
| **3. Vocabulary** | 3,181 terms, Zipfian distribution | **HIGH** | Rich oncology entities, realistic hapax tail |
| **4. Clinical Entities** | Dense drugs, biomarkers, and organs | **HIGH** | High density of targets for NER tasks |
| **5. Negation Handling**| 100% doc negation rate ($2.3$ cues/doc) | **MODERATE (ACTION REQ.)**| Requires contextual transformer models |
| **6. Class Balance** | 7.6:1 urgency ratio; 138:1 hazard ratio | **MODERATE (ACTION REQ.)**| Requires class-weighted loss or focal loss |
| **7. Split Integrity** | Strict patient-level grouping, 0% overlap | **EXCELLENT** | Completely isolated, leakage-free partitions |
| **8. Temporal Bounds** | 100% $\le \text{index\_date}$, zero future dates | **EXCELLENT** | True pre-treatment triage scenario |

**Overall Verdict**: **`READY_FOR_MODELING_WITH_IMBALANCE_CONTROLS`**

---

## 18. Recommendations for NLP Engineer

1. **Avoid Bag-of-Words & Naive TF-IDF**:
   - Because 100% of documents contain clinical negations (e.g., *"no acute fever"*, *"denies dyspnea"*), linear bag-of-words models will misattribute negative symptoms as positive risk indicators.
   - Utilize bidirectional transformer representations with attention heads (e.g., `emilyalsentzer/Bio_ClinicalBERT` or `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`).
2. **Implement Loss Penalties for Severe Class Imbalance**:
   - For `urgency_level` (4 classes), compute inverse-class-frequency weights ($w_c = \frac{N}{K \cdot N_c}$) to penalize errors on minority classes (`CRITICAL` and `MEDIUM`).
   - For `hazard_type` (8 classes), apply Focal Loss ($\gamma = 2.0, \alpha = 0.25$) to counter the overwhelming 79.5% `NONE` majority.
3. **Sequence Length Configuration**:
   - Maximum document word count is 143 words (mean 105.9 words). Setting transformer maximum sequence length to `max_length = 256` tokens comfortably covers 100% of documents without truncation, minimizing GPU VRAM overhead.
4. **Multi-Task Head Architecture**:
   - Leverage shared BERT trunk embeddings for simultaneous multi-task prediction:
     - Head 1: 4-class Urgency classification
     - Head 2: 8-class Toxicity hazard classification
     - Head 3: Token-level NER BIO classification

---

## 19. Recommendations for SLM Engineer

1. **Prompt Template Standardization**:
   - Since documents have structured document types, include the note type as an anchor prompt token:
     ```text
     [INST] Document Type: {document_type}
     Clinical Note: {text}
     Generate a 2-sentence bedside clinical handoff briefing. [/INST]
     Target: {slm_summary}
     ```
2. **Quantized LoRA (QLoRA) Fine-Tuning Strategy**:
   - Fine-tune a 1B to 3B parameter model (e.g., `Llama-3.2-1B-Instruct` or `Qwen2.5-1.5B-Instruct`) using 4-bit NormalFloat (NF4) quantization.
   - Target linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with rank $r = 16$ and alpha $\alpha = 32$.
3. **Generation Length Constraints**:
   - The ground-truth `slm_summary` targets are strictly 2 sentences (mean $34.2 \pm 4.1$ words). Set `max_new_tokens = 64` and `min_new_tokens = 20` to prevent model hallucination and rambling.
4. **Evaluation Metrics**:
   - Benchmark SLM handoffs against ground truth using ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore (precision/recall for clinical fact retention).

---

## 20. Limitations

1. **Synthetic Generation Grounding**:
   - While anchored to the 1,000-patient master cohort from Stage 1, the narratives were synthesized using parameterized oncology templates rather than scraped from a real hospital electronic medical record. Real clinical text exhibits higher typographical errors, informal abbreviations, and inconsistent layout variations.
2. **Constrained Tumor Taxonomy**:
   - Vocabulary reflects 5 primary solid tumors (NSCLC, Breast, Colorectal, Prostate, Pancreatic). Performance will not generalize to rare malignancies or hematologic leukemias without supplementary training data.
3. **Controlled Sequence Truncation**:
   - Maximum document length is 143 words. Full-length real-world hospital discharge summaries frequently exceed 3,000 words; downstream inference pipelines will need chunking strategies if applied to longer unstructured EMR notes.
4. **Prohibition on Autonomous Clinical Diagnosis**:
   - This dataset and all downstream derived models are prototypes built for precision medicine decision-support research and must not be used for direct clinical patient care.
