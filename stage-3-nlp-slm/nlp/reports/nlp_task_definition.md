# Clinical NLP Task Definition Report — Stage 3

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 3 — Clinical NLP Engineering  
**Version**: `1.0.0`  
**Status**: APPROVED & DATA-DRIVEN  

---

## 1. NLP Problem Statement
In precision oncology, clinical documentation (consultations, nurse intake notes, symptom logs, and pathology reports) contains rich, unstructured descriptions of patient symptoms, antineoplastic drug administration, genomic driver mutations, and therapy-induced toxicities. 

The primary problem is to convert these unstructured clinical narratives into structured, calibrated clinical information—specifically:
1. Identifying triage urgency levels to prioritize deteriorating patients before severe treatment complications occur.
2. Classifying organ/system toxicity hazard types.
3. Extracting clinical named entities (`GENE_MUTATION`, `DRUG_NAME`, `DOSAGE`, `ADVERSE_EVENT`) with exact span boundaries.
4. Attributing clinical polarity (Affirmed, Negated, Historical, Resolved) to ensure negated symptoms (e.g., *"denies fever"*) are not conflated with active adverse reactions.

---

## 2. Input Text
- **Field**: `text` (Sanitized clinical narrative) and `cleaned_text` (NFKC normalized text).
- **Source**: Four controlled note categories across 1,000 unique patients (`PT-000001` through `PT-001000`) and 2,038 clinical encounters.
- **Length Characteristics**: Mean word count $105.90 \pm 22.45$ words (Range: 76 to 143 words). Mean character count $781.87 \pm 177.97$ characters.
- **Privacy Profile**: 100% sanitized of direct identifiers (`[NAME]`, `[PHONE]`, `[EMAIL]`, `[MRN]`).

---

## 3. Available Labels & Annotations in Ground Truth
Grounded strictly in the validated Data Engineering dataset (`clinical_nlp_dataset_v1.parquet`):

| Target Field | Task Modality | Vocabulary / Domain | Distribution in Full Dataset (N = 6,098) |
| :--- | :--- | :--- | :--- |
| **`urgency_level`** | Multi-class Classification (Primary) | 4 classes: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` | `LOW`: 4,140 (67.9%), `HIGH`: 836 (13.7%), `CRITICAL`: 577 (9.5%), `MEDIUM`: 545 (8.9%) |
| **`hazard_type`** | Multi-class Classification (Secondary) | 8 classes: `NONE`, `HEPATIC`, `PULMONARY`, `HEMATOLOGIC`, `RENAL`, `NEUROPATHIC`, `DERMATOLOGIC`, `CARDIAC` | `NONE`: 4,845 (79.5%), `HEPATIC`: 474 (7.8%), `PULMONARY`: 367 (6.0%), ..., `CARDIAC`: 35 (0.57%) |
| **`ner_entities`** | Token / Span Extraction | 4 classes: `GENE_MUTATION`, `DRUG_NAME`, `DOSAGE`, `ADVERSE_EVENT` | JSON lists of character spans `[{"start": int, "end": int, "label": str, "text": str}]` |
| **`slm_summary`** | Abstractive Summarization (SLM Handoff) | 2-sentence bedside clinical briefing | Mean word count $34.2 \pm 4.1$ words |

---

## 4. Supported NLP Tasks
1. **Primary Task: Triage Urgency Level Classification**
   - 4-class cost-sensitive classification mapping clinical text to patient risk tier.
2. **Secondary Task: Toxicity Hazard Classification**
   - 8-class organ/system toxicity attribution.
3. **Extraction Task: Clinical Named Entity Recognition (NER)**
   - Rule-based extraction and span localization of medications, genomic mutations, dosages, and adverse events.
4. **Contextual Attribution Task: Clinical Negation & Temporality Detection**
   - Distinguishing active acute symptoms from negated or historical mentions.
5. **Feature Generation Task**:
   - Structured tabular feature matrix generation for integration with Stage 1 tabular ML models.

---

## 5. Output Representations
- **Document-Level Structured Record**:
  ```json
  {
    "document_id": "DOC-000124",
    "triage_urgency": {
      "predicted_class": "HIGH",
      "confidence": 0.8421,
      "class_probabilities": {"LOW": 0.05, "MEDIUM": 0.08, "HIGH": 0.84, "CRITICAL": 0.03}
    },
    "toxicity_hazard": {
      "predicted_class": "HEPATIC",
      "confidence": 0.7912
    },
    "clinical_entities": [
      {"start": 142, "end": 151, "label": "DRUG_NAME", "text": "Cisplatin", "polarity": "AFFIRMED"},
      {"start": 185, "end": 191, "label": "ADVERSE_EVENT", "text": "nausea", "polarity": "NEGATED"}
    ]
  }
  ```
- **Tabular Feature Matrix**:
  - Saved as Parquet (`nlp_features_train.parquet`, `nlp_features_val.parquet`) containing document length, concept counts, negation ratios, high-risk flags, and TF-IDF sparse dimensions.

---

## 6. Clinical Concepts Supported by the Data
- **Antineoplastic Regimens**: Cisplatin, Carboplatin, Osimertinib, Pembrolizumab, Paclitaxel, Fluorouracil, Oxaliplatin, Gemcitabine, Docetaxel, Irinotecan, Trastuzumab, Tamoxifen.
- **Biomarkers & Alterations**: EGFR, KRAS, TP53, BRAF, ALK, T790M, G12D, G12C, V600E.
- **Toxicity & Symptoms**: neutropenia, dyspnea, fatigue, elevated transaminases, diarrhea, neuropathy, paresthesia, rash, acute adverse toxicities, manageable mild fatigue.
- **Severity Grades**: Grade 1, Grade 2, Grade 3, Grade 4.
- **Dosage Units**: `mg`, `mg/m2`, `mg/dL`, `mmHg`, `g/dL`, `%`.

---

## 7. Label Provenance & Limitations
- **Provenance**: Derived from the Stage 1 master patient dataset cohort, parameterized and synthesized by the Stage 3 Data Engineer to mirror electronic health record intakes.
- **Limitations**:
  - Ground truth labels were generated under controlled clinical rules; real-world EHR records exhibit greater ambiguity and unstructured variance.
  - Class imbalance is pronounced: Urgency is skewed 7.60:1 and Hazard is skewed 138.43:1.
  - No synthetic rebalancing or class modification was performed, preserving true natural prevalence.

---

## 8. Patient-Level Split Strategy
- **Partition Scheme**: Group K-Split strictly by `patient_id` (Random seed = 42).
- **TRAIN**: 700 patients (4,261 documents, 69.88%)
- **VALIDATION**: 150 patients (909 documents, 14.91%)
- **LOCKED_TEST**: 150 patients (928 documents, 15.22%)
- **Cross-Split Patient Overlap**: Strictly **0.0%**.
- **Cross-Split Encounter Overlap**: Strictly **0.0%**.

---

## 9. Leakage Guardrails
- All vectorizers, scalers, encoders, and baselines are fitted **ONLY on the TRAIN partition**.
- The `VALIDATION` partition is used exclusively for model selection, threshold evaluation, and error analysis.
- The `LOCKED_TEST` partition is strictly held out and untouched.

---

## 10. Evaluation Methodology
- **Classification Tasks**: Macro F1 (primary safety metric), Weighted F1, Accuracy, and per-class Precision/Recall.
- **Entity Extraction Task**: Span-level Precision, Recall, and F1 score against ground-truth `ner_entities`.

---

## 11. Downstream Use Cases
1. **SLM Engineer**: The SLM Engineer will use these task definitions, preprocessing pipelines, and baseline benchmarks as the comparison standard for fine-tuning 1B–3B Small Language Models.
2. **Integration Engineer**: Tabular feature outputs (`nlp_features_*.parquet`) will be concatenated with Stage 1 tabular features for multimodal late fusion.

---

## 12. Explicit Non-Goals
- The NLP pipeline will **NOT** perform autonomous medical diagnosis.
- The NLP pipeline will **NOT** prescribe or adjust antineoplastic drug dosages.
- The NLP pipeline will **NOT** make autonomous clinical triage decisions without physician review.
