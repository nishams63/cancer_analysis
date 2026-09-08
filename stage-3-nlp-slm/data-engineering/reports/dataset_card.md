# Dataset Card: Stage 3 Clinical NLP & SLM Dataset

## Dataset Summary
- **Dataset Name**: Clinical Oncology NLP & SLM Research Dataset (`processed_v1.0.0`)
- **Domain**: Precision Oncology, Pharmacogenomics, Toxicity Triage, and Clinical Summarization
- **Total Records**: 6,098 validated clinical documents
- **Cohort Size**: 1,000 unique synthetic cancer patients (2,038 encounters)
- **Primary Tasks**:
  1. **Urgency Classification**: 4-class severity triage (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)
  2. **Adverse Event Hazard Categorization**: 8 organ/system toxicity classes
  3. **Named Entity Recognition (NER)**: `GENE_MUTATION`, `DRUG_NAME`, `DOSAGE`, `ADVERSE_EVENT`
  4. **SLM Fine-Tuning**: 2-sentence voice-ready bedside briefing generation

---

## Split Strategy
- **Partitioning Method**: Strict Patient-Level Splitting (Group K-Split)
- **Random Seed**: 42 (100% Deterministic)
- **Train Split**: 4,261 documents (700 patients, 70.0%)
- **Validation Split**: 909 documents (150 patients, 15.0%)
- **Locked Test Split**: 928 documents (150 patients, 15.0%)
- **Cross-Split Patient Overlap**: **0.0%**
- **Cross-Split Encounter Overlap**: **0.0%**

---

## Limitations & Disclaimers
- **Synthetic Data**: Created for NLP and SLM research prototyping. Not real clinical records.
- **Not Clinically Validated**: Models trained on this dataset must not be used for direct patient diagnosis or autonomous medical prescription without formal clinical trials and institutional review.
