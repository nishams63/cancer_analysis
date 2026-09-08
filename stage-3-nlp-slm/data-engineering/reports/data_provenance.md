# Data Source and Provenance Report
**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 3 — NLP & SLM Data Engineering  
**Version**: `processed_v1.0.0`

---

## 1. Upstream Data Lineage

```text
[Master Patient Dataset (Stage 1 ML)]
          │ (1,000 Unique Patients: PT-000001 to PT-001000, Demographics, Genomics, Labs, Regimens)
          ▼
[Synthetic Clinical Narrative Generator (seed=42)]
          │ (Synthesizes Consultation Notes, Pathology Reports, Nurse Intakes, Symptom Logs)
          ▼
[raw_clinical_notes_v1.jsonl] (Raw Ingestion: 6,135 records)
          │
          ├──> Exact Deduplication (Exact row matches dropped)
          ├──> Privacy Sanitization (HIPAA Safe Harbor regex masking)
          ├──> Text Normalization (Unicode NFKC, whitespace, semantic preservation)
          ├──> Leakage Filtering (Temporal boundary document_date <= index_date)
          └──> Patient-Level Splitting (700 Train / 150 Val / 150 Locked Test)
          ▼
[clinical_nlp_dataset_v1.parquet] (Canonical Clean Dataset: 6,098 records)
```

## 2. Source Metadata
- **Upstream Source**: `stage-1-ml/data-engineering/data/processed/master_patient_dataset.csv`
- **Acquisition Date**: Aligned with master cohort encounter observation dates.
- **Population Represented**: 1,000 synthetic cancer patients across multiple oncologic indications (NSCLC, Colorectal, Breast, Pancreatic, Prostate) with genomic driver mutations (EGFR, KRAS, TP53, BRAF, ALK).
- **Transformation History**: Raw multi-source consultation logs synthesized, standardized, and validated with complete reproducible seed = 42.
