# Data Dictionary — Stage 3 Clinical NLP Dataset
**File**: `data/processed/clinical_nlp_dataset_v1.parquet` / `.jsonl` / `.csv`

| Column Name | Data Type | Nullable | Description | Example Value | Downstream Role |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `document_id` | `string` | No | Unique document identifier | `DOC-000124` | Primary Key |
| `patient_id` | `string` | No | Unique patient identifier | `PT-000452` | Patient Key / Group Split |
| `encounter_id` | `string` | No | Linked visit/encounter identifier | `ENC85214069` | Encounter Key |
| `document_type` | `string` | No | Controlled note type | `oncology_consultation` | Metadata / Stratification |
| `document_date` | `string` | No | Note timestamp (YYYY-MM-DD) | `2024-04-16` | Temporal Audit |
| `index_date` | `string` | No | Prediction cutoff timestamp | `2024-04-16` | Leakage Guardrail |
| `text` | `string` | No | De-identified sanitized clinical narrative | `ONCOLOGY CONSULTATION...` | Raw NLP Input |
| `cleaned_text` | `string` | No | Normalized text (NFKC, whitespace) | `ONCOLOGY CONSULTATION...` | Tokenizer Input |
| `word_count` | `int` | No | Token/word count | `184` | Quality Metric |
| `char_count` | `int` | No | Character count | `1248` | Quality Metric |
| `urgency_level` | `string` | No | Triage urgency class | `HIGH` | NLP Classification Target |
| `hazard_type` | `string` | No | Organ/system toxicity hazard | `RENAL` | Secondary Classification Target |
| `ner_entities` | `string` | No | JSON array of character span offsets | `[{"start": 42, "end": 46...}]` | NER Token Classification Target |
| `slm_summary` | `string` | No | 2-sentence clinical briefing | `62yo female with Stage III...` | SLM Seq2Seq / LoRA Target |
| `source` | `string` | No | Originating hospital clinic system | `EHR_ONCOLOGY_CLINIC` | Provenance Metadata |
| `data_split` | `string` | No | Split allocation | `TRAIN`, `VALIDATION`, `LOCKED_TEST` | Machine Learning Split |
| `quality_status` | `string` | No | Data validation status | `VALIDATED` | Quality Status |
| `disclaimer` | `string` | No | Research disclaimer | `RESEARCH PROTOTYPE...` | Governance |
