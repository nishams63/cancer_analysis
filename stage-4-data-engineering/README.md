# Stage 4 — Data Engineering: SLM Fine-Tuning Dataset Pipeline (v2)

**Academic Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: `stage-4-data-engineering`  
**Version**: `2.0.0`  
**Author**: Data Engineering Lead (Stage 4)  
**Status**: Production-Ready & Verified (`patient_leakage = 0`)  

> [!IMPORTANT]
> **Governance & Scope Certification**:
> "Stage 4 does not create a new raw medical dataset. It creates a derived fine-tuning dataset from Stage 3 outputs, using Stage 3's NER as a high-confidence but imperfect reference."

---

## 1. Overview & Purpose

Stage 4 is responsible for ingesting existing Stage 3 clinical notes and verified Named Entity Recognition (NER) outputs, transforming them into a structured, entity-verified, instruction-tuning dataset for Small Language Model (SLM) fine-tuning in Stage 5.

The pipeline ensures:
1. **Zero External Data & Upstream Invariance**: Operates strictly on Stage 3 curated notes without collecting external data or altering frozen Stage 3 artifacts.
2. **Clinical Entity Preservation**: Verifies that every generated training target preserves all key antineoplastic drugs, genomic driver mutations, dosages, and adverse events from Stage 3.
3. **Safety Normalization**: Safely handles case folding, whitespace/Unicode NFKC, dosage format equivalence, dosage ranges, and versioned drug synonyms without changing clinical meaning.
4. **NER Imperfection Calibration**: Explicitly incorporates Stage 3's reported NER baseline ($F_1 = 76.70\%$) into quality assessments, routing samples to human review to distinguish draft generation issues from upstream NER boundary artifacts.
5. **Rejection-Rate Circuit Breaker**: Halts the pipeline if dataset or batch rejection rates exceed the configured 30% safety threshold.
6. **Patient-Level Isolation**: Enforces strict patient-level splitting ($70\%$ Train / $15\%$ Validation / $15\%$ Test) with zero patient or encounter leakage (`patient_leakage = 0`).

---

## 2. Pipeline Architecture

```text
[ Stage 3 Clinical Notes + NER ] (clinical_nlp_dataset_v1.parquet)
              │
              ▼
    [ 1. Robust Data Loader ] ──> Logs missing IDs, joins on document_id & patient_id
              │
              ▼
    [ 2. Data Validator ] ──────> Audits empty/duplicate notes, malformed entities
              │
              ▼
    [ 3. Draft Target Generator ]
              ├─> Generates Risk, Key Finding, Action targets
              └─> Logs raw output, prompt version, and model to generation_log.parquet
              │
              ▼
    [ 4. Entity Preservation Quality Gate ]
              ├─> Normalizes case, dosage spacing ("5 mg" <=> "5mg"), ranges, units
              ├─> Maps generic/brand synonyms via versioned drug_synonyms.yaml
              ├─> Flags/Rejects missing or invented entities
              └─> Calibrates against Stage 3 NER F1 (76.70%)
              │
              ▼
    [ 5. Circuit Breaker Monitor ]
              ├─> Rejection rate <= 30.0% -> CONTINUE
              └─> Rejection rate > 30.0%  -> HALT with CircuitBreakerError
              │
              ▼
    [ 6. Instruction Formatter ]
              ├─> Standardizes PASS records into {instruction, input, target_risk, ...}
              └─> Preserves rejected pairs in rejected_pairs.parquet
              │
              ▼
    [ 7. Patient-Level Splitter ] ──> 700 Train / 150 Val / 150 Test unique patients
              │
              ▼
    [ 8. Multi-Dimensional Leakage Audit ]
              ├─> Patient Overlap Check: Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅
              ├─> Exact SHA-256 Text Match across splits = 0
              ├─> Near-Duplicate TF-IDF Check across splits
              └─> Temporal Validity & Forbidden Outcome Phrases Check
              │
              ▼
    [ 9. Human-Review Sample Audit (N = 70) ]
              ├─> Audits clinical sensibility of PASS samples
              └─> Categorizes REJECT samples: Generation Error vs. Upstream NER Noise
              │
              ▼
    [ 10. Data Quality Reports & Artifacts ]
              ├─> slm_finetune_dataset_v1.parquet
              ├─> data_quality_report.json / .md
              └─> manual_review_audit.md
```

---

## 3. Input and Output Specifications

### Upstream Inputs (Read-Only)
| Input Path | Source | Description |
|:---|:---|:---|
| `stage-3-nlp-slm/data-engineering/data/processed/clinical_nlp_dataset_v1.parquet` | Stage 3 Data Engineering | 6,098 validated clinical oncology records across 1,000 patients |
| `stage-3-nlp-slm/nlp/results/metrics/baseline_validation_metrics.json` | Stage 3 Clinical NLP | Baseline model metrics: Urgency Macro F1 (0.7557), Hazard Macro F1 (0.5214), NER Span F1 (0.7670) |
| `stage-3-nlp-slm/nlp/reports/baseline_report.md` | Stage 3 Clinical NLP | Authoritative Stage 3 baseline evaluation report |

### Pipeline Outputs
| Output Path | Type | Records | Description |
|:---|:---:|:---:|:---|
| `stage-4-data-engineering/data/slm_finetune_dataset_v1.parquet` | Parquet | **5,706** | Final entity-verified instruction-tuning dataset |
| `stage-4-data-engineering/data/rejected_pairs.parquet` | Parquet | **392** | Pre-validation defects and quality gate rejected pairs with reasons |
| `stage-4-data-engineering/data/generation_log.parquet` | Parquet | **5,761** | Raw generator outputs with model version, prompt template, and timestamp |
| `stage-4-data-engineering/reports/data_quality_report.json` | JSON | — | Machine-readable comprehensive quality and leakage metrics |
| `stage-4-data-engineering/reports/data_quality_report.md` | Markdown | — | Human-readable executive summary and certification report |
| `stage-4-data-engineering/reports/manual_review_audit.md` | Markdown | — | Double-cohort spot-check audit (35 PASS and 35 REJECT examples) |

---

## 4. Entity Preservation, Normalization & Quality Gate

### Normalization Rules (Section 4a)
1. **Case & Unicode Normalization**: Applies NFKC unicode standard normalization, trims leading/trailing whitespace, and performs lowercased comparison.
2. **Dosage Spacing Equivalence**: Regex parses numerical magnitude and unit, equating `"5 mg"` with `"5mg"` and `"265.4 mg"` with `"265.4mg"`.
3. **Dosage Range Matching**: Evaluates interval syntax (e.g. `"5-10 mg"` or `"5–10 mg"` with en-dash). A target stating a single value within the range (e.g. `"7 mg"`) is accepted as a valid match.
4. **Safe Unit Conversions**: Supports safe metric scaling (e.g. `5000 mcg` $\leftrightarrow$ `5 mg`, `1000 mg` $\leftrightarrow$ `1 g`). Ambiguous unit transformations are flagged.
5. **Versioned Drug Synonym Table**: Checks generic and brand synonyms against `config/drug_synonyms.yaml` (e.g. `Warfarin` $\leftrightarrow$ `Coumadin`, `Erlotinib` $\leftrightarrow$ `Tarceva`, `Docetaxel` $\leftrightarrow$ `Taxotere`, `Pembrolizumab` $\leftrightarrow$ `Keytruda`). Any synonym not in the checked-in file is strictly treated as a genuine mismatch.

### Quality Gate Categories (Section 6)
- **Missing Entities**: An entity present in Stage 3 NER but omitted from the generated target is flagged/rejected.
- **Invented Entities**: An entity present in the target but unsupported by the source note or Stage 3 NER is flagged/rejected.
- **Correct Preservation**: All entities present without unsupported additions $\rightarrow$ **PASS**.

### Stage 3 NER Baseline Calibration (Section 6a)
Stage 3 reports an entity extraction Span F1 of **76.70%** (Precision: 71.91%, Recall: 87.97%). Because NER is high-confidence but not ground truth, human review confirmed that:
- **11.4%** of rejections were true draft generation omissions.
- **88.6%** of rejections were upstream NER artifacts (e.g., secondary lab values or multi-word boundary offsets).

---

## 5. Rejection-Rate Circuit Breaker (Section 6b)

To prevent silent dataset degradation:
- **Threshold**: $\le 30.0\%$ rejection rate allowed.
- **Warning Threshold**: $20.0\%$.
- **Measured Pipeline Rejection Rate**: **0.95%** (55 rejections out of 5,761 drafted records).
- **Status**: **PASSED**.

---

## 6. Patient-Level Splitting & Leakage Audit

### Partition Distribution
Splits are strictly partitioned at the patient level:
- **TRAIN**: 3,996 records (700 unique patients, 70.0%)
- **VALIDATION**: 849 records (150 unique patients, 14.9%)
- **TEST**: 861 records (150 unique patients, 15.1%)
- **Total Unique Patients**: 1,000

### Leakage Audit Certification
$$\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Val}) = \emptyset$$
$$\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Test}) = \emptyset$$
$$\text{Patients}(\text{Val}) \cap \text{Patients}(\text{Test}) = \emptyset$$

- **Patient Leakage**: **0** (`patient_leakage = 0`).
- **Cross-Split Exact SHA-256 Duplicates**: **0**.
- **Temporal Sequence Inversions ($\text{Doc Date} > \text{Index Date}$)**: **0**.
- **Forbidden Prospective Outcome Keywords**: **0** occurrences.

---

## 7. Example Instruction-Tuning Record

```json
{
  "patient_id": "PT-000001",
  "note_id": "DOC-000001",
  "instruction": "Analyze the clinical note and provide the patient's Risk, Key Finding, and Action.",
  "clinical_note": "ONCOLOGY CONSULTATION PROGRESS NOTE\nPatient ID: PT-000001\nDemographics: 62-year-old male presenting for Cycle 4 evaluation...\nPlan: Continue Docetaxel 265.4 mg IV every 3 weeks...",
  "target_risk": "Increased no acute adverse toxicities and systemic toxicity hazard associated with Docetaxel (265.4 mg) therapy.",
  "target_key_finding": "Kras variant was identified; patient received docetaxel at 265.4 mg; developed no acute adverse toxicities.",
  "target_action": "Continue standard clinical monitoring and maintain current Docetaxel regimen as tolerated.",
  "ner_genes": ["KRAS", "None/Unknown"],
  "ner_drugs": ["Docetaxel"],
  "ner_dosages": ["265.4 mg"],
  "ner_adverse_events": ["no acute adverse toxicities"],
  "entity_check_status": "PASS",
  "entity_coverage": 1.0,
  "missing_entities": [],
  "invented_entities": [],
  "generation_model_version": "clinical-draft-target-generator-v1.0.0",
  "split": "VALIDATION"
}
```

---

## 8. Execution Instructions

### Run Unit Tests
```powershell
.venv\Scripts\pytest -v stage-4-data-engineering/tests
```

### Run End-to-End Pipeline
```powershell
.venv\Scripts\python stage-4-data-engineering/src/pipeline.py
```

---

## 9. Known Limitations

1. **LLM Generation Non-Determinism**: Draft generation performed via non-deterministic LLMs can vary between runs. While Stage 4 version-pins prompt templates and hyperparameters in `config/generation_config.yaml` and logs all raw outputs to `generation_log.parquet`, the raw generation step itself is exempted from exact bitwise reproducibility. Given fixed drafts, all validation, gate checks, splitting, and exports are 100% deterministic.
2. **Upstream NER Noise**: Stage 3 NER operates with an empirical Span F1 of 76.70%. A minority of gate rejections are driven by upstream NER span offsets rather than actual generation errors.
3. **Synthetic Template Overlap**: Notes derived from standardized EHR formats (e.g. surgical pathology templates) share common heading text across different patients; patient-level splitting guarantees zero clinical record leakage.
