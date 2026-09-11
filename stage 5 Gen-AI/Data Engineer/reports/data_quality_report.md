# Stage 5 Data Quality & Quarantine Report

**Execution Timestamp**: 2026-09-11T16:42:59.434614+00:00  
**Dataset Version**: v1.0  
**Quality Policy**: Quarantine & Flag (Zero Silent Deletion)  

---

## 1. Input Datasets Audited

| Source ID | Name | Rows | Columns | File Hash (SHA-256) |
| :--- | :--- | :---: | :---: | :--- |
| **PROJECT_STAGE1** | Master Patient Tabular Dataset | 8,754 | 35 | `b11cf0a33cf42415...` |
| **PROJECT_STAGE2** | Longitudinal Biomarker Trajectories | 617 | 21 | `0e3c6f38a3958269...` |
| **PROJECT_STAGE3** | Clinical NLP Progress Notes | 6,098 | 18 | `426ea0c51f354a5f...` |
| **PROJECT_STAGE4** | Instruction-Tuning Training Pairs | 5,706 | 17 | `95d684c0940be347...` |

---

## 2. Cleaning & Quarantine Issue Summary

Total Tracked Issues: **7094**

| Issue Type | Occurrence Count | Mitigation & Action Taken |
| :--- | :---: | :--- |
| `unrecognized_mutation` | 7,060 | Quarantined / Standardized per clinical ontology dictionary |
| `out_of_physiological_bounds` | 29 | Quarantined / Standardized per clinical ontology dictionary |
| `invalid_range` | 5 | Quarantined / Standardized per clinical ontology dictionary |

---

## 3. Normalization Invariants
- **Demographics**: Clamped age bounds to 18–105 years. Standardized sex labels into canonical `['Male', 'Female', 'Unknown']`.
- **Genomic Mutations**: Cleaned wildtype strings (`None`, `Unknown`, `None/Unknown`) and resolved variant prefixes into canonical gene symbols (`KRAS`, `EGFR`, `TP53`, `ALK`, `MET`, `BRAF`).
- **Physiological Bounds**: Bounded vital signs (systolic BP 60–240 mmHg, heart rate 30–220 bpm) and organ markers (creatinine 0.1–15.0 mg/dL, ALT/AST 1.0–1000.0 U/L).
- **Dosages**: Asserted strictly non-negative dosages and standardized drug names to FDA approved trade/generic names.
- **Audit Lineage**: Complete issue logs are persisted to `C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/interim/cleaned/cleaning_issues_log.json`.
