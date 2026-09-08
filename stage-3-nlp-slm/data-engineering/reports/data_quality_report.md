# Data Quality Audit Report — Stage 3 NLP + SLM
**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Dataset Version**: `processed_v1.0.0`  
**Execution Date**: 2026-09-09 02:17:15  
**Status**: APPROVED & VALIDATED (Quality Gate Passed)

---

## 1. Executive Summary & Audit Ledger
This report documents the rigorous data engineering quality gate executed on the raw clinical narrative dataset for Stage 3 (NLP & SLM). Every record was evaluated against strict schema rules, text cleanliness constraints, temporal ordering, and outcome leakage filters.

| Metric | Raw Ingestion | Post-Engineering Final | Delta / Action Taken |
| :--- | :---: | :---: | :--- |
| **Total Documents** | 6,135 | **6,098** | -37 records removed |
| **Unique Patients** | 1,000 | **1,000** | 1,000 unique patients maintained |
| **Unique Encounters** | 2,060 | **2,038** | Complete cohort coverage |
| **Exact Row Duplicates** | 15 | **0** | Removed via exact hash matching |
| **Composite Key Collisions** | 15 | **0** | Resolved to 1 note per type per encounter |
| **Empty/Corrupt Text** | 10 | **0** | Filtered out via length/character checks |
| **Outcome Leakage Notes** | 12 | **0** | Dropped to prevent target leakage |
| **Unmasked Direct PII** | 8 | **0** | 100% Sanitized via HIPAA Safe Harbor |

---

## 2. Text Quality & Length Distribution

| Parameter | Word Count | Character Count |
| :--- | :---: | :---: |
| **Mean** | 103.4 words | 781.9 characters |
| **Median** | 96.0 words | 772.0 characters |
| **Standard Deviation** | 21.8 words | 178.0 characters |
| **Minimum** | 75 words | 504 characters |
| **Maximum** | 139 words | 1055 characters |

*Quality Rule: All documents satisfy $15 \le \text{word\_count} \le 850$. Zero empty, null, or whitespace-only documents exist in the final processed dataset.*

---

## 3. Label & Class Distributions

### Urgency Level Distribution (Intake Triage Target)
| Urgency Level | Document Count | Percentage |
| :--- | :---: | :---: |
| `LOW` | **4,140** | 67.9% |
| `HIGH` | **836** | 13.7% |
| `CRITICAL` | **577** | 9.5% |
| `MEDIUM` | **545** | 8.9% |

### Adverse Event Hazard Type Distribution
| Hazard Type | Document Count | Percentage |
| :--- | :---: | :---: |
| `NONE` | **4,845** | 79.5% |
| `HEPATIC` | **474** | 7.8% |
| `PULMONARY` | **367** | 6.0% |
| `HEMATOLOGIC` | **144** | 2.4% |
| `RENAL` | **127** | 2.1% |
| `NEUROPATHIC` | **60** | 1.0% |
| `DERMATOLOGIC` | **46** | 0.8% |
| `CARDIAC` | **35** | 0.6% |

### Document Types
| Document Type | Document Count | Percentage |
| :--- | :---: | :---: |
| `oncology_consultation` | **2,033** | 33.3% |
| `nurse_intake_note` | **1,577** | 25.9% |
| `patient_symptom_log` | **1,487** | 24.4% |
| `pathology_report` | **1,001** | 16.4% |

---

## 4. Patient Split Verification
- **Patient Leakage Across Splits**: **0.0%** (Zero overlapping patients)
- **Encounter Leakage Across Splits**: **0.0%** (Zero overlapping encounters)
- **Train Split (70.0%)**: 4,261 documents (700 patients)
- **Validation Split (15.0%)**: 909 documents (150 patients)
- **Locked Test Split (15.0%)**: 928 documents (150 patients)
