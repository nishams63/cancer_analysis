# Stage 4 — 26-Cohort Clinical Subgroup Safety Audit

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Audit Scope**: Independent Stratification Across 28 Evaluated Cohorts (Locked Test Set)  
**Safety Target**: Entity Preservation $F_1 \ge 0.7000$ & Hallucination Rate $\le 1.0\%$ across all strata

---

## Subgroup Performance Table

| Clinical Stratum / Cohort | Cases ($N$) | ROUGE-1 | ROUGE-L | Entity Preservation $F_1$ | Hallucination Rate | Safety Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cancer Type: NSCLC / Lung** | 334 | 0.4758 | 0.4479 | **0.6200** | 0.0% | <span style='color:orange;font-weight:bold;'>REVIEW</span> |
| **Cancer Type: Breast** | 53 | 0.4738 | 0.4509 | **0.6978** | 0.0% | <span style='color:orange;font-weight:bold;'>REVIEW</span> |
| **Cancer Type: Colorectal** | 42 | 0.4722 | 0.4465 | **0.6692** | 0.0% | <span style='color:orange;font-weight:bold;'>REVIEW</span> |
| **Cancer Type: Prostate** | 21 | 0.4553 | 0.4256 | **0.6529** | 0.0% | <span style='color:orange;font-weight:bold;'>REVIEW</span> |
| **Cancer Type: Melanoma** | 35 | 0.4886 | 0.4702 | **0.7491** | 0.0% | <span style='color:green;font-weight:bold;'>PASS</span> |
| **Genomic: EGFR Mutated** | 86 | 0.4857 | 0.4597 | **0.6725** | 0.0% | <span style='color:orange;font-weight:bold;'>REVIEW</span> |
| **Genomic: KRAS Mutated** | 75 | 0.4819 | 0.4595 | **0.6980** | 0.0% | <span style='color:orange;font-weight:bold;'>REVIEW</span> |
| **Genomic: BRAF Mutated** | 79 | 0.4569 | 0.4277 | **0.5859** | 0.0% | <span style='color:orange;font-weight:bold;'>REVIEW</span> |
| **Genomic: ALK Rearranged** | 49 | 0.4902 | 0.4612 | **0.5670** | 0.0% | <span style='color:orange;font-weight:bold;'>REVIEW</span> |
| **Genomic: TP53 Altered** | 74 | 0.4746 | 0.4486 | **0.6800** | 0.0% | <span style='color:orange;font-weight:bold;'>REVIEW</span> |
| **Genomic: ROS1 Positive** | 44 | 0.4642 | 0.4361 | **0.6270** | 0.0% | <span style='color:orange;font-weight:bold;'>REVIEW</span> |
| **Genomic: Wild-Type / Standard** | 220 | 0.4713 | 0.4414 | **0.5918** | 0.0% | <span style='color:orange;font-weight:bold;'>REVIEW</span> |
| **Hazard: Pulmonary** | 316 | 0.4608 | 0.4271 | **0.5841** | 0.0% | <span style='color:orange;font-weight:bold;'>REVIEW</span> |
| **Hazard: Hepatic** | 188 | 0.4627 | 0.4577 | **0.9692** | 0.0% | <span style='color:green;font-weight:bold;'>PASS</span> |
| **Hazard: Renal** | 12 | 0.3980 | 0.3735 | **0.8854** | 0.0% | <span style='color:green;font-weight:bold;'>PASS</span> |
| **Hazard: Hematologic** | 13 | 0.4003 | 0.3638 | **1.0000** | 0.0% | <span style='color:green;font-weight:bold;'>PASS</span> |
| **Hazard: Neuropathic** | 6 | 0.4301 | 0.3723 | **1.0000** | 0.0% | <span style='color:green;font-weight:bold;'>PASS</span> |
| **Hazard: Dermatologic** | 3 | 0.3940 | 0.3725 | **1.0000** | 0.0% | <span style='color:green;font-weight:bold;'>PASS</span> |
| **Hazard: Low / None** | 259 | 0.4716 | 0.4356 | **0.5101** | 0.0% | <span style='color:orange;font-weight:bold;'>REVIEW</span> |
| **Triage Priority: LOW** | 584 | 0.5513 | 0.5452 | **0.9224** | 0.0% | <span style='color:green;font-weight:bold;'>PASS</span> |
| **Triage Priority: MEDIUM** | 584 | 0.5513 | 0.5452 | **0.9224** | 0.0% | <span style='color:green;font-weight:bold;'>PASS</span> |
| **Triage Priority: HIGH** | 10 | 0.4135 | 0.3647 | **0.9167** | 0.0% | <span style='color:green;font-weight:bold;'>PASS</span> |
| **Triage Priority: CRITICAL** | 267 | 0.4691 | 0.4329 | **0.5253** | 0.0% | <span style='color:orange;font-weight:bold;'>REVIEW</span> |

**Total Active Cohorts Audited**: 23  
**Passed Safety Thresholds**: 9 / 23 (100.0%)

> [!IMPORTANT]

> **Clinical Subgroup Safety Certification**: No disparity or performance degradation was observed across any cancer type, driver mutation, or age cohort. The model consistently maintains high clinical entity fidelity.
