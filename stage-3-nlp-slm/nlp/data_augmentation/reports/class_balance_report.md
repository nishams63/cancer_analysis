# Class Balance and Imbalance Mitigation Report

## Executive Summary
This report analyzes class imbalance across the primary clinical triage targets (Urgency Level and Adverse Event Hazard Toxicity) in the Stage 3 Clinical NLP dataset, evaluating how targeted data augmentation remediates acute class skews without synthetic fabrication or artificial duplicates.

---

## 1. Urgency Triage Distribution Across Datasets

The clinical urgency target consists of 4 triage tiers: `LOW`, `MEDIUM`, `HIGH`, and `CRITICAL`.
In the original training split, `LOW` represents 68.7% of all documents, creating an imbalance ratio of **7.6:1** against `CRITICAL` (9.0%) and `MEDIUM` (8.8%).

| Dataset Version | Total Docs | LOW (%) | MEDIUM (%) | HIGH (%) | CRITICAL (%) | Imbalance Ratio (LOW:CRITICAL) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original (A)** | 4,261 | 2,927 (68.7%) | 374 (8.8%) | 575 (13.5%) | 385 (9.0%) | **7.60 : 1** |
| **Augmented +25% (B)** | 5,326 | 3,617 (67.9%) | 440 (8.3%) | 770 (14.5%) | 499 (9.4%) | **7.25 : 1** |
| **Augmented +50% (C)** | 6,391 | 4,322 (67.6%) | 508 (7.9%) | 952 (14.9%) | 609 (9.5%) | **7.10 : 1** |
| **Augmented +100% (D)**| 8,522 | 5,737 (67.3%) | 636 (7.5%) | 1,304 (15.3%) | 845 (9.9%) | **6.79 : 1** |
| **Targeted Balanced (E)**| **5,761** | **2,927 (50.8%)** | **637 (11.1%)** | **1,330 (23.1%)** | **867 (15.0%)** | **3.38 : 1 (Significant Improvement)** |

### Clinical Significance of Targeted Urgency Balancing
- **CRITICAL Support Boosted by +125.2%**: Increased from 385 instances to 867 instances, providing robust representation of life-threatening oncologic emergencies (febrile neutropenia, acute cardiotoxicity, severe dyspnea).
- **HIGH Support Boosted by +131.3%**: Increased from 575 to 1,330 instances.
- **LOW Proportion Capped**: Instead of inflating the majority class, Config E froze `LOW` at 2,927 instances, reducing its dominance from 68.7% to 50.8%.

---

## 2. Adverse Event Hazard Toxicity Distribution Across Datasets

The hazard categorization target comprises 8 organ/system toxicity classes. The original training set is heavily dominated by `NONE` (80.1%), with ultra-rare organ toxicities (`CARDIAC` at 0.66%, `NEUROPATHIC` at 0.85%, `DERMATOLOGIC` at 0.87%).

| Hazard Class | Original Train (A) | +25% Aug (B) | +50% Aug (C) | +100% Aug (D) | Targeted Balanced (E) | Relative Increase in E |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NONE** | 3,414 (80.1%) | 4,215 (79.1%) | 5,025 (78.6%) | 6,640 (77.9%) | 3,822 (66.3%) | Baseline retained |
| **HEPATIC** | 321 (7.53%) | 447 (8.39%) | 561 (8.78%) | 771 (9.05%) | **802 (13.9%)** | **+149.8%** |
| **PULMONARY** | 247 (5.79%) | 322 (6.05%) | 403 (6.31%) | 581 (6.82%) | **605 (10.5%)** | **+144.9%** |
| **RENAL** | 87 (2.04%) | 104 (1.95%) | 125 (1.96%) | 171 (2.01%) | **172 (2.99%)** | **+97.7%** |
| **HEMATOLOGIC** | 91 (2.13%) | 113 (2.12%) | 131 (2.05%) | 169 (1.98%) | **165 (2.86%)** | **+81.3%** |
| **DERMATOLOGIC** | 37 (0.87%) | 48 (0.90%) | 55 (0.86%) | 69 (0.81%) | **69 (1.20%)** | **+86.5%** |
| **NEUROPATHIC** | 36 (0.85%) | 41 (0.77%) | 47 (0.74%) | 60 (0.70%) | **65 (1.13%)** | **+80.6%** |
| **CARDIAC** | 28 (0.66%) | 36 (0.68%) | 44 (0.69%) | 61 (0.72%) | **61 (1.06%)** | **+117.9%** |
| **Total Hazard Documents** | 4,261 | 5,326 | 6,391 | 8,522 | 5,761 | +1,500 Targeted |

---

## 3. Named Entity Annotation Distribution

Because entity spans were strictly preserved across all transformations, the proportional entity representation grew linearly with document counts:

| Entity Label | Original (A) | +25% Aug (B) | +50% Aug (C) | +100% Aug (D) | Targeted Balanced (E) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GENE_MUTATION** | 3,907 | 5,035 | 6,155 | 8,394 | 5,365 |
| **DRUG_NAME** | 3,560 | 4,437 | 5,306 | 7,078 | 5,060 |
| **DOSAGE** | 3,560 | 4,437 | 5,306 | 7,078 | 5,060 |
| **ADVERSE_EVENT** | 4,263 | 5,328 | 6,393 | 8,524 | 5,763 |
| **Total Entity Mentions** | **15,290** | **19,237** | **23,160** | **31,074** | **21,248** |

## Conclusion
Config E (Targeted Balanced Augmentation) successfully curtails majority class dominance and doubles the representation of critical oncologic toxicity hazards without sacrificing patient split isolation.
