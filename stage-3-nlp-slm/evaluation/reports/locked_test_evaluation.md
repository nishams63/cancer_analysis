# Official Locked-Test Evaluation Report — Stage 3 Clinical NLP

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 3 — Clinical NLP & Baseline Systems Evaluation  
**Dataset Partition**: Official LOCKED TEST Partition ($N=928$ documents, 150 unique patients, 310 encounters)  
**Evaluator**: Independent Evaluation Engineer  
**Date**: 2026-09-09  
**Status**: OFFICIAL, FINAL & VIRGIN EVALUATION  

---

## 1. Executive Summary
This report presents the official, independent evaluation of the frozen Stage 3 Clinical NLP system on the strictly held-out **LOCKED TEST** partition.
Prior to this execution, the locked test partition was **never observed, tuned upon, or used for feature selection or threshold calibration**.

### High-Level Benchmark Summary (Locked Test Set)
- **Triage Urgency Accuracy**: **87.07%** (808 / 928 documents) [95% CI: 84.40% – 89.68%]
- **Triage Urgency Macro F1**: **0.7814** [95% CI: 0.7444 – 0.8183] (Weighted F1: 0.8768)
- **Safety-Critical Class Recall**: **97.00%** (97 / 100 true `CRITICAL` records captured) [95% CI: 93.33% – 100.0%]
- **Toxicity Hazard Accuracy**: **77.16%** (716 / 928 documents) [95% CI: 74.57% – 79.85%]
- **Toxicity Hazard Macro F1**: **0.5111** [95% CI: 0.4506 – 0.5649] (Weighted F1: 0.8228)
- **Clinical Entity Extraction (NER Relaxed F1)**: **0.7756** (Precision: 0.7145, Recall: 0.9037)
- **Clinical Entity Extraction (NER Exact F1)**: **0.6439** (Precision: 0.6047, Recall: 0.7352)

---

## 2. Partition Profile & Cohort Characteristics
- **Total Records ($N$)**: **928 clinical narratives**
- **Unique Patients**: **150 patients** (`PT-000851` through `PT-001000`)
- **Unique Encounters**: **310 distinct clinical encounters**
- **Patient Overlap against Train/Val**: Strictly **0.0%**
- **Encounter Overlap against Train/Val**: Strictly **0.0%**
- **Document Length**: Mean word count $106.01 \pm 22.41$ words (Range: 76 to 140 words).

---

## 3. Primary Classification Task: Triage Urgency Level

### 3.1 Point Estimates and 95% Patient-Clustered Bootstrap Confidence Intervals
Resampled across 1,000 patient clusters with fixed seed `42`:

| Metric | Point Estimate | Bootstrap Mean | Std Error | 95% CI Lower | 95% CI Upper | Generalization vs. Val |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | **0.8707** | 0.8705 | 0.0132 | **0.8440** | **0.8968** | +1.48% |
| **Macro Precision** | **0.7693** | 0.7688 | 0.0195 | **0.7312** | **0.8064** | +2.00% |
| **Macro Recall** | **0.7971** | 0.7976 | 0.0212 | **0.7557** | **0.8393** | +3.34% |
| **Macro F1-Score** | **0.7814** | 0.7811 | 0.0189 | **0.7444** | **0.8183** | **+0.0257** |
| **Weighted F1-Score** | **0.8768** | 0.8766 | 0.0127 | **0.8517** | **0.9014** | +0.0163 |
| **Critical Class Recall** | **0.9700** | 0.9695 | 0.0168 | **0.9333** | **1.0000** | **+2.43%** |

### 3.2 Granular Class-Wise Performance Breakdown
| Class | Precision | Recall | F1-Score | Support ($N$) | Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CRITICAL** | **0.9798** | **0.9700** | **0.9749** | 100 | 10.78% |
| **HIGH** | **0.6757** | **0.7519** | **0.7117** | 133 | 14.33% |
| **LOW** | **0.9726** | **0.9235** | **0.9474** | 614 | 66.16% |
| **MEDIUM** | **0.4490** | **0.5432** | **0.4916** | 81 | 8.73% |
| **Macro Average** | **0.7693** | **0.7971** | **0.7814** | 928 | 100.0% |
| **Weighted Average**| **0.8790** | **0.8707** | **0.8768** | 928 | 100.0% |

### 3.3 Locked-Test Confusion Matrix
```
True \ Pred      CRITICAL    HIGH     LOW    MEDIUM    Total
CRITICAL               97       1       0         2      100
HIGH                    0     100      12        21      133
LOW                     0      16     567        31      614
MEDIUM                  2      31       4        44       81
Total                  99     148     583        98      928
```

#### Key Confusion Matrix Observations:
1. **Zero Critical-to-Low Leaks**: Exactly 0 of 100 true `CRITICAL` records were misclassified as `LOW`. 
2. **Critical False Negatives**: Exactly 3 true `CRITICAL` documents were missed (1 classified as `HIGH`, 2 as `MEDIUM`).
3. **Mid-Tier Class Ambiguity**: The primary classification error remains adjacent tier confusion: 31 of 81 true `MEDIUM` records (38.27%) were classified as `HIGH`, and 21 of 133 true `HIGH` records (15.79%) were classified as `MEDIUM`.

---

## 4. Secondary Classification Task: Toxicity Hazard Type

### 4.1 Overall Performance and Confidence Intervals
| Metric | Point Estimate | Bootstrap Mean | Std Error | 95% CI Lower | 95% CI Upper |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | **0.7716** | 0.7717 | 0.0136 | **0.7457** | **0.7985** |
| **Macro Precision** | **0.4636** | 0.4612 | 0.0381 | **0.3912** | **0.5360** |
| **Macro Recall** | **0.7545** | 0.7375 | 0.0560 | **0.6047** | **0.8143** |
| **Macro F1-Score** | **0.5111** | 0.5062 | 0.0300 | **0.4506** | **0.5649** |
| **Weighted F1-Score** | **0.8228** | 0.8238 | 0.0111 | **0.8020** | **0.8454** |
| **Hepatic Recall** | **0.8831** | 0.8832 | 0.0359 | **0.8068** | **0.9494** |

### 4.2 Class-Wise Breakdown across 8 Hazard Types
| Toxicity Category | Precision | Recall | F1-Score | Support ($N$) | Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CARDIAC** | 0.2000 | 1.0000 | 0.3333 | 2 | 0.22% |
| **DERMATOLOGIC** | 0.0667 | 0.8000 | 0.1231 | 5 | 0.54% |
| **HEMATOLOGIC** | 0.4400 | 0.7333 | 0.5500 | 30 | 3.23% |
| **HEPATIC** | 0.6538 | 0.8831 | 0.7514 | 77 | 8.30% |
| **NEUROPATHIC** | 0.2727 | 0.5000 | 0.3529 | 12 | 1.29% |
| **NONE** | 0.9892 | 0.7587 | 0.8587 | 721 | 77.69% |
| **PULMONARY** | 0.9836 | 0.9231 | 0.9524 | 65 | 7.00% |
| **RENAL** | 0.1029 | 0.4375 | 0.1667 | 16 | 1.72% |

---

## 5. Clinical Concept Extraction (NER) on Locked Test

Evaluated across all 928 locked test narratives:

| Entity Type | Predicted Count | Gold Count | Exact F1 | Relaxed Precision | Relaxed Recall | Relaxed F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GENE_MUTATION** | 688 | 848 | **0.8958** | **1.0000** | **0.8113** | **0.8958** |
| **DRUG_NAME** | 695 | 778 | **0.7142** | **0.9266** | **0.8278** | **0.8744** |
| **DOSAGE** | 1,951 | 778 | **0.5694** | **0.3983** | **0.9987** | **0.5694** |
| **ADVERSE_EVENT** | 1,481 | 929 | **0.3759** | **0.5908** | **0.9419** | **0.7261** |
| **Overall Macro Average** | — | — | **0.6439** | **0.7145** | **0.9037** | **0.7756** |
| **Overall Micro Average** | 4,815 | 3,333 | **0.5999** | **0.6197** | **0.8953** | **0.7324** |

---

## 6. Major Error Categories & Known Limitations
1. **Adverse Event Exact Span Mismatch**: Overlap recall is 94.19%, but exact span F1 is 0.3759 due to pre-nominal modifiers (`mild nausea` vs `nausea`).
2. **Dosage Spurious Captures**: Blood pressure (`120/80 mmHg`) and laboratory biomarkers (`1.2 mg/dL`) match regex patterns, depressing dosage precision to 39.83%.
3. **Mid-Tier Urgency Ambiguity**: Intermediate risk cases (`MEDIUM` vs `HIGH`) lack sharp vocabulary distinctions, causing F1 for `MEDIUM` to sit at 0.4916.
4. **Extreme Toxicity Imbalance**: Rare hazards (`DERMATOLOGIC`, `CARDIAC`, `RENAL`) suffer from false positives driven by uniform baseline weighting.

---

## 7. Official Locked-Test Conclusion
The locked test evaluation proves that the Stage 3 Clinical NLP baseline generalizes cleanly to unseen patients:
- **No Performance Degradation**: Urgency Macro F1 improved from 0.7557 (Val) to **0.7814** (Test).
- **Safety Invariant Maintained**: Critical recall reached **97.00%** on unseen patients.
- **Strict Leakage Free**: Certified zero patient and encounter overlap.
This benchmark provides a firm, reliable comparison standard for the upcoming SLM Engineer.
