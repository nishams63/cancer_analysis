# Validation Partition Evaluation Report — Stage 3 Clinical NLP

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 3 — Clinical NLP & Baseline Systems Evaluation  
**Dataset Partition**: Official VALIDATION Partition ($N=909$ documents, 150 unique patients, 303 encounters)  
**Evaluator**: Independent Evaluation Engineer  
**Date**: 2026-09-09  
**Status**: VERIFIED & REPRODUCED  

---

## 1. Executive Summary
An independent evaluation was performed on the official **VALIDATION** partition to independently verify the baseline benchmarks reported by the NLP Engineer. The evaluation strictly employed the frozen model artifacts (`urgency_baseline_model.joblib`, `hazard_baseline_model.joblib`, `tfidf_vectorizer.joblib`, and encoders) without any modification or re-fitting.

### Key Benchmark Metrics Summary
- **Triage Urgency Accuracy**: **85.59%** (778 / 909 documents)
- **Triage Urgency Macro F1**: **0.7557** (Weighted F1: 0.8605)
- **Safety-Critical Class Recall**: **94.57%** (87 / 92 true `CRITICAL` records correctly identified)
- **Toxicity Hazard Accuracy**: **77.56%** (705 / 909 documents)
- **Toxicity Hazard Macro F1**: **0.5214** (Weighted F1: 0.8230)
- **Clinical Entity Extraction (NER Relaxed F1)**: **0.7670** (Precision: 0.7191, Recall: 0.8797)
- **Clinical Entity Extraction (NER Exact F1)**: **0.6437** (Precision: 0.6033, Recall: 0.7388)

---

## 2. Primary Classification Task: Triage Urgency Level

The primary clinical NLP task maps unstructured clinical narratives into four triage risk categories: `CRITICAL`, `HIGH`, `LOW`, and `MEDIUM`.

### 2.1 Overall Performance Metrics
| Metric | Observed Value | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **0.8559** (85.59%) | Proportion of correctly categorized narratives |
| **Macro Precision** | **0.7493** (74.93%) | Unweighted mean precision across classes |
| **Macro Recall** | **0.7637** (76.37%) | Unweighted mean sensitivity across classes |
| **Macro F1-Score** | **0.7557** (75.57%) | Primary performance metric accounting for imbalance |
| **Weighted F1-Score** | **0.8605** (86.05%) | Prevalence-weighted harmonic mean |
| **Critical Class Recall**| **0.9457** (94.57%) | Safety-critical sensitivity (87/92 true CRITICAL cases caught) |

### 2.2 Granular Class-Wise Performance Breakdown
| Class | Precision | Recall | F1-Score | Support ($N$) | Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CRITICAL** | **0.9667** | **0.9457** | **0.9560** | 92 | 10.12% |
| **HIGH** | **0.6408** | **0.7109** | **0.6741** | 128 | 14.08% |
| **LOW** | **0.9654** | **0.9316** | **0.9482** | 599 | 65.90% |
| **MEDIUM** | **0.4242** | **0.4667** | **0.4444** | 90 | 9.90% |
| **Macro Average** | **0.7493** | **0.7637** | **0.7557** | 909 | 100.0% |
| **Weighted Average**| **0.8662** | **0.8559** | **0.8605** | 909 | 100.0% |

### 2.3 Confusion Matrix Analysis
```
True \ Pred      CRITICAL    HIGH     LOW    MEDIUM    Total
CRITICAL               87       1       0         4       92
HIGH                    0      91      12        25      128
LOW                     1      12     558        28      599
MEDIUM                  2      38       8        42       90
Total                  90     142     578        99      909
```
- **False Negative Safety Audit**: Only 1 true `CRITICAL` document was predicted as `HIGH`, 4 as `MEDIUM`, and **0 as `LOW`**. No critical deteriorating patient was classified as routine `LOW` risk.
- **Primary Bottleneck**: The model exhibits significant confusion between adjacent intermediate tiers (`MEDIUM` vs `HIGH`), with 38 true `MEDIUM` records misclassified as `HIGH` (precision of `MEDIUM`: 42.42%).

---

## 3. Secondary Classification Task: Toxicity Hazard Type

The secondary NLP task identifies organ-specific adverse event hazards across 8 organ/system classes.

### 3.1 Overall Performance Metrics
| Metric | Observed Value | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **0.7756** (77.56%) | Dominated by majority `NONE` class (78.1%) |
| **Macro Precision** | **0.4716** (47.16%) | Impacted by low precision on rare classes |
| **Macro Recall** | **0.7153** (71.53%) | Cost-sensitive class weighting boosts minority recall |
| **Macro F1-Score** | **0.5214** (52.14%) | Primary balance metric |
| **Weighted F1-Score** | **0.8230** (82.30%) | High due to correct identification of `NONE` |

### 3.2 Granular Class-Wise Performance Breakdown
| Toxicity Class | Precision | Recall | F1-Score | Support ($N$) | Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CARDIAC** | 0.4000 | 0.8000 | 0.5333 | 5 | 0.55% |
| **DERMATOLOGIC** | 0.0638 | 0.7500 | 0.1176 | 4 | 0.44% |
| **HEMATOLOGIC** | 0.2679 | 0.6522 | 0.3797 | 23 | 2.53% |
| **HEPATIC** | 0.7033 | 0.8421 | 0.7665 | 76 | 8.36% |
| **NEUROPATHIC** | 0.2222 | 0.5000 | 0.3077 | 12 | 1.32% |
| **NONE** | 0.9857 | 0.7746 | 0.8675 | 710 | 78.11% |
| **PULMONARY** | 0.9630 | 0.9455 | 0.9541 | 55 | 6.05% |
| **RENAL** | 0.1667 | 0.4583 | 0.2444 | 24 | 2.64% |

- **Strong Performers**: `PULMONARY` toxicity achieved exceptional metrics (F1: 0.9541, Recall: 94.55%), driven by distinctive terms (`dyspnea`, `pneumonitis`, `cough`). `HEPATIC` also performed reliably (F1: 0.7665, Recall: 84.21%).
- **Severe Imbalance Challenges**: Ultra-rare toxicities (`DERMATOLOGIC` with $N=4$, `CARDIAC` with $N=5$) achieve high recall (>75%) due to class-balanced loss weighting, but at the expense of high false positive rates from unspecific symptom overlaps.

---

## 4. Clinical Named Entity Extraction (NER) Evaluation

Evaluated against ground truth character spans in `ner_entities` across 909 documents:

| Entity Type | Predicted Mentions | Gold Mentions | Relaxed Precision | Relaxed Recall | Relaxed F1 | Exact Span F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GENE_MUTATION** | 684 | 851 | **1.0000** | **0.8038** | **0.8912** | **0.8912** |
| **DRUG_NAME** | 682 | 763 | **0.9238** | **0.8257** | **0.8719** | **0.7107** |
| **DOSAGE** | 1,911 | 764 | **0.4000** | **1.0000** | **0.5714** | **0.5714** |
| **ADVERSE_EVENT** | 1,450 | 910 | **0.5931** | **0.9451** | **0.7291** | **0.3746** |
| **Overall Micro Average** | 4,727 | 3,288 | **0.6232** | **0.8959** | **0.7350** | **0.6019** |
| **Overall Macro Average** | — | — | **0.7191** | **0.8797** | **0.7670** | **0.6437** |

### Key Findings
1. **Mutation Specificity**: Perfect precision (1.0000) for genomic alterations (`EGFR`, `KRAS`, `TP53`, `T790M`).
2. **Dosage Over-Extraction**: Recall is 100%, but precision is 40.0% because lab values (`mmHg`, `mg/dL`) matching regex patterns are extracted even when not explicitly antineoplastic drug dosages.
3. **Exact vs. Relaxed Span Gap**: Adverse events show a significant drop from relaxed F1 (0.7291) to exact F1 (0.3746), reflecting modifier boundaries (e.g. `nausea` vs `severe nausea`).

---

## 5. Structured Feature Matrix Diagnostics (Validation Set)
- **Matrix Dimension**: $909 \times 1,012$ dimensions (1,000 TF-IDF features + 12 structured count indicators).
- **Missing / Infinite Values**: Strictly **0 NaN** and **0 Inf** values.
- **Zero-Variance Columns**: 51 / 1,012 features (5.04%) exhibited zero variance on validation narratives (rare vocabulary unigrams not appearing in validation).
- **Word Count**: Mean $105.74 \pm 22.48$ words.
- **Total Concepts Extracted**: Mean $5.19 \pm 2.74$ concepts per document.

---

## 6. Validation Conclusion & Verdict
The independent evaluation confirms that the NLP Engineer's reported validation metrics are **100% accurate, fully reproducible, and mathematically verified**.
The system establishes a robust, safety-conscious baseline for Stage 3, demonstrating high critical sensitivity (94.57%) and solid macro discrimination (0.7557 F1).
