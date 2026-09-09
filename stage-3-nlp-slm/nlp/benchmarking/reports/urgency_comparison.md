# Clinical Triage Urgency Classification Comparison

## 1. Executive Summary

Clinical triage classification assigns oncology encounter notes into one of four urgency levels: `CRITICAL`, `HIGH`, `MEDIUM`, and `LOW`. Evaluation on the 909 VALIDATION documents demonstrates dramatic improvements when transitioning from the bag-of-words baseline to dense contextual embeddings and hybrid feature representations.

---

## 2. Model Performance Summary

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Critical Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A (Control)** | 85.59% | 0.7451 | 0.7712 | 0.7557 | 0.8576 | 94.57% (87/92) |
| **MiniLM** | 92.63% | 0.8521 | 0.8503 | 0.8499 | 0.9262 | 94.57% (87/92) |
| **MiniLM Hybrid** | **94.50%** | **0.8873** | **0.8813** | **0.8815** | **0.9437** | **100.00% (92/92)** |

---

## 3. Per-Class Performance Breakdown

### A. CRITICAL Triage (Support = 92)
| Model | Precision | Recall | F1 Score | Missed Critical Cases |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline A** | 96.67% | 94.57% | 0.9560 | 5 cases (5.4% false negative rate) |
| **MiniLM** | 97.75% | 94.57% | 0.9613 | 5 cases (5.4% false negative rate) |
| **MiniLM Hybrid** | **97.87%** | **100.00%** | **0.9892** | **0 cases (0.0% false negative rate)** |

*Clinical Impact*: MiniLM Hybrid achieves **100% Critical Recall**, correctly detecting all 92 acute clinical emergencies (severe dyspnea, acute nephrotoxicity, grade 4 toxicities) on validation notes.

### B. HIGH Triage (Support = 128)
| Model | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| **Baseline A** | 63.87% | 77.34% | 0.6996 |
| **MiniLM** | 75.86% | 85.94% | 0.8059 |
| **MiniLM Hybrid** | **79.31%** | **89.84%** | **0.8425** |

### C. MEDIUM Triage (Support = 90)
| Model | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| **Baseline A** | 41.56% | 35.56% | 0.3832 |
| **MiniLM** | 67.90% | 61.11% | 0.6433 |
| **MiniLM Hybrid** | **78.08%** | **63.33%** | **0.6994** |

*Analysis*: Separating `MEDIUM` from `HIGH` urgency was identified by the independent evaluation as a primary weakness of Baseline A (38.3% F1). MiniLM Hybrid nearly doubles MEDIUM F1 to **69.9%** by capturing nuanced phrasing of moderate symptom severity.

### D. LOW Triage (Support = 599)
| Model | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| **Baseline A** | 96.01% | 96.49% | 0.9625 |
| **MiniLM** | 99.33% | 98.50% | 0.9891 |
| **MiniLM Hybrid** | **99.66%** | **99.33%** | **0.9950** |

---

## 4. Confusion Matrices

### Baseline A (Control)
```
Pred ->     CRITICAL  HIGH  LOW  MEDIUM
True:
CRITICAL       87        3    0       2
HIGH            2       99    4      23
LOW             0        5  578      16
MEDIUM          1       48    9      32
```

### MiniLM Hybrid (Contextual + Concept Features)
```
Pred ->     CRITICAL  HIGH  LOW  MEDIUM
True:
CRITICAL       92        0    0       0   <-- 100% Critical Recall!
HIGH            0      115    1      12
LOW             0        0  595       4
MEDIUM          2       30    1      57   <-- Greatly improved MEDIUM recovery!
```
