# Clinical Named Entity Recognition (NER) Comparison

## 1. Evaluation Methodology

Entity extraction is evaluated on the 909 VALIDATION documents using **maximum one-to-one bipartite matching** to strictly prevent double-counting of overlapping predictions.

Two matching protocols are reported:
1. **Exact Span Match**: Requires exact match on character boundaries (`start == gold_start` AND `end == gold_end`) AND matching entity label (`label == gold_label`).
2. **Relaxed Overlap Match**: Requires character span intersection ($\max(\text{start}, \text{gold\_start}) < \min(\text{end}, \text{gold\_end})$) AND matching entity label.

---

## 2. Global NER Performance Summary

| Model / Strategy | Match Type | Precision | Recall | F1 Score | True Positives | Total Predicted | Total Gold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A (Regex Rules)** | Exact | 0.5186 | 0.7136 | **0.6008** | 2,332 | 4,497 | 3,268 |
| **Baseline A (Regex Rules)** | Relaxed | 0.6258 | 0.8611 | **0.7251** | 2,814 | 4,497 | 3,268 |
| **MiniLM Token Head** | Exact | 0.3953 | 0.4752 | **0.4316** | 1,553 | 3,929 | 3,268 |
| **MiniLM Token Head** | Relaxed | 0.7562 | 0.9091 | **0.8256** | 2,971 | 3,929 | 3,268 |
| **Train Span Lexicon** | Exact | 0.6238 | 0.8473 | **0.7186** | 2,769 | 4,439 | 3,268 |
| **Train Span Lexicon** | Relaxed | 0.6238 | 0.8473 | **0.7186** | 2,769 | 4,439 | 3,268 |

---

## 3. Entity Type Breakdown (Exact Span Match)

### A. GENE_MUTATION (Gold Support = 838)
| Model | Precision | Recall | Exact F1 | True Positives | Predicted |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline A** | 1.0000 | 0.9021 | **0.9486** | 756 | 756 |
| **MiniLM Token Head** | 0.6768 | 0.4773 | **0.5598** | 400 | 591 |
| **Train Span Lexicon** | 1.0000 | 0.9021 | **0.9486** | 756 | 756 |

*Analysis*: Mutation formats follow highly structured nomenclatures (e.g., `EGFR T790M`, `KRAS G12C`, `BRAF V600E`). Exact string and regex matching achieve 100% precision. Subword token classification splits mutations across multiple WordPieces, causing boundary fragmentation under exact matching.

### B. DRUG_NAME (Gold Support = 759)
| Model | Precision | Recall | Exact F1 | True Positives | Predicted |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline A** | 0.5861 | 1.0000 | **0.7390** | 759 | 1,295 |
| **MiniLM Token Head** | 0.7509 | 0.8261 | **0.7867** | 627 | 835 |
| **Train Span Lexicon** | 0.5861 | 1.0000 | **0.7390** | 759 | 1,295 |

*Analysis*: MiniLM significantly improves Drug Name precision (+16.5 pts from 58.6% to 75.1%) by leveraging surrounding syntax to avoid false positive drug classifications.

### C. DOSAGE (Gold Support = 759)
| Model | Precision | Recall | Exact F1 | True Positives | Predicted |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline A** | 0.3916 | 0.4545 | **0.4207** | 345 | 881 |
| **MiniLM Token Head** | 0.2291 | 0.3399 | **0.2737** | 258 | 1,126 |
| **Train Span Lexicon** | 0.3916 | 0.4545 | **0.4207** | 345 | 881 |

*Analysis*: Dosage extraction remains challenging for token classification due to variable token splits on numeric values and units (e.g. `mg/m2`, `mcg/kg/min`). MiniLM Relaxed F1 reaches 0.7421, showing that mentions are recognized but character boundaries vary.

### D. ADVERSE_EVENT (Gold Support = 912)
| Model | Precision | Recall | Exact F1 | True Positives | Predicted |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline A** | 0.2974 | 0.5055 | **0.3746** | 461 | 1,550 |
| **MiniLM Token Head** | 0.1946 | 0.2939 | **0.2342** | 268 | 1,377 |
| **Train Span Lexicon** | 0.6032 | 0.9967 | **0.7516** | 909 | 1,507 |

*Analysis*: Ground truth adverse events contain full descriptive noun phrases (e.g., *"moderate fatigue and mild nausea"*). The Train Span Lexicon dramatically improves Adverse Event F1 to **0.7516** (+37.7 pts over baseline) because it memorizes clinical multi-word spans seen in training notes without regex truncation.

---

## 4. Key Recommendations

1. **Hybrid NER Assembly**: The optimal clinical extraction engine should combine the **Train Span Lexicon** (for multi-word adverse events and mutations) with **contextual MiniLM predictions** (for drug and dosage boundary filtering).
2. **Evaluation Awareness**: The gap between Exact F1 (0.4316) and Relaxed F1 (0.8256) demonstrates that contextual token classification correctly captures entity semantics, but subword detokenization requires morphological rule smoothing.
