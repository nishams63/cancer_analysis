# Stage 3 NLP Model Comparison — Development Benchmark

## 1. Executive Summary

This report documents the empirical evaluation of all candidate NLP models on the official Stage 3 clinical development set ($N=4,261$ TRAIN, $N=909$ VALIDATION). All evaluations strictly adhere to the frozen development protocol: model selection is conducted exclusively on VALIDATION data, while the LOCKED TEST set ($N=928$) remains sealed for independent evaluation.

The goal is to determine whether contextual Transformer encoders or hybrid representations outperform the frozen baseline (**Baseline A: TF-IDF + Regex Rules + Logistic Regression**).

---

## 2. Multi-Task Performance Matrix

| Model Candidate | Model Type / Architecture | Status | Urgency Accuracy | Urgency Macro F1 | Urgency Critical Recall | Hazard Accuracy | Hazard Macro F1 | NER Exact F1 | NER Relaxed F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A (Control)** | TF-IDF (1,000) + 12 Regex Counts + LR | **EVALUATED** | 85.59% | 0.7557 | 94.57% | 77.56% | 0.5214 | 0.6008 | 0.7251 |
| **MiniLM** | `all-MiniLM-L6-v2` Frozen + Linear Heads | **EVALUATED** | 92.63% | 0.8499 | 94.57% | 96.92% | 0.8872 | 0.4316 | 0.8256 |
| **MiniLM Hybrid** | MiniLM Embeddings + Scaled Structured Counts + LR | **EVALUATED** | **94.50%** | **0.8815** | **100.00%** | **98.79%** | **0.9551** | 0.4316 | 0.8256 |
| **Train Span Lexicon** | Annotated Phrase Matching + Frozen Classifiers | **EVALUATED** | 85.59% | 0.7557 | 94.57% | 77.56% | 0.5214 | **0.7186** | 0.7186 |
| **DistilBERT** | `distilbert-base-uncased` | **NOT RUN** | NA | NA | NA | NA | NA | NA | NA |
| **BioBERT** | `dmis-lab/biobert-base-cased-v1.2` | **NOT RUN** | NA | NA | NA | NA | NA | NA | NA |
| **ClinicalBERT** | `emilyalsentzer/Bio_ClinicalBERT` | **NOT RUN** | NA | NA | NA | NA | NA | NA | NA |
| **PubMedBERT** | `microsoft/BiomedNLP-BiomedBERT-base-uncased` | **NOT RUN** | NA | NA | NA | NA | NA | NA | NA |
| **SLM (Qwen 0.5B)**| `Qwen/Qwen2.5-0.5B-Instruct` | **NOT COMPLETED** | NA | NA | NA | NA | NA | NA | NA |

---

## 3. Detailed Status of Candidate Models

### 1. Baseline A (Frozen Control)
- **Architecture**: 1,000 unigram/bigram TF-IDF features with sublinear term frequency scaling, combined with 12 rule/regex-extracted clinical concept counts, classified via balanced `LogisticRegression` ($C=1.0$).
- **Role**: Serves as the immutable control standard.

### 2. MiniLM (`sentence-transformers/all-MiniLM-L6-v2`)
- **Architecture**: Frozen 6-layer 384-dimensional contextual encoder (22.7M parameters). Mean-pooled document embeddings fed into balanced Logistic Regression classifiers for Urgency and Hazard. Token-level subword embeddings fed into a supervised linear BIO classification head.
- **Key Findings**: Achieves substantial gains in document classification (Urgency Macro F1 +9.4 pts, Hazard Macro F1 +36.6 pts, Relaxed NER F1 +10.0 pts), but token-level exact boundary alignment underperforms dictionary matching (Exact F1 0.4316 vs 0.6008).

### 3. MiniLM Hybrid (Contextual + Structured Features)
- **Architecture**: Concatenation of 384-dim contextual document embeddings and 12 z-score standardized clinical concept counts (396 dimensions total).
- **Key Findings**: Highest performing model overall. Reaches **94.50% Urgency Accuracy**, **0.8815 Urgency Macro F1** (+12.6 pts over baseline), and **100% Critical Recall** (92 out of 92 validation critical cases identified). Resolves rare hazard categories to achieve **0.9551 Hazard Macro F1** (+43.4 pts over baseline).

### 4. Train Span Lexicon (Exact Phrase Extraction)
- **Architecture**: Exact substring matching derived strictly from the 4,261 training notes (`1,127` unique annotated clinical entities).
- **Key Findings**: Delivers superior exact span boundary detection, achieving **0.7186 Exact Span F1** (an +11.8 pt improvement over baseline 0.6008).

### 5. External Pretrained Encoders (DistilBERT, BioBERT, ClinicalBERT, PubMedBERT)
- **Status**: `NOT RUN`.
- **Root Cause**: Hugging Face Hub downloads were inaccessible in the local environment due to blocked network access / DNS timeouts, and checkpoints were not present in local cache.

### 6. Small Language Model (SLM: Qwen2.5-0.5B-Instruct)
- **Status**: `NOT COMPLETED`.
- **Root Cause**: Checkpoint was cached locally and encoding started on the training set. However, concurrent PyTorch execution reduced available host system RAM to `0.64 GB`. The process was cleanly halted to prevent system freezing and memory thrashing. Per benchmarking protocol rules, no score is fabricated.

---

## 4. Summary Findings

1. **Document-Level Classification**: Dense contextual embeddings (`all-MiniLM-L6-v2`) combined with structured clinical concept counts (**MiniLM Hybrid**) overwhelmingly outperform the bag-of-words baseline.
2. **Clinical Safety Preservation**: MiniLM Hybrid achieves **100.0% Critical Case Recall**, completely eliminating false negative triage events on the validation cohort.
3. **Span Boundary Extraction**: Dictionary/lexicon extraction strictly derived from TRAIN annotations achieves the highest exact entity boundary F1 (`0.7186`), whereas subword token classifiers without full fine-tuning suffer boundary fragmentation (`0.4316`).
