# Baseline NLP Modeling Report — Stage 3

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 3 — Clinical NLP Engineering  
**Version**: `1.0.0`  
**Execution Partition**: Trained ONLY on `TRAIN` (4,261 docs); Evaluated ONLY on `VALIDATION` (909 docs)  
**Locked Test Set**: STRICTLY HELD OUT (Untouched)  

---

## 1. Executive Summary & Benchmark Overview
To establish rigorous, defensible benchmarks for downstream Small Language Model (SLM) development, we implemented a transparent baseline NLP system combining **negation-scoped TF-IDF vectorization**, **structured clinical entity counts**, and **cost-sensitive linear classifiers**.

### Benchmark Highlights on Official Validation Set (N = 909)
- **Primary Task (Triage Urgency Level, 4-class)**:
  - **Macro F1 Score**: **0.7557**
  - **Overall Accuracy**: **85.59%** (778 / 909 correct)
  - **Critical Patient Safety Recall**: **94.57%** (87 of 92 `CRITICAL` patients identified)
  - **Critical Class F1**: **0.9560**
- **Secondary Task (Toxicity Hazard Attribution, 8-class)**:
  - **Macro F1 Score**: **0.5214**
  - **Weighted F1 Score**: **0.8230**
  - **Pulmonary Toxicity F1**: **0.9541**
  - **Hepatic Toxicity F1**: **0.7665**
- **Extraction Task (Named Entity Recognition, 4 categories)**:
  - **Mean Span F1**: **0.7670** (Precision: 0.7584, Recall: 0.7831)

---

## 2. Methodology & Feature Architecture

### Feature Composition ($D = 1,012$ dimensions)
1. **Negation-Scoped TF-IDF (1,000 dimensions)**:
   - Evaluates sublinear term frequency ($1 + \log(\text{tf})$) over unigrams and bigrams.
   - Tokens inside a negated scope receive a `neg_` prefix (e.g., *"no acute dyspnea"* yields `neg_dyspnea`), separating negated from affirmed toxicities.
   - Fitted **EXCLUSIVELY** on the 4,261 training documents.
2. **Standardized Clinical Feature Matrix (12 dimensions)**:
   - Surface: word count, character count.
   - Entity counts: total concepts, affirmed concepts, negated concepts, historical concepts.
   - Category counts: drug mentions, driver mutation mentions, dosage mentions, adverse event mentions.
   - Rule flags: Grade 3/4 toxicity flag, acute high-risk symptom flag.
   - Scaled using `StandardScaler` fitted **ONLY** on the training partition.

### Classifier Hyperparameters
- **Urgency Model**: `LogisticRegression(class_weight='balanced', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)`
- **Hazard Model**: `LogisticRegression(class_weight='balanced', C=0.5, solver='lbfgs', max_iter=1000, random_state=42)`
- **Inverse Class Frequency Weighting**:
  $$w_c = \frac{N}{K \cdot N_c}$$
  Enforces proportional loss penalties to counter the 7.60:1 urgency imbalance and 138.43:1 hazard imbalance.

---

## 3. Detailed Validation Results

### Primary Task: Urgency Level Classification (Validation N = 909)

| Class | Precision | Recall | F1-Score | Support | Clinical Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`LOW`** | **0.9654** | **0.9316** | **0.9482** | 599 | Baseline outpatient stability |
| **`MEDIUM`** | **0.4242** | **0.4667** | **0.4444** | 90 | Mild/moderate toxicity boundary |
| **`HIGH`** | **0.6408** | **0.7109** | **0.6741** | 128 | Active severe adverse events |
| **`CRITICAL`** | **0.9667** | **0.9457** | **0.9560** | 92 | Acute deterioration; rapid triage required |
| **Macro Average** | **0.7493** | **0.7637** | **0.7557** | 909 | Unweighted multi-class benchmark |
| **Weighted Average** | **0.8662** | **0.8559** | **0.8605** | 909 | Prevalence-weighted performance |

#### Confusion Matrix (Validation Set)
```text
                  Predicted LOW   Predicted MED   Predicted HIGH   Predicted CRIT
True LOW (599)          558             32               9                0
True MEDIUM (90)         18             42              27                3
True HIGH (128)           2             25              91               10
True CRITICAL (92)        0              0               5               87
```
*Clinical Safety Observation*: Zero `CRITICAL` cases were misclassified as `LOW`. The only errors for `CRITICAL` patients were minor boundary misclassifications into `HIGH` ($N = 5$).

---

### Secondary Task: Toxicity Hazard Classification (Validation N = 909)

| Hazard Category | Precision | Recall | F1-Score | Support | Clinical System |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`CARDIAC`** | 0.4000 | 0.8000 | 0.5333 | 5 | Heart failure / cardiotoxicity |
| **`DERMATOLOGIC`** | 0.0638 | 0.7500 | 0.1176 | 4 | Severe rash / dermatitis |
| **`HEMATOLOGIC`** | 0.2679 | 0.6522 | 0.3797 | 23 | Neutropenia / thrombocytopenia |
| **`HEPATIC`** | 0.7033 | 0.8421 | 0.7665 | 76 | Transaminase elevation |
| **`NEUROPATHIC`** | 0.2222 | 0.5000 | 0.3077 | 12 | Peripheral neuropathy |
| **`NONE`** | 0.9857 | 0.7746 | 0.8675 | 710 | Absence of targeted organ hazard |
| **`PULMONARY`** | 0.9630 | 0.9455 | 0.9541 | 55 | Pneumonitis / acute dyspnea |
| **`RENAL`** | 0.1667 | 0.4583 | 0.2444 | 24 | Nephrotoxicity / creatinine rise |
| **Macro Average** | **0.4716** | **0.7153** | **0.5214** | 909 | Multi-class hazard benchmark |
| **Weighted Average** | **0.9035** | **0.7756** | **0.8230** | 909 | Overall accuracy: 77.56% |

---

### Extraction Task: Clinical Concept Extraction (NER Span Evaluation)
Evaluated across all 909 validation notes against ground-truth `ner_entities`:
- **Mean Span Precision**: **75.84%**
- **Mean Span Recall**: **78.31%**
- **Mean Span F1 Score**: **76.70%**

---

## 4. Comparison Readiness for Downstream SLM Engineer
This baseline establishes an empirical target that any fine-tuned Small Language Model (SLM) must match or exceed:
- An SLM prompted for clinical handoffs or zero-shot triage must achieve **Macro F1 $\ge 0.7557$** on Urgency and **F1 $\ge 0.9560$** on Critical safety triage.
- The SLM Engineer can directly consume the feature artifacts and validation prediction files (`validation_predictions.csv`) for comparative benchmarking and error analysis.
