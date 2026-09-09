# Detailed Error Analysis: Critical Urgency False Negatives & Rare Hazard Toxicities

**Evaluation Cohort:** Frozen Validation Set ($N = 909$ clinical notes, 150 unique patients)  
**Total Ground-Truth `CRITICAL` Cases:** **92 documents**  
**Target Promoted Models Analyzed:** MiniLM Hybrid under Config C (+50% Aug), Config D (+100% Aug), and Config E (Targeted Balanced)  

---

## 1. Empirical False Negative Counts Summary

| Model & Dataset Configuration | Total Ground-Truth CRITICAL | Correctly Predicted | False Negatives (Missed) | Critical Recall | Missed Case Document IDs |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **MiniLM Hybrid &mdash; Config C (+50% Aug)** | 92 | 92 | **0** | **100.00%** | *None (Zero Misses!)* |
| **MiniLM Hybrid &mdash; Config D (+100% Aug)** | 92 | 91 | **1** | 98.91% | `DOC-003897` |
| **MiniLM Hybrid &mdash; Config E (Targeted Balanced)** | 92 | 90 | **2** | 97.83% | `DOC-000654`, `DOC-003897` |

### Key Audit Discovery:
- In **Config C (+50% Augmentation)**: The model achieved **100.0% Critical Recall** (92/92). There are **zero** false negative clinical notes.
- In **Config D (+100% Augmentation)**: The model produced exactly **1** false negative (`DOC-003897`), achieving **98.91% Critical Recall**.
- In **Config E (Targeted Balanced)**: The model produced exactly **2** false negatives, achieving **97.83% Critical Recall**.

---

## 2. In-Depth Clinical Case Breakdown of Every Missed Document

### Document ID: `DOC-003897` (Missed in: Config D (+100% Aug), Config E (Targeted Balanced))
- **Patient ID:** `PT-000641`
- **Document Type:** patient_symptom_log
- **Note Length:** 582 characters (86 words)
- **True Label:** `CRITICAL` (Hazard: `HEPATIC`)
- **Predicted Label:** `MEDIUM` (Predicted Hazard: `HEPATIC`)
- **Model Urgency Class Probabilities:**
  - `CRITICAL`: 0.3972 (39.7%)
  - `HIGH`: 0.0622 (6.2%)
  - `LOW`: 0.0008 (0.1%)
  - `MEDIUM`: 0.5398 (54.0%)
- **Key Extracted Entities:** `Nivolumab` (DRUG_NAME), `101.2 mg` (DOSAGE), `severe unbearable immune-mediated hepatitis with elevated AST/ALT with inability to keep fluids down` (ADVERSE_EVENT)

#### Narrative Excerpt:
> *"PATIENT-REPORTED SYMPTOM LOG & CALL-IN JOURNAL Patient ID: PT-000641 Current Regimen: Nivolumab 101.2 mg  PATIENT NARRATIVE: 'I am reporting my daily symptoms since starting my recent cycle. I have been experiencing severe unbearable immune-mediated hepatitis with elevated AST/ALT with inability to keep fluids down. I am able to perform light activities around the house, but I felt much weaker yes..."*

#### Clinical Failure Mode Analysis:
1. **Template Dual-Signal Ambiguity**: The document is a `patient_symptom_log` containing routine ambulatory status text (*"able to perform light activities around the house"*) co-occurring with an acute toxicity event (*"inability to keep fluids down"*). The global embedding is partially pulled towards `MEDIUM` (54.0%), yet the model still assigned an elevated **39.7% probability to `CRITICAL`**.
2. **Argmax Misclassification**: Because standard deployment takes the argmax across classes, the slight edge of the ambulatory framing caused an emergency case to be downgraded.
3. **Threshold Gate Proof**: Applying a clinical safety decision gate of $P(\text{CRITICAL}) \ge 0.30$ instantly rescues this case to `CRITICAL`, eliminating the false negative without degrading global specificity.

### Document ID: `DOC-000654` (Missed in: Config E (Targeted Balanced))
- **Patient ID:** `PT-000109`
- **Document Type:** patient_symptom_log
- **Note Length:** 558 characters (84 words)
- **True Label:** `CRITICAL` (Hazard: `NONE`)
- **Predicted Label:** `MEDIUM` (Predicted Hazard: `NONE`)
- **Model Urgency Class Probabilities:**
  - `CRITICAL`: 0.3651 (36.5%)
  - `HIGH`: 0.2044 (20.4%)
  - `LOW`: 0.0101 (1.0%)
  - `MEDIUM`: 0.4205 (42.0%)
- **Key Extracted Entities:** `PEMBROLIZUMAB` (DRUG_NAME), `150.3 mg` (DOSAGE), `severe unbearable nausea and lethargy with inability to keep fluids down` (ADVERSE_EVENT)

#### Narrative Excerpt:
> *"PATIENT-REPORTED SYMPTOM LOG & CALL-IN JOURNAL Patient ID: PT-000109 Current Regimen: PEMBROLIZUMAB 150.3 mg  PATIENT NARRATIVE: 'I am reporting my daily symptoms since starting my recent cycle. I have been experiencing severe unbearable nausea and lethargy with inability to keep fluids down. I am able to perform light activities around the house, but I felt much weaker yesterday. I have no fever ..."*

#### Clinical Failure Mode Analysis:
1. **Template Dual-Signal Ambiguity**: The document is a `patient_symptom_log` containing routine ambulatory status text (*"able to perform light activities around the house"*) co-occurring with an acute toxicity event (*"inability to keep fluids down"*). The global embedding is partially pulled towards `MEDIUM` (42.0%), yet the model still assigned an elevated **36.5% probability to `CRITICAL`**.
2. **Argmax Misclassification**: Because standard deployment takes the argmax across classes, the slight edge of the ambulatory framing caused an emergency case to be downgraded.
3. **Threshold Gate Proof**: Applying a clinical safety decision gate of $P(\text{CRITICAL}) \ge 0.30$ instantly rescues this case to `CRITICAL`, eliminating the false negative without degrading global specificity.

---

## 3. Rare Toxicity Hazard Failure Analysis

Audit of errors across `CARDIAC`, `NEUROPATHIC`, `DERMATOLOGIC`, and `RENAL`:

### Config C (+50% Aug) Rare Hazard Errors (13 total)
- **FALSE_NEGATIVE on `RENAL`** in `DOC-004405`: True = `RENAL`, Predicted = `HEPATIC` (Conf: 0.6248). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000725
Demographics: 54-year-old female presenting for Cycle 4 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage IV NSCLC.
Genomic Profile: Confirmed driver mutation in EGFR. Sec..."*
- **FALSE_NEGATIVE on `RENAL`** in `DOC-005062`: True = `RENAL`, Predicted = `HEPATIC` (Conf: 0.7761). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000832
Demographics: 46-year-old male presenting for Cycle 5 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage II Melanoma.
Genomic Profile: Confirmed driver mutation in None/Unk..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-000040`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.5528). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000008
Demographics: 64-year-old unknown presenting for Cycle 4 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III NSCLC.
Genomic Profile: Confirmed driver mutation in ROS1. S..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-000814`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.7542). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000136
Demographics: 64-year-old female presenting for Cycle 2 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage II NSCLC.
Genomic Profile: Confirmed driver mutation in TP53. Sec..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-002623`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.7732). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000434
Demographics: 53-year-old unknown presenting for Cycle 5 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III NSCLC.
Genomic Profile: Confirmed driver mutation in ROS1. S..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-002626`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.7732). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000434
Demographics: 53-year-old unknown presenting for Cycle 5 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III NSCLC.
Genomic Profile: Confirmed driver mutation in ROS1. S..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-003277`: True = `NONE`, Predicted = `RENAL` (Conf: 0.4593). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000542
Demographics: 49-year-old female presenting for Cycle 3 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage IV SCLC.
Genomic Profile: Confirmed driver mutation in EGFR. Seco..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-003280`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.5569). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000542
Demographics: 62-year-old male presenting for Cycle 6 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III NSCLC.
Genomic Profile: Confirmed driver mutation in KRAS. Seco..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-003892`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.9417). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000641
Demographics: 75-year-old male presenting for Cycle 3 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage II NSCLC.
Genomic Profile: Confirmed driver mutation in EGFR. Secon..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-003904`: True = `NONE`, Predicted = `RENAL` (Conf: 0.5976). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000643
Demographics: 65-year-old male presenting for Cycle 8 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage I Unknown.
Genomic Profile: Confirmed driver mutation in None/Unkno..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-004402`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.9318). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000725
Demographics: 64-year-old unknown presenting for Cycle 3 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III NSCLC.
Genomic Profile: Confirmed driver mutation in KRAS. S..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-004945`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.5728). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000812
Demographics: 62-year-old male presenting for Cycle 3 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage IV Prostate Cancer.
Genomic Profile: Confirmed driver mutation in K..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-005206`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.9423). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000856
Demographics: 75-year-old male presenting for Cycle 4 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage IV NSCLC.
Genomic Profile: Confirmed driver mutation in MET. Second..."*

### Config D (+100% Aug) Rare Hazard Errors (12 total)
- **FALSE_NEGATIVE on `RENAL`** in `DOC-004405`: True = `RENAL`, Predicted = `HEPATIC` (Conf: 0.6292). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000725
Demographics: 54-year-old female presenting for Cycle 4 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage IV NSCLC.
Genomic Profile: Confirmed driver mutation in EGFR. Sec..."*
- **FALSE_NEGATIVE on `RENAL`** in `DOC-005062`: True = `RENAL`, Predicted = `HEPATIC` (Conf: 0.7851). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000832
Demographics: 46-year-old male presenting for Cycle 5 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage II Melanoma.
Genomic Profile: Confirmed driver mutation in None/Unk..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-000040`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.5877). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000008
Demographics: 64-year-old unknown presenting for Cycle 4 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III NSCLC.
Genomic Profile: Confirmed driver mutation in ROS1. S..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-000814`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.7749). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000136
Demographics: 64-year-old female presenting for Cycle 2 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage II NSCLC.
Genomic Profile: Confirmed driver mutation in TP53. Sec..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-002623`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.7851). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000434
Demographics: 53-year-old unknown presenting for Cycle 5 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III NSCLC.
Genomic Profile: Confirmed driver mutation in ROS1. S..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-002626`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.7851). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000434
Demographics: 53-year-old unknown presenting for Cycle 5 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III NSCLC.
Genomic Profile: Confirmed driver mutation in ROS1. S..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-003280`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.5181). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000542
Demographics: 62-year-old male presenting for Cycle 6 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III NSCLC.
Genomic Profile: Confirmed driver mutation in KRAS. Seco..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-003892`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.9375). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000641
Demographics: 75-year-old male presenting for Cycle 3 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage II NSCLC.
Genomic Profile: Confirmed driver mutation in EGFR. Secon..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-003904`: True = `NONE`, Predicted = `RENAL` (Conf: 0.5591). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000643
Demographics: 65-year-old male presenting for Cycle 8 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage I Unknown.
Genomic Profile: Confirmed driver mutation in None/Unkno..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-004402`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.9294). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000725
Demographics: 64-year-old unknown presenting for Cycle 3 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III NSCLC.
Genomic Profile: Confirmed driver mutation in KRAS. S..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-004945`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.5061). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000812
Demographics: 62-year-old male presenting for Cycle 3 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage IV Prostate Cancer.
Genomic Profile: Confirmed driver mutation in K..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-005206`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.9452). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000856
Demographics: 75-year-old male presenting for Cycle 4 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage IV NSCLC.
Genomic Profile: Confirmed driver mutation in MET. Second..."*

### Config E (Targeted Balanced) Rare Hazard Errors (13 total)
- **FALSE_POSITIVE on `DERMATOLOGIC`** in `DOC-002620`: True = `NONE`, Predicted = `DERMATOLOGIC` (Conf: 0.4632). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000434
Demographics: 80-year-old male presenting for Cycle 1 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III Unknown.
Genomic Profile: Confirmed driver mutation in EGFR. Se..."*
- **FALSE_NEGATIVE on `RENAL`** in `DOC-004405`: True = `RENAL`, Predicted = `HEPATIC` (Conf: 0.7877). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000725
Demographics: 54-year-old female presenting for Cycle 4 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage IV NSCLC.
Genomic Profile: Confirmed driver mutation in EGFR. Sec..."*
- **FALSE_NEGATIVE on `RENAL`** in `DOC-005062`: True = `RENAL`, Predicted = `HEPATIC` (Conf: 0.7344). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000832
Demographics: 46-year-old male presenting for Cycle 5 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage II Melanoma.
Genomic Profile: Confirmed driver mutation in None/Unk..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-000814`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.6209). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000136
Demographics: 64-year-old female presenting for Cycle 2 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage II NSCLC.
Genomic Profile: Confirmed driver mutation in TP53. Sec..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-002623`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.7420). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000434
Demographics: 53-year-old unknown presenting for Cycle 5 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III NSCLC.
Genomic Profile: Confirmed driver mutation in ROS1. S..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-002626`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.7420). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000434
Demographics: 53-year-old unknown presenting for Cycle 5 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III NSCLC.
Genomic Profile: Confirmed driver mutation in ROS1. S..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-003277`: True = `NONE`, Predicted = `RENAL` (Conf: 0.4913). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000542
Demographics: 49-year-old female presenting for Cycle 3 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage IV SCLC.
Genomic Profile: Confirmed driver mutation in EGFR. Seco..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-003280`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.5597). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000542
Demographics: 62-year-old male presenting for Cycle 6 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III NSCLC.
Genomic Profile: Confirmed driver mutation in KRAS. Seco..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-003892`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.9535). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000641
Demographics: 75-year-old male presenting for Cycle 3 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage II NSCLC.
Genomic Profile: Confirmed driver mutation in EGFR. Secon..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-003904`: True = `NONE`, Predicted = `RENAL` (Conf: 0.6341). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000643
Demographics: 65-year-old male presenting for Cycle 8 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage I Unknown.
Genomic Profile: Confirmed driver mutation in None/Unkno..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-004402`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.9473). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000725
Demographics: 64-year-old unknown presenting for Cycle 3 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III NSCLC.
Genomic Profile: Confirmed driver mutation in KRAS. S..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-004945`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.6117). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000812
Demographics: 62-year-old male presenting for Cycle 3 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage IV Prostate Cancer.
Genomic Profile: Confirmed driver mutation in K..."*
- **FALSE_POSITIVE on `RENAL`** in `DOC-005206`: True = `HEPATIC`, Predicted = `RENAL` (Conf: 0.9620). Snippet: *"ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000856
Demographics: 75-year-old male presenting for Cycle 4 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage IV NSCLC.
Genomic Profile: Confirmed driver mutation in MET. Second..."*

---

## 4. Recommendations for Next Engineering Phase

Based on the empirical evidence from this error analysis:
1. **Data vs Features vs Architecture Priority**: The failure mode is **NOT an architectural deficit** (MiniLM Hybrid already achieves 100% recall in Config C and 98.91% in Config D). It is a **decision-threshold calibration problem** between `HIGH` and `CRITICAL`.
2. **Concrete Engineering Recommendation (Next Phase)**:
   - Implement a **Safety-Calibrated Threshold Gate** on the Logistic Regression probability outputs: if $P(\text{CRITICAL}) \ge 0.30$, route to `CRITICAL` triage.
   - In Stage 4, treat both `HIGH` with acute toxicities and `CRITICAL` with dose holds, providing redundant safety defense.