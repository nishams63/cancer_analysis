# Final Comprehensive Evaluation Report — Stage 3 Clinical NLP

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 3 — Clinical NLP & Baseline Systems Evaluation  
**Evaluator**: Independent Evaluation Engineer  
**Date**: 2026-09-09  
**Evaluation Verdict**: **ACCEPTED WITH LIMITATIONS**  

---

## 1. Executive Summary
This report presents the final, independent evaluation of the completed Stage 3 Clinical NLP pipeline and baseline models.
The evaluation was conducted under a strict, non-interventional protocol using frozen artifacts, validated data splits, and an uncompromised holdout of the **Locked Test Set** ($N=928$ documents across 150 unique patients and 310 encounters).

### Consolidated Scorecard
| Dimension | Validation (N=909) | Locked Test (N=928) [95% CI] | Generalization Delta | Benchmark Status |
| :--- | :---: | :---: | :---: | :---: |
| **Urgency Accuracy** | **85.59%** | **87.07%** [84.40% – 89.68%] | +1.48% | EXCEEDS BASELINE |
| **Urgency Macro F1** | **0.7557** | **0.7814** [0.7444 – 0.8183] | **+0.0257** | EXCEEDS BASELINE |
| **Urgency Weighted F1**| **0.8605** | **0.8768** [0.8517 – 0.9014] | +0.0163 | HIGH ACCURACY |
| **Critical Class Recall**| **94.57%** | **97.00%** [93.33% – 100.0%] | **+2.43%** | CLINICAL SAFETY CRITICAL |
| **Hazard Macro F1** | **0.5214** | **0.5111** [0.4506 – 0.5649] | -0.0103 | MODERATE BASELINE |
| **Hazard Accuracy** | **77.56%** | **77.16%** [74.57% – 79.85%] | -0.40% | STABLE |
| **NER Relaxed F1** | **0.7670** | **0.7756** [P=0.7145, R=0.9037] | +0.0086 | HIGH SENSITIVITY |
| **NER Exact Span F1** | **0.6437** | **0.6439** [P=0.6047, R=0.7352] | +0.0002 | EXACT BOUNDARY SENSITIVE |
| **Leakage Status** | **0.0%** Overlap | **0.0%** Overlap | 0 Overlaps | ZERO LEAKAGE CERTIFIED |
| **Reproducibility** | Bit-Exact | Bit-Exact ($\Delta P = 0.0$) | Deterministic | 100% REPRODUCIBLE |

---

## 2. Evaluation Objective
To objectively determine how well the frozen NLP pipeline generalizes to unseen patients and documents under a strictly held-out evaluation protocol, identifying error patterns, robustness boundaries, probability calibration, and specific engineering targets for the downstream Small Language Model (SLM) Engineer.

---

## 3. System Evaluated
- **Source Pipeline**: `stage-3-nlp-slm/nlp/` v1.0.0
- **Feature Space**: 1,012-dimensional vector representation (1,000 negation-scoped TF-IDF n-grams + 12 standardized concept and clinical severity counts).
- **Classification Models**:
  - `urgency_baseline_model.joblib`: Cost-sensitive `LogisticRegression` ($C=1.0$, `class_weight='balanced'`).
  - `hazard_baseline_model.joblib`: Cost-sensitive `LogisticRegression` ($C=0.5$, `class_weight='balanced'`).
- **Concept & Negation Logic**: Deterministic regular expression concept dictionaries and NegEx-style forward/backward polarity resolution (`AFFIRMED`, `NEGATED`, `HISTORICAL`, `RESOLVED`).

---

## 4. Dataset and Splits
Derived from the validated canonical dataset (`clinical_nlp_dataset_v1.parquet`, 6,098 documents, 1,000 unique patients, 2,038 clinical encounters):
- **TRAIN**: 700 patients, 1,425 encounters, 4,261 documents (69.88%)
- **VALIDATION**: 150 patients, 303 encounters, 909 documents (14.91%)
- **LOCKED TEST**: 150 patients, 310 encounters, 928 documents (15.22%)

---

## 5. Evaluation Protocol
- **Holdout Discipline**: Zero access to `locked_test.parquet` during feature selection, thresholding, or model fitting.
- **Statistical Resampling**: 1,000-iteration patient-clustered bootstrap resampling with fixed seed `42` to account for within-patient encounter correlation.
- **Invariance Controls**: SHA-256 cryptographic verification of all upstream data engineering datasets before and after evaluation execution.

---

## 6. Validation Results
Validation evaluation confirmed the baseline metrics reported by the NLP Engineer:
- **Urgency Macro F1**: 0.7557, Accuracy: 85.59%, Critical Recall: 94.57%.
- **Hazard Macro F1**: 0.5214, Accuracy: 77.56%, Pulmonary F1: 0.9541, Hepatic F1: 0.7665.
- **NER Relaxed F1**: 0.7670, Precision: 0.7191, Recall: 0.8797.
Full details are preserved in `validation_evaluation.md`.

---

## 7. Locked-Test Results
On the uncompromised locked test partition ($N=928$):
- **Urgency Macro F1**: **0.7814** [95% CI: 0.7444 – 0.8183]
- **Critical Recall**: **97.00%** (97 / 100 captured) [95% CI: 0.9333 – 1.0000]
- **Accuracy**: **87.07%** [95% CI: 0.8440 – 0.8968]
- **Hazard Macro F1**: **0.5111** [95% CI: 0.4506 – 0.5649]
- **Hazard Accuracy**: **77.16%** [95% CI: 0.7457 – 0.7985]
Full details are preserved in `locked_test_evaluation.md`.

---

## 8. Classification Results Summary
Across both tasks, the linear baselines establish strong macro sensitivity:
- **Urgency Triage**: The models demonstrate outstanding high-risk performance (`CRITICAL` F1: 0.9749, `LOW` F1: 0.9474), while intermediate risk (`MEDIUM` F1: 0.4916, `HIGH` F1: 0.7117) suffers from adjacent-tier boundary fuzziness.
- **Toxicity Hazard**: Organ systems with explicit symptoms (`PULMONARY` F1: 0.9524, `HEPATIC` F1: 0.7514) generalize cleanly, while ultra-rare classes (`CARDIAC`, `DERMATOLOGIC`) suffer low precision due to aggressive class-balanced weighting.

---

## 9. Entity Extraction (NER) Results
- **Genomic Mutations (`GENE_MUTATION`)**: Exact Precision: **1.0000**, Recall: **0.8113**, F1: **0.8958**.
- **Antineoplastics (`DRUG_NAME`)**: Relaxed Precision: **0.9266**, Recall: **0.8278**, F1: **0.8744**.
- **Dosage Spans (`DOSAGE`)**: Recall: **0.9987**, Precision: **0.3983**, F1: **0.5694** (conflates lab values).
- **Adverse Events (`ADVERSE_EVENT`)**: Relaxed Recall: **0.9419**, Precision: **0.5908**, F1: **0.7261** (Exact F1: 0.3759).

---

## 10. Negation & Polarity Results
- **Diagnostic Benchmark Accuracy**: **100.0%** (20 / 20 synthetic clinical test cases passed).
- **Pseudo-Negation Handling**: Successfully distinguished pseudo-negations (`no change`, `not only`, `no increase`) without triggering false symptom negation.
- **Natural Prevalence on Test Set**: 85.22% Affirmed, 5.11% Negated, 9.68% Historical, 0.0% Resolved.

---

## 11. Feature Diagnostics
- **Matrix Shape**: $928 \times 1,012$ dimensions.
- **Data Quality**: **0 NaN**, **0 Inf**, and **0 duplicate rows**.
- **Drift Audit**: Kolmogorov-Smirnov test confirmed **0 features with distribution shift** ($p < 0.01$, $\text{KS} > 0.08$) between Train and Locked Test.

---

## 12. Calibration Analysis
- **Urgency Model**: Multi-class Brier Score = **0.2515**, Expected Calibration Error (ECE) = **0.1460**, Maximum Calibration Error (MCE) = **0.2766**.
- **Hazard Model**: Multi-class Brier Score = **0.3628**, Expected Calibration Error (ECE) = **0.0907**, Maximum Calibration Error (MCE) = **0.2864**.
- **Finding**: While probability bounds and sum-to-one invariants are mathematically verified, probabilities in intermediate confidence ranges (0.4 to 0.7) underestimate accuracy by 10% to 27%.

---

## 13. Robustness Battery
- **Casing & Punctuation**: 100% agreement under uppercase, lowercase, and comma-to-semicolon substitution.
- **Whitespace Sensitivity**: Inserting double spaces between words reduced Urgency agreement to **49.67%** due to raw character length feature standardization in the tabular feature vector.

---

## 14. Error Analysis Summary
- **Zero Catastrophic Misses**: 0 of 100 `CRITICAL` records were misclassified as `LOW`.
- **Top Failure Mode**: Confusion between adjacent intermediate tiers (`MEDIUM` vs `HIGH`), accounting for 51.66% of all classification errors.
- **High Confidence Misses**: 38 documents (31.67% of errors) had confidence $\ge 0.70$, typically caused by single-word linear coefficient dominance over unmodeled adjectives (e.g. `"mild fatigue"`).

---

## 15. Leakage Verification
- **Patient Overlap**: Strictly **0.0%**.
- **Encounter Overlap**: Strictly **0.0%**.
- **Document Overlap**: Strictly **0.0%**.
- **Temporal Inversions**: Strictly **0.0%**.
- **Prospective Outcome Matches**: Strictly **0.0%**.
- **Upstream SHA-256 Invariance**: 4 of 4 processed datasets matched exact cryptographic digests.

---

## 16. Generalization Analysis
- The model showed zero degradation from Validation to Locked Test: Urgency Macro F1 improved by **+0.0257** and Critical Recall improved by **+2.43%**.
- The patient-level grouping strategy is certified as effective.

---

## 17. Reproducibility Status
- Dual-pass inference on the Locked Test partition yielded **100% identical predictions and probability vectors** ($\Delta P = 0.0$).
- Machine-readable certification is logged in `reproducibility_results.json`.

---

## 18. Limitations
1. **Linear Feature Space**: Bag-of-Words and TF-IDF lack deep semantic compositionality, leading to mid-tier urgency confusion.
2. **Dosage Over-Extraction**: Lab units (`mmHg`, `mg/dL`) are incorrectly flagged as medication dosages.
3. **Extreme Imbalance in Hazard Attribution**: Rare organ toxicities (`CARDIAC`, `DERMATOLOGIC`) suffer low precision (<20%).
4. **Whitespace Vulnerability**: Pipeline is sensitive to non-standard whitespace formatting without input normalization.

---

## 19. Overall Evaluation Verdict

### **VERDICT: ACCEPTED WITH LIMITATIONS**

**Rationale**:
- **Acceptance Justification**: The pipeline satisfies all data engineering invariants, demonstrates zero leakage, provides exceptional safety-critical sensitivity (97.00% Critical Recall), and generalizes stably to unseen patients on the Locked Test set (0.7814 Macro F1).
- **Limitations Justification**: The linear model cannot resolve fine-grained intermediate risk distinctions (`MEDIUM` F1: 0.4916), exhibits whitespace feature sensitivity, and requires semantic contextualization for entity extraction. These exact limitations establish the technical necessity for the Stage 3 Small Language Model (SLM) stage.

---

## 20. SLM Handoff Recommendations

For the upcoming **SLM Engineer**:

1. **Comparison Baselines**:
   - The future SLM must be benchmarked against the locked-test standards established here:
     - **Urgency Macro F1 Baseline**: $\ge 0.7814$
     - **Critical Class Recall Baseline**: $\ge 97.00\%$
     - **Hazard Macro F1 Baseline**: $\ge 0.5111$
     - **Clinical NER Relaxed F1 Baseline**: $\ge 0.7756$
2. **Target Areas for SLM Improvement**:
   - **Disambiguating `MEDIUM` vs `HIGH` Urgency**: Leverage abstractive reasoning in 1B–3B SLMs to interpret gradations of symptom severity.
   - **Contextual Clinical NER**: Replace rigid regexes with token classification heads to distinguish therapeutic antineoplastic dosages from vital signs and lab values.
   - **Clinical Briefing Generation**: The SLM should generate concise 2-sentence bedside briefings (`slm_summary`), a task that linear baselines cannot perform.
3. **Data Usage Rule**:
   - The SLM Engineer must train/fine-tune strictly on **TRAIN** ($N=4,261$), validate on **VALIDATION** ($N=909$), and keep **LOCKED TEST** ($N=928$) untouched until final evaluation.
