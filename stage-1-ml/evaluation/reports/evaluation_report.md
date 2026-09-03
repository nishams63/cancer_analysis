# Stage 1 ML Evaluation Report

**Project Title**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Role**: Evaluation Engineer (Independent Assessment)  
**Target Variable**: `toxicity_risk` (`Low`, `Moderate`, `High`)  
**Target Mapping**: `Low` $\rightarrow$ `0`, `Moderate` $\rightarrow$ `1`, `High` $\rightarrow$ `2`  

---

## 1. Objective
The objective of this report is to provide an independent, rigorous evaluation and validation of the Stage 1 Machine Learning system built by the ML Engineer. As the Evaluation Engineer, our role is to independently reproduce all reported test set performance metrics, stress-test model behavior across error categories and probability confidence ranges, run pipeline robustness sanity checks, and determine if the system is ready to hand off to the Integration Engineer as a research prototype.

> [!NOTE]
> **Strict Evaluation Guardrails**: The model (**Tuned LightGBM**) was evaluated strictly as provided by the ML Engineer. No retraining, hyperparameter tuning, model re-selection, or modification of the locked patient-level test split was performed.

---

## 2. Model Evaluated
- **Selected Model**: **Tuned LightGBM** (`LightGBMClassifier` with `class_weight='balanced'`, `n_estimators=100`, `max_depth=8`, `num_leaves=31`, `learning_rate=0.05`).
- **Artifact Path**: `stage-1-ml/ml/models/best_model/model.joblib`
- **Preprocessor Artifact**: `stage-1-ml/ml/artifacts/preprocessor/preprocessor.joblib`
- **Model Provenance**: Selected by the ML Engineer via 5-fold `StratifiedGroupKFold` cross-validation on `patient_id` (CV Macro F1 = 0.5348).

---

## 3. Test Dataset & Patient-Level Split Verification

The master dataset (`stage-1-ml/data-engineering/data/processed/master_patient_dataset.csv`) contains **8,754 rows** and **35 columns** across 6,000 unique patients.

### Verification Audit
- **Test Set Encounters**: **1,750 rows** (20.0% of total dataset).
- **Test Set Unique Patients**: **1,200 unique patients**.
- **Train Set Encounters**: **7,004 rows** across **4,800 unique patients**.
- **Patient-Level Split Validation**: `set(train_patient_ids).intersection(set(test_patient_ids))` $\equiv \emptyset$ (**0 patient overlap**).
- **Target Class Mapping Verification**: `Low` = 0, `Moderate` = 1, `High` = 2.
- **Leakage Audit**: Excluded non-predictive identifiers (`patient_id`, `encounter_id`, `observation_date`) and concurrent treatment outcome leakage (`treatment_response`).

---

## 4. Overall Performance Summary

The independently reproduced evaluation metrics on the locked test set (1,750 rows / 1,200 patients) match the ML Engineer's reported figures:

| Metric | ML Engineer Reported | Evaluation Engineer Reproduced | Match Status |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.5749 | **0.5749** | EXACT |
| **Macro Precision** | 0.5143 | **0.5143** | EXACT |
| **Macro Recall** | 0.5313 | **0.5313** | EXACT |
| **Macro F1 Score** | 0.5204 | **0.5204** | EXACT |
| **Weighted F1 Score** | 0.5721 | **0.5721** | EXACT |
| **High-Risk Recall** | 0.5808 | **0.5808** | EXACT |

---

## 5. Class-Level Performance

Per-class performance breakdown on the locked test set:

| Class Label | Encoded ID | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Low Risk** | 0 | 0.7191 | 0.7091 | 0.7141 | 942 |
| **Moderate Risk** | 1 | 0.3495 | 0.3038 | 0.3251 | 474 |
| **High Risk** | 2 | 0.4743 | 0.5808 | 0.5222 | 334 |

---

## 6. Confusion Matrix Analysis

```
                     Predicted Low    Predicted Moderate    Predicted High
Actual Low              668 (70.9%)      191 (20.3%)           83 (8.8%)
Actual Moderate         198 (41.8%)      144 (30.4%)          132 (27.8%)
Actual High              63 (18.9%)       77 (23.1%)          194 (58.1%)
```

### Key Observation Patterns
1. **Low Risk Dominance**: The model performs best on `Low Risk` (F1 = 0.7141), correctly classifying 668 out of 942 encounters (70.9%).
2. **Moderate Risk Difficulty**: `Moderate Risk` is the hardest class to predict (F1 = 0.3251). It suffers high confusion, with 41.8% of Moderate cases misclassified as `Low` and 27.8% misclassified as `High`.
3. **Bimodal Distinctions**: The model exhibits strong separation between extreme ends (`Low` vs `High`), with only 8.8% of `Low` cases predicted as `High` and 18.9% of `High` cases predicted as `Low`.

---

## 7. High-Risk Class Deep-Dive

In oncology clinical decision support, identifying patients at high risk of severe toxicity is paramount:

- **Total Actual High-Risk Samples**: **334**
- **True Positives (Correctly Identified High Risk)**: **194**
- **False Positives (Falsely Flagged High Risk)**: **215**
- **False Negatives (Missed High Risk Cases)**: **140**
  - High Risk misclassified as Moderate: **77**
  - High Risk misclassified as Low: **63**
- **High-Risk Recall**: **58.08%**
- **High-Risk Precision**: **47.43%**
- **High-Risk F1 Score**: **52.22%**

> [!IMPORTANT]
> **Safety Context of False Negatives**: While class balancing improved High-Risk recall to 58.08%, 140 actual High-Risk patient encounters were missed (63 labeled Low and 77 labeled Moderate). In a real-world clinical deployment, missed high-risk toxicities could lead to unmonitored adverse drug events. Therefore, this model must remain strictly a research prototype.

---

## 8. Error Analysis

Row-by-row analysis of all 1,750 test predictions revealed **744 errors** (Error Rate: **42.51%**). The error transition breakdown is summarized below:

| Error Transition | Occurrences | Pct of Total Errors | Description & Impact |
| :--- | :---: | :---: | :--- |
| **Moderate $\rightarrow$ Low** | 198 | 26.6% | Most frequent error; moderate toxicities underestimated as low risk. |
| **Low $\rightarrow$ Moderate** | 191 | 25.7% | Over-conservative prediction; low risk patients flagged for moderate monitoring. |
| **Moderate $\rightarrow$ High** | 132 | 17.7% | Over-conservative prediction; moderate risk escalated to high risk. |
| **Low $\rightarrow$ High** | 83 | 11.2% | Severe over-estimation of risk for baseline low-risk patients. |
| **High $\rightarrow$ Moderate** | 77 | 10.3% | Severe toxicity risk underestimated by one risk tier. |
| **High $\rightarrow$ Low** | 63 | 8.5% | Critical error; severe toxicity risk completely missed as low risk. |

---

## 9. Probability & Confidence Analysis

All test predictions were subjected to probability validation:
1. **3-Class Probabilities**: Every sample produced exact 3-class probability distributions (`prob_low`, `prob_moderate`, `prob_high`).
2. **Bounds & Sums**: All probabilities were bounded in $[0.0, 1.0]$ and summed to $1.0000 \pm 10^{-4}$.
3. **Argmax Match**: The predicted class matched `np.argmax(probabilities)` in 100% of cases.

### Prediction Confidence vs. Accuracy

Grouping test predictions by maximum predicted probability reveals a clear, monotonic relationship between model confidence and accuracy:

| Confidence Bin | Sample Count | Pct of Test Set | Empirical Accuracy |
| :--- | :---: | :---: | :---: |
| **0.33 – 0.50** (Low Confidence) | 721 | 41.2% | 43.69% |
| **0.50 – 0.60** (Moderate Confidence) | 452 | 25.8% | 59.07% |
| **0.60 – 0.70** (Good Confidence) | 331 | 18.9% | 71.60% |
| **0.70 – 0.80** (High Confidence) | 177 | 10.1% | 75.14% |
| **0.80 – 0.90** (Very High Confidence) | 62 | 3.5% | 79.03% |
| **0.90 – 1.00** (Extreme Confidence) | 7 | 0.4% | 71.43% |

*Insight*: When prediction confidence exceeds 0.70, accuracy rises above 75%, indicating that prediction probability can serve as a useful confidence flag for downstream consumers.

---

## 10. Robustness Verification

The evaluation test suite ([`test_evaluation.py`](file:///C:/Users/nisham/.gemini/antigravity-ide/scratch/cancer_analysis/stage-1-ml/evaluation/tests/test_evaluation.py)) executed 10 automated unit tests:

```
stage-1-ml/evaluation/tests/test_evaluation.py::test_metric_calculation PASSED
stage-1-ml/evaluation/tests/test_evaluation.py::test_classification_report_generation PASSED
stage-1-ml/evaluation/tests/test_evaluation.py::test_confusion_matrix_generation PASSED
stage-1-ml/evaluation/tests/test_evaluation.py::test_probability_validation PASSED
stage-1-ml/evaluation/tests/test_evaluation.py::test_invalid_nan_probability_detection PASSED
stage-1-ml/evaluation/tests/test_evaluation.py::test_target_mapping PASSED
stage-1-ml/evaluation/tests/test_evaluation.py::test_model_artifact_loading PASSED
stage-1-ml/evaluation/tests/test_evaluation.py::test_preprocessor_artifact_loading PASSED
stage-1-ml/evaluation/tests/test_evaluation.py::test_prediction_shape PASSED
stage-1-ml/evaluation/tests/test_evaluation.py::test_patient_overlap_detection PASSED
```

- **Result**: **10 / 10 Tests PASSED** (0 failures).
- **Sanity Verification**: Single patient dictionary records pass through feature engineering, preprocessor transformation, and LightGBM prediction without producing NaN or invalid probabilities.

---

## 11. Limitations & Non-Medical Prototype Disclaimer

> [!WARNING]
> **Research Prototype Disclaimer**: This system is an educational research prototype built to explore multi-modal oncology risk modeling. It has **NOT** been clinically validated or cleared for diagnostic or therapeutic use.

### Explicit Model Limitations
1. **Overall Performance Cap**: Macro F1 score is 0.5204 and Accuracy is 57.49%, leaving substantial room for error.
2. **Weak Moderate Risk Prediction**: Moderate Risk F1 score is only 0.3251 due to severe overlap with Low and High risk feature profiles.
3. **Synthetic / Dataset Constraints**: Performance on this dataset does not reflect real-world clinical population distributions or multi-site hospital EHR datasets.
4. **No Causation**: Predictive feature associations (e.g. blood pressure ratio or biomarker trends) do not establish biological causation.

---

## 12. Final Evaluation Conclusion & Handoff Recommendation

### Final Verdict: **APPROVED FOR HANDOFF TO INTEGRATION ENGINEER**

1. **Reproducibility**: The ML Engineer's reported locked-test results were 100% independently reproduced without discrepancy.
2. **Data Integrity & Leakage**: Patient-level splitting was strictly enforced with 0 patient overlap.
3. **Pipeline Readiness**: The model (`model.joblib`), preprocessor (`preprocessor.joblib`), target mapping (`target_mapping.json`), and standalone inference API (`predict.py`) are robust, testable, and ready for integration.

The evaluation suite, error analysis tables, metrics, plots, and test cases are ready for handoff to the **Integration Engineer**.
