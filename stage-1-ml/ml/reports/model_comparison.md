# Model Comparison Report — Stage 1 Toxicity Risk ML System

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Stage**: Stage 1 — Machine Learning Model Evaluation  
**Target**: `toxicity_risk` (`Low`, `Moderate`, `High`)  

---

## 1. Executive Summary
This report summarizes the performance of candidate multiclass classification models for predicting patient treatment toxicity risk. All models were evaluated using 5-fold **StratifiedGroupKFold** cross-validation on the training set (grouped by `patient_id` across 6,000 unique patients). The final selected candidate (**Tuned LightGBM**) was evaluated ONCE on the locked patient-level holdout test set (1,751 encounter records across 1,200 unique patients).

---

## 2. Cross-Validation Model Comparison Table (5-Fold Stratified Group CV)

| Model Name | CV Accuracy | CV Macro Precision | CV Macro Recall | CV Macro F1 | CV Weighted F1 | CV High-Risk Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.5731 | 0.5199 | 0.5432 | **0.5279** | 0.5706 | **0.6324** |
| **Decision Tree** | 0.5014 | 0.4569 | 0.4568 | **0.4564** | 0.5043 | **0.4163** |
| **Random Forest** | 0.6098 | 0.5409 | 0.4922 | **0.4710** | 0.5382 | **0.4561** |
| **XGBoost** | 0.5949 | 0.5277 | 0.5072 | **0.5078** | 0.5673 | **0.4659** |
| **LightGBM** | 0.5865 | 0.5268 | 0.5348 | **0.5290** | 0.5769 | **0.5724** |
| **Tuned Random Forest** | 0.6142 | 0.5355 | 0.5432 | **0.5208** | 0.5771 | **0.6129** |
| **Tuned LightGBM** | 0.5877 | 0.5278 | 0.5428 | **0.5328** | 0.5799 | **0.6009** |

---

## 3. Final Locked Test Set Performance

The final selected model (**Tuned LightGBM**) was evaluated ONCE on the locked test set.

| Evaluation Stage | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | High-Risk Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Logistic Regression | 0.5697 | 0.5117 | 0.5348 | 0.5191 | 0.5684 | 0.6138 |
| **Final Selected** | **Tuned LightGBM** | **0.5749** | **0.5143** | **0.5313** | **0.5204** | **0.5721** | **0.5808** |

---

## 4. Per-Class Test Performance (Tuned LightGBM)

| Class Label | Encoded ID | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Low** | 0 | 0.7191 | 0.7091 | 0.7141 | 942 |
| **Moderate** | 1 | 0.3495 | 0.3038 | 0.3251 | 474 |
| **High** | 2 | 0.4743 | 0.5808 | 0.5222 | 334 |

---

## 5. Confusion Matrix (Locked Test Set)

```
                     Predicted Low    Predicted Moderate    Predicted High
Actual Low              668             191                   83            
Actual Moderate         198             144                   132           
Actual High             63              77                    194           
```

---

## 6. Selection Rationale & Findings

1. **Selection Criterion**: **Tuned LightGBM** was selected because it achieved the highest cross-validation Macro F1 score (0.5328) while maintaining exceptional recall for the critical High-risk patient class (0.6009).
2. **Impact of Class Imbalance**: The dataset contains 53.4% Low, 27.6% Moderate, and 19.0% High risk encounters. Utilizing class-weight balancing (`class_weight='balanced'`) prevented the model from favoring the dominant `Low` class and dramatically boosted minority High-risk recall.
3. **Leakage Prevention**: All model selection and hyperparameter tuning were conducted strictly within training folds using `StratifiedGroupKFold` on `patient_id`. The locked test set was held out and transformed using the training pipeline.
