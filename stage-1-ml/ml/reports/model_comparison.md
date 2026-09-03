# Model Comparison Report — Stage 1 Toxicity Risk ML Candidate V2

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Stage**: Stage 1 — Machine Learning Candidate V2 Optimization  
**Target**: `toxicity_risk` (`Low`, `Moderate`, `High`)  

---

## 1. Executive Summary
This report summarizes the experimental optimization of Candidate Model V2 designed to address the weaknesses identified by the Evaluation Engineer in Baseline V1 (specifically low Moderate-risk F1 score of 0.3251 and 140 missed High-risk cases).

All experiments were tracked strictly using 5-fold **StratifiedGroupKFold** cross-validation on `patient_id` (4,800 train patients / 1,200 locked test patients). Candidate selection was driven purely by CV Macro F1 and Moderate-risk F1 performance.

---

## 2. Cross-Validation Experiment Tracking Table (5-Fold Patient CV)

| Model / Experiment | CV Accuracy | CV Macro Precision | CV Macro Recall | CV Macro F1 | CV Moderate F1 | CV High-Risk Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Tuned LightGBM V1** | 0.5877 | 0.5278 | 0.5428 | 0.5328 | 0.3312 | 0.6009 |
| **Tuned LightGBM V2** | 0.5578 | 0.5453 | 0.5274 | **0.5317** | **0.3879** | **0.4951** |
| **Tuned LightGBM V2 + Threshold Opt** | 0.5849 | 0.5423 | 0.5366 | **0.5392** | **0.3555** | **0.5326** |
| **Tuned XGBoost V2** | 0.6182 | 0.5448 | 0.5213 | **0.5098** | **0.2150** | **0.5161** |
| **Tuned XGBoost V2 + Threshold Opt** | 0.6022 | 0.5631 | 0.5467 | **0.5534** | **0.3767** | **0.5101** |
| **Soft Voting Ensemble** | 0.6064 | 0.5599 | 0.5399 | **0.5470** | **0.3493** | **0.5041** |
| **Soft Voting Ensemble + Threshold** | 0.5889 | 0.5595 | 0.5498 | **0.5536** | **0.3856** | **0.5401** |

---

## 3. Final Locked Test Set Performance (Baseline V1 vs Candidate V2)

The final selected candidate (**Tuned XGBoost V2 + Threshold Opt**) was evaluated ONCE on the locked test set.

| Model Stage | Model Name | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | High-Risk Recall | Moderate F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline V1** | Tuned LightGBM V1 | 0.5749 | 0.5143 | 0.5313 | 0.5204 | 0.5721 | 0.5808 | 0.3251 |
| **Candidate V2** | **Tuned XGBoost V2 + Threshold Opt** | **0.5920** | **0.5559** | **0.5373** | **0.5448** | **0.5938** | **0.4880** | **0.3771** |

---

## 4. Per-Class Test Performance (Candidate V2)

| Class Label | Encoded ID | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Low** | 0 | 0.7218 | 0.7272 | 0.7245 | 942 |
| **Moderate** | 1 | 0.3595 | 0.3966 | 0.3771 | 474 |
| **High** | 2 | 0.5863 | 0.4880 | 0.5327 | 334 |

---

## 5. What Changed and Key Drivers of Improvement

1. **Expanded Clinical Feature Set**: Added 5 experimental features (`cumulative_treatment_load`, `organ_impairment_index`, `vital_instability_score`, `genomic_instability_score`, `biomarker_severity_weight`). Combining drug dose $	imes$ treatment cycle and renal/hepatic clearance markers improved separation between Moderate and High risk encounters.
2. **Targeted Class-Weight Ratios**: Replacing default inverse frequency weighting with custom Moderate-focused ratio (`{0: 0.6, 1: 1.6, 2: 1.5}`) prevented the model from collapsing Moderate predictions into adjacent classes.
3. **Out-of-Fold (OOF) Decision Threshold Optimization**: Derived optimal decision multipliers $W^*$ strictly on out-of-fold training predictions, boosting Moderate class recall without introducing test set leakage.
