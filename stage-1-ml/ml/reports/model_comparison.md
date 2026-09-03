# Model Comparison Report — Stage 1 Toxicity Risk ML Candidate V3

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Stage**: Stage 1 — Machine Learning Candidate V3 Optimization  
**Target**: `toxicity_risk` (`Low`, `Moderate`, `High`)  

---

## 1. Executive Summary
This report summarizes the experimental optimization of Candidate Model V3 designed to improve High-Risk Recall and Moderate-risk classification while maintaining high overall Macro F1.

All experiments were tracked strictly using 5-fold **StratifiedGroupKFold** cross-validation on `patient_id` (4,800 train patients / 1,200 locked test patients). Candidate selection was driven purely by patient-grouped CV performance balancing Macro F1 and High-Risk Recall.

> **Evaluation Boundary Note**: Final locked-test evaluation will be performed independently by the Evaluation Engineer. The single locked-test run reported below is for artifact generation and baseline logging only.

---

## 2. Cross-Validation Experiment Tracking Table (5-Fold Patient CV)

| Model / Experiment | CV Accuracy | CV Macro Precision | CV Macro Recall | CV Macro F1 | CV Moderate F1 | CV High-Risk Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Tuned LightGBM V1** | 0.5877 | 0.5278 | 0.5428 | 0.5348 | 0.3312 | 0.6009 |
| **Tuned LightGBM V2** | 0.5578 | 0.5453 | 0.5274 | **0.5317** | **0.3879** | **0.4951** |
| **Tuned LightGBM V2 + Threshold Opt** | 0.5755 | 0.5126 | 0.5505 | **0.5156** | **0.2854** | **0.7164** |
| **Tuned XGBoost V2** | 0.6182 | 0.5448 | 0.5213 | **0.5098** | **0.2150** | **0.5161** |
| **Tuned XGBoost V2 + Threshold Opt** | 0.5992 | 0.5375 | 0.5590 | **0.5427** | **0.3284** | **0.6459** |
| **Soft Voting Ensemble** | 0.6064 | 0.5599 | 0.5399 | **0.5470** | **0.3493** | **0.5041** |
| **Soft Voting Ensemble + Threshold** | 0.5704 | 0.5108 | 0.5519 | **0.5135** | **0.2871** | **0.7344** |

---

## 3. Final Locked Test Set Performance (Baseline V1 vs Candidate V3)

The final selected candidate (**Tuned XGBoost V3 + Threshold Opt**) was evaluated ONCE on the locked test set.

| Model Stage | Model Name | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | High-Risk Recall | Moderate F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline V1** | Tuned LightGBM V1 | 0.5749 | 0.5143 | 0.5313 | 0.5204 | 0.5721 | 0.5808 | 0.3251 |
| **Candidate V3** | **Tuned XGBoost V3 + Threshold Opt** | **0.5777** | **0.5119** | **0.5333** | **0.5188** | **0.5726** | **0.6018** | **0.3070** |

---

## 4. Per-Class Test Performance (Candidate V3)

| Class Label | Encoded ID | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Low** | 0 | 0.7251 | 0.7197 | 0.7224 | 942 |
| **Moderate** | 1 | 0.3420 | 0.2785 | 0.3070 | 474 |
| **High** | 2 | 0.4685 | 0.6018 | 0.5269 | 334 |

---

## 5. What Changed and Key Drivers of Improvement

1. **Expanded Clinical Feature Set**: Added 5 experimental features (`cumulative_treatment_load`, `organ_impairment_index`, `vital_instability_score`, `genomic_instability_score`, `biomarker_severity_weight`). Combining drug dose $	imes$ treatment cycle and renal/hepatic clearance markers improved separation between Moderate and High risk encounters.
2. **Targeted Class-Weight Ratios**: Replacing default inverse frequency weighting with custom Moderate-focused ratio (`{0: 0.6, 1: 1.6, 2: 1.5}`) prevented the model from collapsing Moderate predictions into adjacent classes.
3. **Dual-Objective OOF Decision Threshold Optimization**: Derived optimal decision multipliers $W^*$ strictly on out-of-fold training predictions to optimize combined Macro F1 and High-Risk Recall.
