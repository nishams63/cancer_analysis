# Model Comparison Report -- V1 vs V3 vs V4

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Stage**: Stage 1 -- Candidate V4 Generalization-First Optimization  

---

## Cross-Validation Performance Comparison

| Model | CV Macro F1 | CV HR Recall | CV Mod F1 | CV Accuracy | Train F1 | Gap | Status / Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **V1 Tuned LightGBM** | 0.5348 | 0.6009 | 0.3312 | 0.5877 | N/A | N/A | Superseded baseline |
| **V3 XGB + Thresh** | 0.5427 | 0.6459 | 0.3284 | 0.5992 | 0.7763 | 0.2665 | Rejected: Severe overfitting |
| **V4 Regularized LightGBM** | **0.5427** | **0.6444** | **0.3332** | **0.5889** | **0.6378** | **0.0950** | **SELECTED & FROZEN** |
| *Class Balancing B2 (Mod 1.2x)* | 0.5431 | 0.5882 | 0.3716 | 0.5691 | 0.6452 | 0.1021 | Rejected: HR Recall dropped < 0.62 |
| *Class Balancing B8 (ROS)* | 0.5397 | 0.6227 | 0.3213 | 0.5898 | 0.6365 | 0.0968 | Rejected: Below V4 Macro F1 |
| *Class Balancing B9 (SMOTE)* | 0.5056 | 0.5844 | 0.1869 | 0.6239 | 0.5774 | 0.0718 | Rejected: Distortion of mixed space |

## Class Balancing Investigation Outcome
In September 2026, an exhaustive study benchmarked 12 balancing strategies (B0–B11) using patient-aware 5-fold CV to test if class imbalance explained Moderate-risk weakness.
- **Finding**: Moderate risk comprises 27.6% of training encounters and is not rare. Aggressive weighting or resampling shifts decision boundaries into the High-risk zone, causing High-risk Recall to collapse (from 63.8% down to 41.1%).
- **Decision**: No strategy reached the promotion criteria ($\text{Macro F1} \ge 0.5470$, $\text{HR Recall} \ge 0.6200$). **Candidate V4 remains the preferred, frozen model.**

## Locked Test Set Performance

| Model | Test Macro F1 | Test HR Recall | Test Mod Recall | Test Acc | Status |
|:---|:---:|:---:|:---:|:---:|:---|
| V1 | 0.5204 | 0.5808 | 0.2911 | 0.5686 | Historical Evaluation |
| V3 | 0.5188 | 0.6018 | 0.2848 | 0.5634 | Historical Evaluation |
| **V4** | **0.5288** | **0.6287** | **0.3122** | **0.5766** | **Independently Evaluated** |

> **Audit Note**: The locked test set was strictly excluded from all model selection, class balancing, and threshold tuning decisions.
