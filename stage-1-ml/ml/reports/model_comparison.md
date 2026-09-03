# Model Comparison Report -- V1 vs V3 vs V4

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Stage**: Stage 1 -- Candidate V4 Generalization-First Optimization  

---

## Cross-Validation Performance Comparison

| Model | CV Macro F1 | CV HR Recall | CV Mod F1 | CV Accuracy | Train F1 | Gap | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **V1 Tuned LightGBM** | 0.5348 | 0.6009 | 0.3312 | 0.5877 | N/A | N/A | Original baseline |
| **V3 XGB + Thresh** | 0.5427 | 0.6459 | 0.3284 | 0.5992 | 0.7763 | 0.2665 | Severe overfitting |
| **V4 V4-Conservative (W=[1.0, 1.05, 1.05])** | **0.5427** | **0.6444** | **0.3332** | **0.5889** | **0.6378** | **0.0950** | Generalization-first |

## Locked Test Set Performance

| Model | Test Macro F1 | Test HR Recall | Status |
|:---|:---:|:---:|:---|
| V1 | 0.5204 | 0.5808 | Evaluated |
| V3 | 0.5188 | 0.6018 | Evaluated |
| **V4** | **PENDING** | **PENDING** | **Awaiting Evaluation Engineer** |

> **Note**: V4 was deliberately NOT evaluated on the locked test set during ML Engineering. The Evaluation Engineer will perform this independently.
