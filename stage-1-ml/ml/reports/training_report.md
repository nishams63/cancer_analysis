# Candidate V4 Training Report -- Generalization-First Optimization

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Stage**: Stage 1 -- ML Candidate V4  
**Timestamp**: 2026-09-03 11:01:55  
**Selected Candidate**: **V4-Conservative (W=[1.0, 1.05, 1.05])**  
**Target**: `toxicity_risk` (`Low`, `Moderate`, `High`)  

---

## 1. Executive Summary

Candidate V4 was developed with a **generalization-first** objective, addressing the severe train/validation gap (0.2665) observed in V3. Through systematic diagnostics, the primary overfitting sources were identified as:
1. **Model complexity** (V3 XGBoost depth=5 with 200 trees)
2. **Aggressive threshold optimization** (V3 multipliers [1.0, 1.8, 2.7])

V4 uses heavily regularized LightGBM (depth 3-4) with conservative or no decision thresholds.

> **IMPORTANT**: Final locked-test evaluation has NOT been performed. The Evaluation Engineer will independently evaluate the frozen V4 candidate.

---

## 2. Data Leakage Verification

| Check | Status |
|:---|:---:|
| `patient_id` excluded from features | PASSED |
| `encounter_id` excluded from features | PASSED |
| `observation_date` excluded from features | PASSED |
| `treatment_response` excluded from features | PASSED |
| `toxicity_risk` (target) excluded from features | PASSED |
| Preprocessing fitted inside each CV fold | PASSED |
| Zero patient overlap across all 5 CV folds | PASSED |
| Feature engineering uses row-level transforms only | PASSED |

---

## 3. Feature Ablation Results

| Feature Set | # Features | Val Macro F1 | F1 Std | Train/Val Gap | Val HR Recall | Val Mod F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1A. Baseline (30 feats)** | 30 feats | 0.5341 | 0.0070 | 0.0954 | 0.6369 | 0.3093 |
| **1B. Original (36 feats)** | 36 feats | 0.5360 | 0.0052 | 0.0969 | 0.6392 | 0.3073 |
| **1C. Expanded (41 feats)** | 41 feats | 0.5417 | 0.0040 | 0.0957 | 0.6429 | 0.3211 |

**Selected Feature Set**: 1C. Expanded (41 feats)

---

## 4. Class Weight Comparison

| Strategy | Val Macro F1 | F1 Std | Train/Val Gap | Val HR Recall | Val Mod F1 | Val Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **2A. Unweighted (None)** | 0.5068 | 0.0088 | 0.0875 | 0.5146 | 0.2055 | 0.6175 |
| **2B. Balanced (sklearn)** | 0.5417 | 0.0040 | 0.0957 | 0.6429 | 0.3211 | 0.5951 |
| **2C. Moderate Boost (0.7/1.3/1.3)** | 0.5496 | 0.0130 | 0.0985 | 0.5259 | 0.3568 | 0.5995 |
| **2D. Mod-Focused (0.6/1.5/1.4)** | 0.5387 | 0.0100 | 0.1002 | 0.5011 | 0.4053 | 0.5543 |
| **2E. V3 Aggressive (0.6/1.6/1.5)** | 0.5297 | 0.0067 | 0.1033 | 0.4981 | 0.4086 | 0.5383 |
| **2F. Heavy Mod (0.5/1.8/1.6)** | 0.4834 | 0.0084 | 0.0846 | 0.4936 | 0.4308 | 0.4656 |

**Selected Class Weight**: 2B. Balanced (sklearn) = `balanced`

---

## 5. Regularization Sweep

| Config | Train F1 | Val F1 | F1 Std | Gap | Val HR | Val Mod F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **3A. d=3,n=60,lr=0.12,mcs=30,L1=0.5,L2=1.0** | 0.6288 | 0.5386 | 0.0043 | 0.0902 | 0.6399 | 0.3135 |
| **3B. d=3,n=80,lr=0.10,mcs=30,L1=0.5,L2=1.0** | 0.6374 | 0.5417 | 0.0040 | 0.0957 | 0.6429 | 0.3211 |
| **3C. d=3,n=100,lr=0.08,mcs=30,L1=0.5,L2=1.0** | 0.6356 | 0.5352 | 0.0055 | 0.1004 | 0.6362 | 0.3090 |
| **3D. d=3,n=100,lr=0.10,mcs=25,L1=0.3,L2=0.8** | 0.6608 | 0.5352 | 0.0098 | 0.1256 | 0.6347 | 0.3169 |
| **3E. d=3,n=80,lr=0.10,mcs=35,L1=0.8,L2=1.5** | 0.6348 | 0.5396 | 0.0055 | 0.0952 | 0.6414 | 0.3153 |
| **3F. d=4,n=80,lr=0.10,mcs=30,L1=0.5,L2=1.0** | 0.6958 | 0.5356 | 0.0044 | 0.1602 | 0.6257 | 0.3184 |
| **3G. d=4,n=100,lr=0.08,mcs=30,L1=0.5,L2=1.5** | 0.6964 | 0.5376 | 0.0082 | 0.1588 | 0.6264 | 0.3187 |
| **3H. d=4,n=120,lr=0.08,mcs=25,L1=0.3,L2=1.0** | 0.7217 | 0.5429 | 0.0064 | 0.1788 | 0.6249 | 0.3311 |
| **3I. d=4,n=100,lr=0.10,mcs=35,L1=0.8,L2=2.0** | 0.6982 | 0.5413 | 0.0072 | 0.1569 | 0.6287 | 0.3287 |

**Selected Regularization**: 3B. d=3,n=80,lr=0.10,mcs=30,L1=0.5,L2=1.0

---

## 6. Decision Rule Comparison

| Decision Rule | Val Macro F1 | F1 Std | Gap | Val HR Recall | Val Mod F1 | Val Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **4B. Conservative Thresh** | 0.5427 | 0.0041 | 0.0950 | 0.6444 | 0.3332 | 0.5889 |
| **4C. Moderate Thresh** | 0.5417 | 0.0034 | 0.0960 | 0.6467 | 0.3313 | 0.5885 |
| **4D. V3-Aggressive Thresh** | 0.4477 | 0.0115 | 0.0860 | 0.7645 | 0.3368 | 0.4453 |

---

## 7. Final Model Selection

### Multi-Criteria Scoring

| Candidate | Score | Val F1 | F1 Std | Gap | HR Recall | Mod F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **V4-Conservative (W=[1.0, 1.05, 1.05])** | 0.6374 | 0.5427 | 0.0041 | 0.0950 | 0.6444 | 0.3332 |
| **V4-Moderate (W=[1.0, 1.05, 1.06])** | 0.6370 | 0.5417 | 0.0034 | 0.0960 | 0.6467 | 0.3313 |
| **V4-Argmax (3B. d=3,n=80,lr=0.10,mcs=30,L1=0.5,L2=1.0)** | 0.6362 | 0.5417 | 0.0040 | 0.0957 | 0.6429 | 0.3211 |
| **V4-Alt (3H. d=4,n=120,lr=0.08,mcs=25,L1=0.3,L2=1.0)** | 0.6319 | 0.5429 | 0.0064 | 0.1788 | 0.6249 | 0.3311 |
| **V4-Alt (3I. d=4,n=100,lr=0.10,mcs=35,L1=0.8,L2=2.0)** | 0.6309 | 0.5413 | 0.0072 | 0.1569 | 0.6287 | 0.3287 |

### Selected: **V4-Conservative (W=[1.0, 1.05, 1.05])**

| Metric | V1 Baseline | V3 (XGB+Thresh) | **V4 Selected** |
|:---|:---:|:---:|:---:|
| CV Macro F1 | 0.5348 | 0.5427 | **0.5427** |
| CV HR Recall | 0.6009 | 0.6459 | **0.6444** |
| CV Mod F1 | 0.3312 | 0.3284 | **0.3332** |
| Train Macro F1 | N/A | 0.7763 | **0.6378** |
| Train/Val Gap | N/A | 0.2665 | **0.0950** |
| Decision Rule | argmax | W=[1.0, 1.8, 2.7] | **W=[1.0, 1.05, 1.05]** |

---

## 8. Why V4 Should Generalize Better

1. **Reduced train/val gap**: V3 gap = 0.2665, V4 gap = 0.0950 -- the model is no longer memorizing training noise.
2. **Shallow trees with strong regularization**: depth 3-4, L1/L2 penalties, large min_child_samples prevent over-specialization.
3. **Conservative decision rules**: Unlike V3's aggressive [1.0, 1.8, 2.7] multipliers, V4 uses conservative thresholds.
4. **Stable fold performance**: F1 std = 0.0041 indicates consistent performance across patient groups.

## 9. Why V4 Is Not Excessively Underfit

1. **Train F1 (0.6378) > Val F1 (0.5427)**: The model learns signal from training data.
2. **Val F1 (0.5427) >> random baseline (~0.33)**: The model captures real discriminative patterns.
3. **HR Recall (0.6444)**: High-risk patients identified at clinically useful rates.
4. **The gap is controlled, not zero**: A small positive gap indicates the model uses training data effectively without memorizing.

---

## 10. Feature Importances (Top 15)

| Rank | Feature | Importance |
|:---:|:---|:---:|
| 1 | `previous_toxicity_grade` | 107.0000 |
| 2 | `drug_dose` | 101.0000 |
| 3 | `hemoglobin` | 73.0000 |
| 4 | `organ_impairment_index` | 67.0000 |
| 5 | `comorbidity_age_interaction` | 63.0000 |
| 6 | `white_blood_cell_count` | 59.0000 |
| 7 | `cancer_stage_Stage IV` | 58.0000 |
| 8 | `previous_adverse_event_True` | 56.0000 |
| 9 | `platelet_count` | 55.0000 |
| 10 | `diastolic_bp` | 52.0000 |
| 11 | `ctdna_level` | 50.0000 |
| 12 | `cumulative_treatment_load` | 50.0000 |
| 13 | `tumor_marker_level` | 50.0000 |
| 14 | `inflammation_marker` | 49.0000 |
| 15 | `liver_function_marker` | 47.0000 |

---

## 11. Artifacts for Evaluation Engineer

| Artifact | Path |
|:---|:---|
| Model | `models/best_model/model.joblib` |
| Preprocessor | `artifacts/preprocessor/preprocessor.joblib` |
| Target Mapping | `artifacts/encoders/target_mapping.json` |
| V4 Config | `results/v4_candidate_config.json` |
| Feature Importance | `results/feature_importance.csv` |
| Training Report | `reports/training_report.md` |

**Reproducible command**: `py stage-1-ml/ml/src/train.py`

> **NOTE**: The Evaluation Engineer must perform the final locked-test evaluation independently.
