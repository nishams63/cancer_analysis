# Class Balance & Per-Class Performance Investigation Report

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Stage**: Stage 1 Machine Learning — Class Imbalance & Per-Class Optimization Study  
**Role**: ML Engineer  
**Date**: September 2026  
**Status**: Investigation Complete — Empirical Verdict: **Retain Candidate V4**  

---

## 1. Executive Summary

We conducted an investigation into whether class imbalance is the root cause of the weaker Moderate-risk classification performance observed in Candidate V4 (Macro F1 = 54.27%, Moderate Recall = 31.22% on locked test).

Using strictly patient-aware 5-fold cross-validation on training data with zero test-set exposure, we evaluated **12 distinct balancing strategies (B0 – B11)** encompassing natural unweighted training, fold-local balanced class weighting, targeted Moderate-class weighting grids (1.2x to 2.0x), asymmetric loss penalties, Random Over-Sampling (ROS), SMOTE, Random Under-Sampling (RUS), and out-of-fold decision-rule adjustments.

### Primary Empirical Findings:
1. **Moderate Class Is Not Rare**: Moderate-risk encounters comprise **27.63% of the training cohort** (1,935 encounters) and **27.09% of the locked test set** (474 encounters). The imbalance ratio between majority Low (53.34%) and Moderate is only **1.93 : 1.00**, indicating that Moderate-risk weakness is **not caused by sample scarcity**.
2. **The Moderate–High Trade-off**: Increasing Moderate class weights (B2–B6) successfully forced higher Moderate Recall (from 27.70% up to 73.90%), but caused **catastrophic collapse in High-risk Recall** (dropping from 63.84% down to 41.11%) and degraded overall CV Macro F1 from 53.61% down to 45.22%.
3. **SMOTE Degradation**: Consistent with warnings regarding mixed continuous and one-hot categorical feature spaces, SMOTE (B9) produced synthetic interpolations that distorted decision boundaries, yielding the lowest Moderate F1 (18.69%) and degrading Macro F1 to 50.56%.
4. **V5 Rejection Verdict**: Under the strict pre-registered decision rule requiring CV Macro F1 $\ge 0.5470$ while preserving High-risk Recall $\ge 0.6200$, **no candidate qualified to replace V4**. Candidate V4 remains the optimal, robust, and safest model.

---

## 2. Dataset Class Distribution

| Cohort | Encounters | Unique Patients | Low Risk (0) | Moderate Risk (1) | High Risk (2) | Imbalance Ratio (Low : Mod : High) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Master Dataset** | 8,754 | 6,000 | 4,678 (53.44%) | 2,409 (27.52%) | 1,667 (19.04%) | 2.81 : 1.45 : 1.00 |
| **Training Set** | 7,004 | 4,800 | 3,736 (53.34%) | 1,935 (27.63%) | 1,333 (19.03%) | 2.80 : 1.45 : 1.00 |
| **Locked Test Set** *(Report Only)* | 1,750 | 1,200 | 942 (53.83%) | 474 (27.09%) | 334 (19.09%) | 2.82 : 1.42 : 1.00 |

### 5-Fold StratifiedGroupKFold Validation Distribution
Validation fold splits were generated using `StratifiedGroupKFold(n_splits=5)` on `patient_id` to guarantee zero patient leakage:
- **Fold 0**: 1,401 encounters (Low: 748 [53.39%], Mod: 387 [27.62%], High: 266 [18.99%])
- **Fold 1**: 1,401 encounters (Low: 747 [53.32%], Mod: 387 [27.62%], High: 267 [19.06%])
- **Fold 2**: 1,401 encounters (Low: 747 [53.32%], Mod: 387 [27.62%], High: 267 [19.06%])
- **Fold 3**: 1,401 encounters (Low: 747 [53.32%], Mod: 387 [27.62%], High: 267 [19.06%])
- **Fold 4**: 1,400 encounters (Low: 747 [53.36%], Mod: 387 [27.64%], High: 266 [19.00%])

---

## 3. Experimental Strategies Benchmarked (B0 – B11)

All experiments were trained on the Candidate V4 LightGBM backbone (`depth=3`, `n_estimators=80`, `learning_rate=0.10`, `min_child_samples=30`, `reg_alpha=0.5`, `reg_lambda=1.0`, 41 features). Preprocessing and resampling were fitted strictly on each training fold.

```
B0:  No Balancing (class_weight=None)
B1:  V4 Balanced Class Weights (fold-local N_fold / (3 * N_{c}))
B2:  Moderate-Weight 1.2x (fold-local balanced * 1.2 for class 1)
B3:  Moderate-Weight 1.4x (fold-local balanced * 1.4 for class 1)
B4:  Moderate-Weight 1.6x (fold-local balanced * 1.6 for class 1)
B5:  Moderate-Weight 1.8x (fold-local balanced * 1.8 for class 1)
B6:  Moderate-Weight 2.0x (fold-local balanced * 2.0 for class 1)
B7:  Asymmetric Misclassification Penalty (targeted High/Moderate protection)
B8:  Random Over-Sampling (ROS inside fold train)
B9:  SMOTE (k=5 interpolation inside fold train)
B10: Random Under-Sampling (RUS inside fold train)
B11: Out-Of-Fold Decision Rule Adjustment (calibrated probability multipliers)
```

---

## 4. Benchmark Results Table

| Strategy ID | Strategy Name | CV Macro F1 (± std) | Train/Val Gap | HR Recall | Mod F1 | Mod Recall | Low F1 | High F1 | Accuracy | Mod $\rightarrow$ Low Errors | High $\rightarrow$ Low Errors | V5 Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0** | No Balancing (Unweighted) | 0.5078 ± 0.010 | 0.0865 | 0.5229 | 0.2033 | 0.1421 | 0.7605 | 0.5596 | 0.6179 | 1,319 | 414 | REJECT |
| **B1** | V4 Balanced Class Weights | **0.5361 ± 0.008** | **0.1025** | **0.6384** | **0.3106** | **0.2770** | **0.7256** | **0.5722** | **0.5908** | **880** | **215** | **BASELINE** |
| **B2** | Moderate-Weight 1.2x | 0.5431 ± 0.011 | 0.1021 | 0.5882 | 0.3716 | 0.4052 | 0.6869 | 0.5708 | 0.5691 | 718 | 167 | REJECT (HR < 0.62) |
| **B3** | Moderate-Weight 1.4x | 0.5306 ± 0.009 | 0.1034 | 0.5319 | 0.4053 | 0.5199 | 0.6340 | 0.5526 | 0.5380 | 559 | 128 | REJECT (HR < 0.62) |
| **B4** | Moderate-Weight 1.6x | 0.5132 ± 0.007 | 0.0959 | 0.4944 | 0.4258 | 0.6181 | 0.5726 | 0.5412 | 0.5083 | 419 | 95 | REJECT (HR < 0.62) |
| **B5** | Moderate-Weight 1.8x | 0.4852 ± 0.009 | 0.0947 | 0.4516 | 0.4304 | 0.6863 | 0.5046 | 0.5206 | 0.4746 | 324 | 71 | REJECT (HR < 0.62) |
| **B6** | Moderate-Weight 2.0x | 0.4522 ± 0.008 | 0.0965 | 0.4111 | 0.4304 | 0.7390 | 0.3957 | 0.5305 | 0.4332 | 250 | 43 | REJECT (HR < 0.62) |
| **B7** | Asymmetric Misclassification Penalty | 0.5319 ± 0.011 | 0.1004 | 0.5964 | 0.3799 | 0.4543 | 0.6480 | 0.5678 | 0.5523 | 608 | 136 | REJECT (HR < 0.62) |
| **B8** | Random Over-Sampling (ROS) | 0.5397 ± 0.006 | 0.0968 | 0.6227 | 0.3213 | 0.2873 | 0.7225 | 0.5753 | 0.5898 | 891 | 228 | REJECT (< 0.5470) |
| **B9** | SMOTE (k=5 Interpolation) | 0.5056 ± 0.014 | 0.0718 | 0.5844 | 0.1869 | 0.1256 | 0.7645 | 0.5654 | 0.6239 | 1,282 | 388 | REJECT (Distortion) |
| **B10** | Random Under-Sampling (RUS) | 0.5300 ± 0.010 | 0.0916 | 0.6459 | 0.3118 | 0.2837 | 0.7099 | 0.5683 | 0.5794 | 853 | 227 | REJECT (< 0.5470) |
| **B11** | OOF Decision Rule Adjustment | 0.5356 ± 0.007 | 0.1035 | 0.6399 | 0.3196 | 0.2966 | 0.7208 | 0.5663 | 0.5867 | 840 | 201 | REJECT (< 0.5470) |

*(Note: Full Candidate V4 with out-of-fold calibration in train.py achieved CV Macro F1 = 0.5427 ± 0.0041, HR Recall = 0.6444 ± 0.0115, Moderate F1 = 0.3332 ± 0.0061).*

---

## 5. In-Depth Technical Analysis

### A. Why Moderate-Weighting Fails (The Intermediate Class Dilemma)
In multi-class medical diagnosis, **Moderate Toxicity Risk** is an intermediate condition situated directly between **Low Risk** and **High Risk** along the clinical spectrum. It does not occupy a distinct, isolated cluster in feature space.

When we increase the class weight on Moderate (B2 $\rightarrow$ B6):
- The model shifts decision thresholds toward Moderate to minimize loss.
- While Moderate Recall rises from 27.7% to 73.9%, the model accomplishes this by **classifying borderline High-risk patients as Moderate**.
- High-risk Recall collapses from **63.84% down to 41.11%** (a 22.7% absolute drop!). In an oncology decision-support tool, missing more than half of severe toxicities is clinically unacceptable.
- Simultaneously, Low-risk F1 collapses from 72.56% to 39.57% due to massive false-positive Moderate assignments.
- Consequently, overall CV Macro F1 drops monotonically from **54.31% down to 45.22%**.

### B. Why SMOTE Severely Degraded Performance
SMOTE (B9) assumes a locally convex Euclidean space where linear combinations of minority samples remain valid. However:
1. The preprocessed feature space has 113 dimensions, containing one-hot encoded categorical indicators (`cancer_type`, `mutation_primary`, `drug_name`, etc.).
2. Linear interpolation between one-hot vectors produces continuous fractional representations (e.g. `cancer_type_NSCLC = 0.4, Breast = 0.6`) that do not correspond to any real clinical patient.
3. These synthetic boundary artifacts diluted the tree split criteria, resulting in a **Moderate F1 of only 18.69%** (worse than unweighted training) and the lowest overall Macro F1 (50.56%).

### C. The Unweighted Baseline Confirms Imbalance Effect
Strategy B0 (unweighted) confirmed that doing nothing is also suboptimal:
- The majority Low class (53.3%) dominated gradient updates.
- High $\rightarrow$ Low errors spiked to **414** (nearly double V4's 215).
- Moderate $\rightarrow$ Low errors spiked to **1,319** (68% of all Moderate cases misclassified as Low).
- Balanced class weighting (B1/V4) remains essential, but further aggressive weighting harms the global objective.

---

## 6. Leakage Prevention Protocol

All experiments adhered strictly to the project's zero-leakage constraints:
1. **Patient-Level Isolation**: Encounters from the same patient were strictly isolated within either training or validation folds. Overlap was verified to be exactly 0 across all 5 folds.
2. **Fold-Internal Transformations**:
   - Feature engineering applied row-level clinical formulas only.
   - Missing value imputation and scaling were fitted on fold training data only.
   - Categorical encodings were fitted on fold training data only.
   - Resamplers (ROS, SMOTE, RUS) transformed only the fold training subset.
3. **Fold-Local Weight Computation**: Class weights for B1–B7 were computed dynamically from each fold's empirical class counts, preventing any cross-fold distributional leakage.
4. **Locked Test Set Untouched**: The locked test set was completely excluded from all tuning, hyperparameter search, threshold calibration, and candidate selection.

---

## 7. Final Decision Verdict & Interpretation

### Pre-Registered V5 Promotion Criteria:
1. CV Macro F1 $\ge 0.5470$ (+0.0043 margin over V4) $\rightarrow$ **NO strategy achieved this**.
2. CV High-risk Recall $\ge 0.6200$ $\rightarrow$ B2 reached Macro F1 0.5431 but HR Recall dropped to **0.5882 (FAILED)**.
3. Fold stability ($\sigma \le 0.0080$) $\rightarrow$ B2 std was 0.0110 (FAILED).
4. Train/Val Gap $\le 0.1200$ $\rightarrow$ Satisfied by all.

### Official Verdict:
**Candidate V5 was not promoted. Candidate V4 remains the preferred Stage 1 ML model.**

### Required Interpretation & Scientific Synthesis:
The results do not support class scarcity as the primary explanation for the weaker Moderate-risk performance. Moderate risk represents approximately 27.6% of training encounters and therefore is not an extremely rare class. 

The training-data evidence demonstrates that increasing Moderate weighting improved Moderate-risk sensitivity (raising Moderate Recall up to 73.90%), but this came at the severe expense of High-Risk Recall (which collapsed down to 41.11%) and overall predictive generalization behavior (Macro F1 dropped to 45.22%). Because missing High-risk toxicity encounters introduces critical clinical risks, trading High-risk Recall to artificially inflate Moderate Recall is unacceptable.

Therefore, the completed B0–B11 cross-validation experiments do not justify replacing V4 with a balancing-based V5. Candidate V4 is retained based on patient-aware CV evidence as the official Stage 1 ML model.
