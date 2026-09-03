# Model Comparison Report -- Stage 1 ML

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Stage**: Stage 1 Machine Learning — Multi-Candidate Evaluation & Class Balancing Investigation  
**Role**: ML Engineer  
**Status**: Official Decision — **Candidate V4 Retained; Candidate V5 Not Promoted**  

---

## 1. Candidate Model Benchmark Summary

| Model | Architecture | CV Macro F1 | CV HR Recall | CV Mod F1 | CV Accuracy | Train Macro F1 | Train/Val Gap | Status / Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **V1 Tuned LightGBM** | LightGBM (depth 6, 200 trees) | 0.5348 | 0.6009 | 0.3312 | 0.5877 | N/A | N/A | Superseded baseline |
| **V3 XGB + Thresh** | XGBoost (depth 5, W=[1.0, 1.8, 2.7]) | 0.5427 | 0.6459 | 0.3284 | 0.5992 | 0.7763 | 0.2665 | Rejected: Severe overfitting |
| **V4 Regularized LightGBM** | LightGBM (depth 3, W=[1.0, 1.05, 1.05]) | **0.5427 ± 0.0041** | **0.6444 ± 0.0115** | **0.3332 ± 0.0061** | **0.5889 ± 0.0092** | **0.6378** | **0.0950 ± 0.0075** | **OFFICIAL PREFERRED MODEL (FROZEN)** |

---

## 2. Class Balance & Per-Class Performance Investigation

### 2.1 Motivation
In Candidate V4, Moderate-risk classification performance was weaker than Low-risk and High-risk performance (CV Moderate Recall ~32.0%, test Moderate Recall ~31.2%). An investigation was conducted to determine whether class imbalance is the primary cause of this weakness and whether targeted class weighting or resampling could justify promoting a Candidate V5.

### 2.2 Training Class Distribution
Analysis of the 7,004 patient encounters in the training set (4,800 unique patients):
- **Low Risk (0)**: 3,736 encounters (**53.34%**)
- **Moderate Risk (1)**: 1,935 encounters (**27.63%**)
- **High Risk (2)**: 1,333 encounters (**19.03%**)
- *Imbalance Ratio*: $2.80 : 1.45 : 1.00$

The results do not support class scarcity as the primary explanation for the weaker Moderate-risk performance. Moderate risk constitutes approximately 27.6% of training encounters and is not an extremely rare class.

### 2.3 Patient-Aware Cross-Validation Methodology
All balancing strategies were evaluated using 5-fold `StratifiedGroupKFold` grouped on `patient_id` within the training set:
- **Zero Patient Overlap**: Exactly 0 patients shared between fold training and validation splits.
- **Fold-Internal Balancing**: All resampling (ROS, SMOTE, RUS) and weight derivations were computed strictly inside fold training subsets.
- **Untouched Validation & Locked Test**: Validation folds and the locked test set (1,750 encounters, 1,200 patients) remained completely untouched with zero synthetic samples or tuning exposure.

### 2.4 B0–B11 Benchmark Table

The following empirical results were obtained across the 12 evaluated strategies:

| Strategy ID | Strategy Name | CV Macro F1 | CV HR Recall | CV Mod F1 | CV Mod Recall | Train/Val Gap | Fold Std ($\sigma$) | Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **B0** | No Balancing (Unweighted) | 0.5078 ± 0.010 | 0.5229 | 0.2033 | 0.1421 | 0.0865 | 0.0101 | Rejected: Majority domination |
| **B1** | V4 Balanced Class Weights | **0.5361 ± 0.008** | **0.6384** | **0.3106** | **0.2770** | **0.1025** | **0.0075** | Optimal balanced baseline (without OOF mult) |
| **B2** | Moderate-Weight 1.2x | 0.5431 ± 0.011 | 0.5882 | 0.3716 | 0.4052 | 0.1021 | 0.0110 | Rejected: High-Risk Recall < 0.6200 |
| **B3** | Moderate-Weight 1.4x | 0.5306 ± 0.009 | 0.5319 | 0.4053 | 0.5199 | 0.1034 | 0.0088 | Rejected: Severe High-Risk degradation |
| **B4** | Moderate-Weight 1.6x | 0.5132 ± 0.007 | 0.4944 | 0.4258 | 0.6181 | 0.0959 | 0.0068 | Rejected: Severe High-Risk degradation |
| **B5** | Moderate-Weight 1.8x | 0.4852 ± 0.009 | 0.4516 | 0.4304 | 0.6863 | 0.0947 | 0.0088 | Rejected: Severe High-Risk degradation |
| **B6** | Moderate-Weight 2.0x | 0.4522 ± 0.008 | 0.4111 | 0.4304 | 0.7390 | 0.0965 | 0.0077 | Rejected: Severe High-Risk degradation |
| **B7** | Asymmetric Loss Penalty | 0.5319 ± 0.011 | 0.5964 | 0.3799 | 0.4543 | 0.1004 | 0.0110 | Rejected: High-Risk Recall < 0.6200 |
| **B8** | Random Over-Sampling (ROS) | 0.5397 ± 0.006 | 0.6227 | 0.3213 | 0.2873 | 0.0968 | 0.0060 | Rejected: Below V5 promotion threshold |
| **B9** | SMOTE (k=5 Interpolation) | 0.5056 ± 0.014 | 0.5844 | 0.1869 | 0.1256 | 0.0718 | 0.0140 | Rejected: Poor performance & categorical distortion |
| **B10** | Random Under-Sampling (RUS) | 0.5300 ± 0.010 | 0.6459 | 0.3118 | 0.2837 | 0.0916 | 0.0099 | Rejected: Information loss |
| **B11** | OOF Decision Rule Adjustment | 0.5356 ± 0.007 | 0.6399 | 0.3196 | 0.2966 | 0.1035 | 0.0072 | Rejected: Below V5 promotion threshold |

*(Reference: Full Candidate V4 with out-of-fold calibration reached CV Macro F1 = 0.5427 ± 0.0041, HR Recall = 0.6444 ± 0.0115, Moderate F1 = 0.3332 ± 0.0061).*

### 2.5 Moderate-Risk Performance Analysis
Increasing Moderate class weighting from 1.2x up to 2.0x (B2–B6) successfully increased Moderate Recall from 27.70% to 73.90% and Moderate F1 from 0.3106 to 0.4304. However, because Moderate risk is clinically intermediate between Low and High, the model achieved higher sensitivity by predicting Moderate for borderline Low-risk and borderline High-risk cases.

### 2.6 High-Risk Trade-Off Analysis
The increase in Moderate Recall came at the direct expense of High-risk detection:
- In B1 (balanced baseline), High-risk Recall was **0.6384**.
- In B2 (1.2x weight), High-risk Recall dropped to **0.5882** (-5.0%).
- In B3 (1.4x weight), High-risk Recall dropped to **0.5319** (-10.7%).
- In B6 (2.0x weight), High-risk Recall collapsed to **0.4111** (-22.7%).

Missing high-risk toxicity encounters in oncology decision support presents serious clinical safety concerns. Sacrificing High-risk Recall to inflate Moderate Recall is clinically and methodologically unacceptable.

### 2.7 Stability & Overfitting Analysis
- **Fold Stability**: B2 exhibited higher fold variance ($\sigma = 0.0110$) compared to V4 ($\sigma = 0.0041$).
- **SMOTE Distortion**: SMOTE (B9) exhibited unstable folds ($\sigma = 0.0140$) and generated synthetic points in a 113-dimensional mixed continuous/one-hot feature space that blurred clinical boundaries, yielding the lowest Moderate F1 (0.1869).
- **Train/Val Gaps**: Remained well-controlled ($\le 0.1035$) across all strategies, confirming regularization held, but predictive balance deteriorated.

### 2.8 Pre-Registered V5 Promotion Criteria
To qualify as Candidate V5, an approach was required to achieve:
1. CV Macro F1 $\ge 0.5470$ (+0.0043 meaningful margin over V4)
2. High-Risk Recall $\ge 0.6200$
3. Meaningful Moderate-risk improvement
4. Train-validation Macro F1 gap $\le 0.1200$
5. Fold F1 standard deviation $\le 0.0080$
6. Strictly zero patient leakage

### 2.9 Final Decision
**Candidate V5 NOT PROMOTED — Candidate V4 RETAINED.**

No balancing strategy satisfied all promotion criteria. The class-balance experiments improved Moderate-risk sensitivity in some configurations, but the gains consistently involved unacceptable losses in High-Risk Recall, Macro F1, stability, or generalization; therefore V4 remains the strongest overall Stage 1 candidate.

### 2.10 Why V4 is Retained
Candidate V4 provides the optimal empirical Pareto frontier on training data:
- High generalization stability: Train/Val gap = `0.0950 ± 0.0075`, fold std = `0.0041`.
- Preserved safety: High-risk Recall = `0.6444 ± 0.0115`.
- Highest overall CV Macro F1: `0.5427 ± 0.0041`.
- Clean decision boundaries without synthetic categorical distortion.

---

## 3. Historical Locked Test Set Performance (Reference Only)

The locked test set (1,750 encounters, 1,200 unique patients) was strictly reserved for the independent Evaluation Engineer and was **not** used during development, class balancing, or model selection.

| Model | Test Macro F1 | Test HR Recall | Test Mod Recall | Test Acc | High $\rightarrow$ Low Errors | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| V1 Tuned LightGBM | 0.5204 | 0.5808 | 0.2911 | 0.5686 | 62 | Historical Baseline |
| V3 XGB + Thresh | 0.5188 | 0.6018 | 0.2848 | 0.5634 | 49 | Historical (Overfitted) |
| **V4 Regularized LightGBM** | **0.5288** | **0.6287** | **0.3122** | **0.5766** | **50** | **Evaluated Candidate** |
