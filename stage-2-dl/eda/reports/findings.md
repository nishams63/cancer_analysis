# Stage 2 Deep Learning Dataset — Exploratory Data Analysis Findings Matrix

> [!IMPORTANT]
> **SYNTHETIC DATA PROVENANCE & REGULATORY NOTICE:**  
> **Synthetic histopathology-like images and biomarker trajectories generated for deep-learning pipeline prototyping; not clinically validated.**  
> $$\text{Synthetic data} \ne \text{Real patient evidence} \ne \text{Clinical validation}$$  
> All findings documented below pertain strictly to the synthetic research benchmark dataset.

---

## Findings Severity Summary

| Category | Finding Description | Severity | Status | Action / Recommendation for DL Engineer |
|---|---|---|---|---|
| **Patient Leakage** | Patient-level disjoint sets strictly verified across 1,000 patients | **LOW** | `PASS` | No patient overlap across train/val/test splits. Safe for training. |
| **Class Balance** | Pathology classes are exactly balanced (4,000 benign, 4,000 malignant, 4,000 inflammation) | **LOW** | `PASS` | No class weighting or loss reweighting needed based on class frequency alone. |
| **Synthetic Shortcuts** | Color temperature, brightness, contrast, and H&E stain variations are randomized across classes | **LOW** | `PASS` | High p-values (ANOVA/Chi2 > 0.05) and negligible effect sizes demonstrate independence. |
| **Image Resolution** | Uniform 224x224x3 RGB image tiles across all 12,000 samples | **LOW** | `PASS` | Standard input dimensions. Direct feed into CNN backbones. |
| **Pixel Normalization** | True dataset pixel mean is [0.842, 0.734, 0.834] and std is [0.193, 0.229, 0.150] | **MEDIUM** | `INFORMATIONAL` | Consider comparing custom normalization vs standard ImageNet normalization in ablation studies. |
| **Temporal Monotonicity** | Strict monotonic ordering of clinical timepoints with zero timestamp inversions | **LOW** | `PASS` | Sequence timestamps and delta days are chronologically valid. |
| **Controlled Missingness** | Biomarker missingness rates stay strictly within 5.3% – 5.6% target range | **LOW** | `PASS` | Mask columns (`_missing`) match NaNs with 100% precision. Use masking or forward-fill. |
| **Forecasting Boundary** | Strict segregation between historical window (<=90d) and future prediction (>90d) | **LOW** | `PASS` | Zero boundary violations. Exactly 7,229 input and 8,783 forecast observations. |
| **Temporal Target Leakage** | `future_ctDNA_30d_target` and `future_progression_trend` isolated from input features | **HIGH** | `AUDITED` | **CRITICAL RULE**: Do NOT feed future targets as input features into LSTMs/Transformers. |
| **Temporal Irregularity** | Observation intervals vary between 11 and 16 days (mean ~14 days) | **LOW** | `INFORMATIONAL` | Model architectures should utilize `delta_days` or continuous-time positional embeddings. |

---

## Detailed Findings Analysis

### 1. Dataset Strengths
1. **Perfect Class Parity:** The visual pathology dataset features exactly 4,000 tiles per class across all 1,000 patients, completely eliminating majority class bias.
2. **Deterministic Anti-Shortcut Decoupling:** Empirical hypothesis testing (ANOVA & Chi-Square) confirmed that stroma background temperature, exposure/brightness factor, contrast, and stain variations have zero statistically significant correlation with malignancy ($p > 0.35$).
3. **Comprehensive Temporal Coverage:** Every patient has a longitudinal span exceeding 180 days with an average of 16.01 clinical assessments, providing ample depth for recurrent and attention-based models.
4. **Complete Split Independence:** Zero patient overlap ($Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$, $Val \cap Test = \emptyset$).

### 2. Potential Leakage Risks & Safeguards
- **Severity: HIGH (Mitigated by Protocol)**
- **Evidence:** The processed biomarker dataset includes ground truth targets (`future_ctDNA_30d_target`, `future_progression_trend`) on the same rows for historical observations to enable supervised sequence training.
- **Impact:** If an unvetted dataloader feeds all CSV columns into an LSTM or Transformer, the model will achieve artificial 100% predictive accuracy via direct target memorization.
- **Enforced DL Recommendation:** Models must be restricted to `is_input_window == 1` and trained strictly on `input_features` (`timepoint_index`, `days_from_baseline`, `delta_days`, biomarker values, and missingness masks).

### 3. Outlier and Boundary Analysis
- **Negative Values:** 0 negative values found across all 16,012 biomarker observations.
- **Infinities:** 0 infinite values detected.
- **Corrupted / Blank Tiles:** 0 blank, 0 corrupted, and 0 out-of-bounds dimension tiles observed in the physical file audit.
