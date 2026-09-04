# Stage 2 Exploratory Data Analysis (EDA) Module

## Overview
This module conducts comprehensive **Exploratory Data Analysis (EDA)**, statistical quality auditing, and anti-shortcut verification on the validated **Stage 2 Deep Learning Dataset v2**.

The analysis covers both modalities:
1. **Computer Vision:** 12,000 histopathology tiles ($224 \times 224 \times 3$ RGB) across benign, malignant, and inflammation classes.
2. **Longitudinal Biomarkers:** 16,012 observations across 1,000 patients tracking ctDNA VAF, CEA, CA-125, LDH, and CRP over 236 days.

---

> [!IMPORTANT]
> **Mandatory Clinical & Synthetic Data Notice:**  
> **Synthetic histopathology-like images and biomarker trajectories generated for deep-learning pipeline prototyping; not clinically validated.**  
> $$\text{Synthetic data} \ne \text{Real patient evidence} \ne \text{Clinical validation}$$  
> All findings, figures, and statistical patterns describe this synthetic research prototype dataset. Never cite downstream DL results as real clinical diagnostic accuracy.

---

## Directory Layout

```text
stage-2-dl/eda/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Central configurations, paths, palette, disclaimers
│   ├── split_eda.py              # Cohort split disjointness & comparability audits
│   ├── image_eda.py              # Pathology image checks, pixel distributions, physical audit
│   ├── shortcut_eda.py           # Anti-shortcut hypothesis testing (ANOVA, Chi2, Cramér's V)
│   ├── temporal_eda.py           # Longitudinal biomarker distributions & forecasting audits
│   ├── visualization.py          # High-resolution rendering engine for 13 diagnostic figures
│   └── run_eda.py                # Main pipeline orchestrator
├── reports/
│   ├── eda_report.md             # 21-section primary EDA report with embedded figures
│   ├── findings.md               # Severity-rated (LOW/MED/HIGH/CRITICAL) findings matrix
│   ├── image_statistics.csv      # Split- and class-wise pixel & exposure statistics
│   ├── temporal_statistics.csv   # Biomarker summary statistics, missingness, percentiles
│   └── split_statistics.csv      # Partition-level patient, image, and trajectory distributions
├── figures/                      # 13 Publication-ready diagnostic figures (PNG)
│   ├── image_class_distribution.png
│   ├── image_examples.png
│   ├── image_pixel_distribution.png
│   ├── image_brightness_by_class.png
│   ├── image_color_distribution.png
│   ├── image_stain_variation.png
│   ├── temporal_biomarker_distributions.png
│   ├── temporal_missingness.png
│   ├── temporal_observations_over_time.png
│   ├── temporal_trajectory_examples.png
│   ├── temporal_trajectory_distribution.png
│   ├── forecasting_window_distribution.png
│   └── split_comparison.png
├── tests/
│   ├── __init__.py
│   └── test_eda.py               # Automated unit tests for pipeline outputs & non-modification
├── requirements.txt              # Module dependencies
└── README.md                     # This file
```

---

## Source Data Inputs (Read-Only)

All inputs are loaded from `stage-2-dl/data-engineering/data/v2/`:
- `processed/pathology_tiles/`
- `processed/image_metadata.csv`
- `processed/biomarkers_processed.csv`
- `splits/train_patients.csv` (700 patients)
- `splits/validation_patients.csv` (150 patients)
- `splits/test_patients.csv` (150 patients)

*Note: EDA operates strictly in read-only mode and does not alter, rebalance, or overwrite source datasets.*

---

## How to Run EDA

### 1. Install Dependencies
```powershell
pip install -r stage-2-dl/eda/requirements.txt
```

### 2. Execute EDA Pipeline
```powershell
python stage-2-dl/eda/src/run_eda.py
```

### 3. Run Test Suite
```powershell
python stage-2-dl/eda/tests/test_eda.py -v
```

---

## Key EDA Findings Summary

1. **Cohort & Splits:** 1,000 patients partitioned into 700 Train / 150 Val / 150 Test with mathematically proven 0% patient overlap ($\text{Train} \cap \text{Val} = \emptyset$, $\dots$).
2. **Class Balance:** Exact 1:1:1 balance (4,000 benign, 4,000 malignant, 4,000 inflammation). No class frequency reweighting needed.
3. **Anti-Shortcut Verification:** Stroma tint ($p=0.67$), brightness ($p=0.87$), contrast ($p=0.36$), and stain variations ($p=0.83$) show zero statistically significant correlation with malignancy.
4. **Physical Tile Quality:** 100% readable $224 \times 224 \times 3$ RGB images; 0 corrupted or blank files.
5. **Measured RGB Normalization:**
   - Channel Mean: `[0.842, 0.734, 0.834]`
   - Channel Std: `[0.193, 0.229, 0.150]`
6. **Temporal Windows & Leakage Prevention:**
   - Days 0–90: Historical sequence inputs (`is_input_window = 1`, 7,229 records).
   - Days 91+: Future evaluation window (`is_input_window = 0`, 8,783 records).
   - 0 boundary violations. Targets (`future_ctDNA_30d_target`, `future_progression_trend`) are strictly forbidden from model input tensors.

---

## Handoff to Deep Learning Engineer
The dataset is certified and ready for DL modeling in `stage-2-dl/dl/`. Please consult `reports/eda_report.md` for complete hyperparameter guidance, normalization recommendations, and temporal windowing instructions.
