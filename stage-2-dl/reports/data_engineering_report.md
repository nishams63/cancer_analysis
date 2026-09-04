# Stage 2 Deep Learning Dataset (v2) — Data Engineering & Audit Report

**Status:** `PASS`  
**Audit Timestamp:** `2026-09-04T22:43:17.270985`  
**Cohort Scale:** 1,000 unique patients | 12,000 pathology tiles | 16012 temporal observations  

> [!IMPORTANT]
> **MANDATORY CLINICAL & SYNTHETIC DATA NOTICE:**  
> **Synthetic data created for deep-learning research and pipeline prototyping. These data are not real patient records and are not clinically validated.**  
> $$\text{Synthetic data} \ne \text{Real patient evidence} \ne \text{Clinical validation}$$  
> This dataset has been engineered deterministically for machine learning model development, dataloader benchmarking, and multi-modal pipeline architecture validation. Never present downstream model evaluations as real clinical efficacy.

---

## 1. Executive Quality Summary

| Metric / Requirement | Target / Benchmark | Observed v2 Metric | Status |
|---|---|---|---|
| **Patient Cohort Size** | Exactly 1,000 unique patients | 1000 unique patients | `PASSED` |
| **Patient Partitioning** | 700 Train / 150 Val / 150 Test | 700 / 150 / 150 | `PASSED` |
| **Cross-Split Patient Leakage** | Strict 0% overlap | 0 overlapping patients across all splits | `PASSED` |
| **Pathology Tile Count** | Exactly 12,000 tiles | 12000 valid 224x224x3 RGB tiles | `PASSED` |
| **Class Balance** | ~33% benign, ~33% malignant, ~33% inflammation | Benign: 4000, Malignant: 4000, Inflam: 4000 | `PASSED` |
| **Anti-Shortcut Engineering** | Randomized stroma temp, exposure, stain gains | Balanced across warm, neutral, and cool stroma | `PASSED` |
| **Temporal Observations** | 14,000–18,000 observations | 16012 observations (16.01 / patient) | `PASSED` |
| **Temporal Monotonicity** | Strict $t_i > t_{i-1}$ with zero inversions | 0 inversions, 0 duplicate timestamps | `PASSED` |
| **Longitudinal Span** | Spans at least 180 days per patient | 100% of patients span $\ge 180$ days | `PASSED` |
| **Physiological Boundaries** | Non-negative biomarker values | 0 negative values across all measurements | `PASSED` |
| **Controlled Missingness** | 3.0% – 8.0% missingness with indicator masks | All biomarkers within 3–8% target range | `PASSED` |
| **Forecasting Structure** | Days 0–90 Input, Days 91+ Prediction | Strict window segregation with 30-day targets | `PASSED` |
| **Duplicate Image Audit** | SHA-256 hash collision audit | 0 duplicates (0.00%) | `PASSED` |
| **Augmentation Confinement** | Applied exclusively to Training split | Validation and Test strictly unaugmented | `PASSED` |

---

## 2. Patient Partitioning & Zero-Leakage Audit

The 1,000-patient cohort was partitioned deterministically using `RANDOM_SEED = 42`:
- **Train Split:** 700 patients (70.0%)
- **Validation Split:** 150 patients (15.0%)
- **Test Split:** 150 patients (15.0%)

### Disjoint Set Formal Verification:
- $\text{Train} \cap \text{Validation} = \emptyset$ (Overlap: **0**)
- $\text{Train} \cap \text{Test} = \emptyset$ (Overlap: **0**)
- $\text{Validation} \cap \text{Test} = \emptyset$ (Overlap: **0**)

All 12,000 tiles and all 16012 temporal records inherit their split strictly from the patient ID, preventing patient-level leakage.

---

## 3. Visual Pathology Dataset & Anti-Shortcut Audit

- **Total Processed Tiles:** 12000
- **Tile Format:** 224 x 224 pixels, 3 channels (RGB uint8)
- **Class Breakdown:**
  - `benign`: 4000 tiles (33.3%)
  - `malignant`: 4000 tiles (33.3%)
  - `inflammation`: 4000 tiles (33.3%)
- **Tiles per Split:**
  - `train`: 8400 tiles
  - `validation`: 1800 tiles
  - `test`: 1800 tiles
- **Anti-Shortcut Decoupling:**
  - Background stroma temperatures randomized across all classes: {'neutral': 4044, 'cool': 4021, 'warm': 3935}
  - Exposure and stain gains randomly varied across all classes so the CNN cannot take color/illumination shortcuts.

---

## 4. Longitudinal Biomarkers & Forecasting Windows

- **Total Observations:** 16012
- **Observations per Patient:** Min: 14, Max: 18, Mean: 16.01
- **Temporal Inversions:** 0
- **Forecasting Window Segregation:**
  - Historical Input Window (Days 0–90): 7229 observations (`is_input_window = 1`)
  - Future Prediction Window (Days 91–180+): 8783 observations (`is_input_window = 0`)
- **Forward Prediction Targets:**
  - `future_ctDNA_30d_target`: 13273 observations (82.9%) have forward 30-day targets.
  - `future_progression_trend`: 402 progressing patients, 598 non-progressing patients.
- **Controlled Missingness Rates:**
| ctDNA Missing | CEA Missing | CA-125 Missing | LDH Missing | CRP Missing |
|---|---|---|---|---|
| 5.33% | 5.60% | 5.53% | 5.55% | 5.31% |

---

## 5. Certification & DL Engineer Handoff

The Stage 2 v2 dataset satisfies all data engineering specifications.
**Dataset Status: CERTIFIED PRODUCTION READY FOR DEEP LEARNING MODELING.**
