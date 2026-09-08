# Dataset V2 Challenge Layer: Stress-Testing & Robustness Design

**Project:** Personalized Precision Medicine for Oncology Treatment Optimization  
**Subsystem:** `stage-2-dl`  
**Document:** `DATASET_V2_DESIGN.md`  
**Status:** Approved Specification  

---

## 1. Motivation & Scientific Rationale

The baseline Stage 2 dataset consists of 1,000 synthetic patients with 12,000 histopathology tiles and 16,012 longitudinal laboratory observations. While mathematically consistent and strictly split at the patient level, synthetic datasets possess structural regularities that yield artificially high performance ($>98\%$ accuracy, low loss).

In real-world clinical oncology, digital pathology and laboratory time series face substantial domain shifts:
1. **Pre-analytic Histopathology Variations:** Tissue microtome thickness fluctuations, stain vendor batch variance (H&E staining intensity), digital slide scanner chromatic aberrations, focal blur, and lossy slide compression (JPEG/WebP).
2. **Longitudinal Lab & Real-World Visit Noise:** Inconsistent patient compliance (missed clinic visits, irregular follow-up intervals ranging from 14 to 75 days), analytical assay coefficient of variation (inter-assay CV $\sim 5\text{--}15\%$), and assay threshold sensitivity dropouts.
3. **Multimodal Asynchrony & Incompleteness:** Patients frequently possess longitudinal ctDNA blood draws without concurrent repeat tissue biopsies, or histological slides without recent serial lab panels.

**Dataset V2 does not overwrite the clean benchmark data.** Instead, it provides an on-the-fly programmatic challenge layer to stress-test trained architectures under biologically plausible perturbations without altering ground-truth diagnostic or clinical trajectory labels.

---

## 2. Controlled Pathology Perturbation Suite

Each transformation operates on normalized tensor representations or raw RGB tiles prior to evaluation:

| Challenge Type | Transformation Formulation | Biological / Clinical Analog | Parameter Range |
| :--- | :--- | :--- | :--- |
| **Brightness Shift** | $I' = \text{clamp}(I + \delta_{\text{bright}}, 0, 1)$ | Scanner illumination differences, light source decay | $\delta_{\text{bright}} \in [\pm 0.20, \pm 0.35]$ |
| **Contrast Shift** | $I' = \text{clamp}(\bar{I} + \gamma \cdot (I - \bar{I}), 0, 1)$ | Over/under hematoxylin differentiation | $\gamma \in [0.50, 0.75]$ |
| **Stain / Color Jitter** | Hue and saturation channel perturbation | Variation in eosin/hematoxylin dye chemistry between pathology laboratories | $\Delta \text{Hue} = 0.08, \Delta \text{Sat} = 0.25$ |
| **Gaussian Blur** | $I' = I * \mathcal{K}_{\sigma}$ | Out-of-focus slide scanning, optical refraction | Kernel $5 \times 5, \sigma \in [1.0, 2.0]$ |
| **Additive Noise** | $I' = \text{clamp}(I + \epsilon, 0, 1), \ \epsilon \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)$ | Electronic sensor thermal noise at high optical magnification | $\sigma_{\text{noise}} \in [0.05, 0.12]$ |
| **Lossy Compression** | JPEG encode/decode cycle | Whole-slide image archive compression bandwidth limits | Quality factor $Q \in [30, 45]$ |

**Label Invariance Guarantee:** None of the visual transformations obliterate nuclear chromatin architecture or tissue topology sufficiently to convert a malignant tile to benign or vice versa.

---

## 3. Controlled Temporal & Biomarker Perturbation Suite

The historical sequence $X_{1:L_i} \in \mathbb{R}^{L_i \times 13}$ for patient $i$ undergoes controlled clinical perturbations:

| Challenge Type | Mathematical Operation | Clinical Scenario | Perturbation Strength |
| :--- | :--- | :--- | :--- |
| **Measurement Noise** | $v_{t, k}' = v_{t, k} \cdot (1 + \eta), \ \eta \sim \mathcal{N}(0, \sigma_k^2)$ | Analytical lab assay coefficient of variation (e.g. digital droplet PCR ctDNA noise) | $\sigma_k = 0.15$ (15% noise) |
| **Irregular Intervals** | $\Delta t' = \Delta t + \text{Uniform}(-10, 25)$ days | Patient appointment rescheduling, delayed clinical visits | Jitter $\in [-10, +25]$ days |
| **Missing Visits** | Drop timepoint $t \in (1, \dots, L_i-1)$ with probability $p_{\text{drop}}$ | Patient missed scheduled interim follow-up | $p_{\text{drop}} = 0.33$ |
| **Missing Biomarkers** | Set biomarker $k$ to imputed 0 and trigger mask indicator $M_{t, k} = 1$ | Specific assay not ordered by attending oncologist | $p_{\text{drop\_feat}} = 0.25$ |
| **Sporadic Lab Outliers** | $v_{t, k}' = v_{t, k} \cdot 3.0$ with probability $p=0.05$ | Transient acute inflammation flare (e.g. CRP/LDH infection spike) | Spike factor $3.0\times$ |

---

## 4. Multimodal Asymmetry & Partial Availability

To evaluate multimodal resilience, the pipeline evaluates:
1. **Full Multimodal (Reference):** All 12 pathology tiles and complete historical visits present.
2. **Missing Pathology (Temporal Only):** Pathology bag unavailable; system must rely exclusively on longitudinal lab trajectory.
3. **Missing Temporal (Pathology Only):** No longitudinal blood draws available; system must infer baseline risk purely from tissue tiles.
4. **Partial Pathology Bag:** Patient has only 3 or 6 tiles available instead of 12 (simulating limited needle core biopsy samples).
5. **Combined Partial Degradation:** Partial tiles (6 tiles) + missing visits (1 interim visit dropped).

---

## 5. Evaluation Protocol

All challenge evaluations are conducted on the **Locked Test Set ($N=150$ patients)** using models whose weights and hyperparameters were frozen after validation.

Metrics reported:
- Absolute performance under challenge: $\text{F1}_{\text{challenge}}$, $\text{ROC-AUC}_{\text{challenge}}$, $\text{MAE}_{\text{challenge}}$.
- Performance degradation percentage:
  $$\Delta_{\text{degradation}} (\%) = \frac{\text{Metric}_{\text{clean}} - \text{Metric}_{\text{challenge}}}{\text{Metric}_{\text{clean}}} \times 100$$
- Uncertainty calibration and distribution shift flags under challenge conditions.
