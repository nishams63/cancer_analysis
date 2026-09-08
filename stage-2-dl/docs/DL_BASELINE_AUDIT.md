# Stage 2 Deep Learning Baseline System Audit

**Project:** Personalized Precision Medicine for Oncology Treatment Optimization  
**Repository:** `https://github.com/nishams63/cancer_analysis`  
**Subsystem:** `stage-2-dl`  
**Audit Date:** September 2026  
**Auditor:** Senior Deep Learning Engineer & Medical AI Research Engineer  
**Status:** Audit Completed — Baselines Frozen & Preserved  

---

## 1. Executive Summary & Overview

This document presents a rigorous technical and methodological audit of the existing Stage 2 Deep Learning system. In accordance with clinical AI research standards and the core engineering mandate:
- **No baseline code or models have been destroyed or overwritten.**
- **The existing ResNet-18 pathology classifier and BiLSTM temporal forecaster remain intact as permanent baselines.**
- **Data partitions and temporal constraints are strictly audited for leakage.**
- **The test set is confirmed locked and protected from parameter optimization or checkpoint selection.**

The system under audit comprises two unimodal deep learning branches and an integration layer:
1. **Pathology Branch:** Transfer learning with ResNet-18 (and a 4-stage Custom CNN baseline) operating on $224 \times 224 \times 3$ RGB hematoxylin and eosin (H&E) biopsy tiles across 3 morphological classes (`benign`, `malignant`, `inflammation`).
2. **Temporal Biomarker Branch:** 2-layer Bidirectional LSTM processing longitudinal multivariate laboratory trajectories (13 features: 8 numerical + 5 missingness indicators) up to Day 90, performing joint multi-task prediction (30-day future ctDNA VAF regression + future progression/recurrence binary classification).
3. **Multimodal Fusion & Integration Layer:** Patient-level tile aggregation (mean/median/max probability) coupled with fixed-weight linear fusion (0.35 pathology + 0.40 progression + 0.25 ctDNA index) yielding a prototype composite risk score.

---

## 2. Detailed Component Audit

### 2.1 Data Pipeline & Partitions
- **Cohort Manifest:** 1,000 synthetic oncology patients split strictly at the patient level into:
  - **Training:** 700 patients (8,400 tiles, 11,208 temporal visits)
  - **Validation:** 150 patients (1,800 tiles, 2,402 temporal visits)
  - **Locked Test:** 150 patients (1,800 tiles, 2,402 temporal visits)
- **Zero Patient Overlap:** Confirmed across `train_patients.csv`, `validation_patients.csv`, and `test_patients.csv`. No patient ID appears in more than one partition.
- **Image Metadata Manifest:** 12,000 tiles total, exactly 12 tiles per patient.
  - Class distribution across tiles: 4,000 benign, 4,000 malignant, 4,000 inflammation (balanced 1:1:1).
- **Temporal Manifest:** 16,012 total visit records.
  - Historical observation window: `days_from_baseline <= 90` and `is_input_window == 1` (max 4 historical visits per patient, $t_0, t_{30}, t_{60}, t_{90}$).
  - Future observation window: `days_from_baseline > 90` (used strictly for target extraction: $t_{120}$ ctDNA VAF and future progression events).

### 2.2 Pathology Branch (Image Model)
- **Architecture:** `ResNet18Transfer`
  - Backbone: PyTorch `torchvision.models.resnet18` initialized with ImageNet weights.
  - Top classification head: `Linear(512, 128) -> ReLU() -> Dropout(p=0.3) -> Linear(128, 3)`.
  - Alternative baseline: `PathologyCNN` (4-stage Conv2D + BatchNorm + ReLU + MaxPool with 32, 64, 128, 128 channels).
- **Input Specifications:**
  - Input tensor shape: `(batch_size, 3, 224, 224)`
  - Value range: Normalized float tensor using ImageNet parameters ($\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$) or empirical dataset statistics ($\mu=[0.8422, 0.7340, 0.8337], \sigma=[0.1928, 0.2293, 0.1498]$).
- **Augmentation Pipeline:**
  - `train_transform`: `RandomHorizontalFlip(p=0.5)`, `RandomVerticalFlip(p=0.5)`, `RandomRotation(degrees=90)`, `ColorJitter(brightness=0.1, contrast=0.1)`.
  - `val/test_transform`: Deterministic `Resize((224, 224))`, `ToTensor()`, `Normalize()`.
- **Output Specifications:**
  - Logits shape: `(batch_size, 3)`.
  - Probabilities via Softmax: `[P_benign, P_malignant, P_inflammation]`.
- **Loss Function:** Uniform Cross Entropy (`nn.CrossEntropyLoss()`).
- **Optimization:** AdamW ($\text{lr}=10^{-3}$, weight decay $=10^{-4}$), Cosine Annealing scheduler.

### 2.3 Temporal Biomarker Branch (Time Series Model)
- **Architecture:** `BiLSTMForecaster`
  - Input dimension: 13 features.
  - Recurrent backbone: 2-layer Bidirectional LSTM, hidden size 64 per direction (128-dim bidirectional concatenated hidden state).
  - Sequence representation: Extracted at true sequence length $(L_i - 1)$ via `gather` to avoid padding contamination.
  - Multi-Task Prediction Heads:
    - **Head A (Regression):** `Linear(128, 32) -> ReLU() -> Dropout(0.2) -> Linear(32, 1)` for 30-day future ctDNA VAF.
    - **Head B (Classification):** `Linear(128, 32) -> ReLU() -> Dropout(0.2) -> Linear(32, 1)` for future disease progression logit.
- **Input Features (13 Total):**
  - Continuous / Numerical (8): `ctDNA_vaf_percent`, `cea_ng_ml`, `ca125_u_ml`, `ldh_u_l`, `crp_mg_l`, `ctDNA_velocity_30d`, `delta_days`, `days_from_baseline`.
  - Missingness Indicators (5): `ctDNA_missing`, `cea_missing`, `ca125_missing`, `ldh_missing`, `crp_missing`.
- **Normalization:** Continuous features normalized with z-score parameters learned strictly on the training partition ($N_{\text{train}}=700$).
- **Loss Formulation:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SmoothL1}}(\hat{y}_{\text{ctDNA}}, y_{\text{ctDNA}}) + \lambda_{\text{cls}} \cdot \mathcal{L}_{\text{BCEWithLogits}}(\hat{z}_{\text{prog}}, y_{\text{prog}})$$
  Default $\lambda_{\text{cls}} = 1.0$.
- **Optimization:** AdamW ($\text{lr}=2 \times 10^{-3}$, weight decay $=10^{-3}$), `ReduceLROnPlateau` on validation composite score.

### 2.4 Multimodal Integration & Fusion Layer
- **Patient-Level Tile Aggregation:**
  - Baseline uses uniform mean pooling of tile probabilities:
    $$\bar{P}_c = \frac{1}{12}\sum_{k=1}^{12} P_{c, k}, \quad c \in \{\text{benign}, \text{malignant}, \text{inflammation}\}$$
  - Also provides median and max-risk heuristic aggregation.
- **Fixed Linear Risk Fusion:**
  - Composite Risk Index:
    $$S_{\text{multimodal}} = 0.35 \cdot P_{\text{malignant}} + 0.40 \cdot P_{\text{progression}} + 0.25 \cdot \tilde{V}_{\text{ctDNA}}$$
    where $\tilde{V}_{\text{ctDNA}} = \text{clamp}\left(\frac{\hat{V}_{\text{ctDNA}} - 0.0}{10.0 - 0.0}, 0, 1\right)$.
  - Prototype alert stratification: Low ($<0.33$), Moderate ($[0.33, 0.66)$), High ($\ge 0.66$).

---

## 3. Methodological & Risk Audit

### 3.1 Potential Data Leakage Risks & Verification
| Risk Area | Mechanism Audited | Status / Findings | Recommendation |
| :--- | :--- | :--- | :--- |
| **Patient-Split Leakage** | Patient ID crossing between Train, Val, and Test sets. | **VERIFIED CLEAN:** Splits generated via hash/stratified patient manifests. | Maintain hard assertion checks in all dataset loaders. |
| **Temporal Target Leakage** | Using observations beyond Day 90 in the input sequence. | **VERIFIED CLEAN:** Filter `days_from_baseline <= 90` is strictly enforced in `TemporalSequenceDataset`. | Retain strict landmark boundary at $t=90$ days. |
| **Feature Normalization Leakage** | Calculating $\mu$ and $\sigma$ across the entire dataset. | **VERIFIED CLEAN:** `norm_params` learned exclusively on `split == 'train'`. | Ensure newly added models reuse the same training normalization dictionary. |
| **Tile-Patient Independence Leakage** | Treating tiles as independent samples during evaluation. | **POTENTIAL RISK IDENTIFIED:** Tile classifier evaluated tile-by-tile. Patient-level metrics are computed downstream in integration. | Introduce native Patient-level evaluation directly in the benchmark suite (MIL). |

### 3.2 Potential Overfitting & Generalization Risks
1. **Synthetic Feature Regularity:** Synthetic biomarker trajectories exhibit lower variance than real clinical cohorts. High accuracy ($>95\%$) on validation may lead to false optimism.
2. **Fixed Head Frozen Backbone:** Transfer learning with frozen ResNet-18 layers limits representation adaptation for tissue architecture.
3. **Fixed Fusion Weights:** Hardcoded linear weights ($0.35, 0.40, 0.25$) do not account for patient-specific modality reliability or missing tile patterns.
4. **Tile Independence Assumption:** Uniform mean pooling over 12 tiles ignores focal malignant nests (a single malignant tile in 12 should heavily elevate patient risk).

---

## 4. Current Baseline Limitations & Upgrade Opportunities

1. **Pathology Feature Extraction:**
   - ResNet-18 is a compact general-purpose backbone; it lacks multi-scale receptive fields and self-attention over spatial patches.
   - Upgrade: Benchmark against ResNet-50, EfficientNet-B0/B2, and Vision Transformer (ViT-B/16 or compact ViT).
2. **Patient-Level Aggregation:**
   - Current mean aggregation treats all 12 tiles identically.
   - Upgrade: Implement Attention-based Multiple Instance Learning (Attention MIL) to learn tile importance weights $\alpha_k$ dynamically:
     $$H_{\text{patient}} = \sum_{k=1}^{12} \alpha_k H_k, \quad \alpha_k = \frac{\exp\{w^T \tanh(V H_k^T)\}}{\sum_{j} \exp\{w^T \tanh(V H_j^T)\}}$$
3. **Temporal Dynamics & Irregular Time-Series:**
   - BiLSTM assumes sequential recurrence; though `delta_days` is an input feature, standard LSTM does not embed continuous time intervals geometrically.
   - Upgrade: Implement a Multi-Task Temporal Transformer with explicit continuous positional/delta-time embeddings and causal padding masks.
4. **Learned Multimodal Fusion:**
   - Linear fixed weighting cannot model cross-modal interactions (e.g., high ctDNA velocity amplifying borderline pathology).
   - Upgrade: Implement learned Concatenation+MLP, Gated Multimodal Fusion, and Cross-Attention Fusion.
5. **Robustness & Uncertainty Under Clinical Reality:**
   - Current baseline lacks out-of-distribution (OOD) flagging, temperature-scaled calibration (ECE), and Monte Carlo Dropout uncertainty estimation.
   - Upgrade: Build a comprehensive challenge robustness suite and calibration pipeline.

---

## 5. Upgrade Action Plan

| Phase | Milestone | Expected Deliverable |
| :--- | :--- | :--- |
| **Phase 1** | Baseline Audit & Test Policy | `docs/DL_BASELINE_AUDIT.md`, `docs/TEST_SET_POLICY.md` |
| **Phase 2** | Controlled Pathology Benchmark | ResNet-18 (baseline), ResNet-50, EfficientNet-B0, ViT + Realistic Augmentation Pipeline |
| **Phase 3** | Patient-Level MIL Pathology | Attention-MIL, Mean, Median, Max comparison with tile attention maps |
| **Phase 4** | Temporal Transformer Upgrade | Continuous time encoding, multi-task heads, irregular interval testing |
| **Phase 5** | Learned Multimodal Fusion | Gated, Attention, and Concat MLP vs Fixed Linear Baseline |
| **Phase 6** | Uncertainty, Calibration & OOD | MC Dropout, Temperature Scaling, ECE, Mahalanobis/Embedding OOD |
| **Phase 7** | Robustness & Challenge Testing | Stress testing (noise, blur, stain jitter, missing visits, partial modalities) |
| **Phase 8** | Verification & Final Reporting | `results/model_comparison.csv`, `results/ablation.csv`, `docs/MODEL_COMPARISON.md`, `docs/ROBUSTNESS_REPORT.md`, `docs/STAGE_2_DL_FINAL_REPORT.md` |

---
*Certified reproducible by Senior Deep Learning Engineer & Medical AI Research Engineer.*
