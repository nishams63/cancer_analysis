# Test Set Governance & Anti-Overfitting Policy

**Project:** Personalized Precision Medicine for Oncology Treatment Optimization  
**Subsystem:** `stage-2-dl`  
**Effective Date:** September 2026  
**Status:** ACTIVE & STRICTLY ENFORCED  

---

## 1. Principle of Locked Test Evaluation

In high-stakes oncology AI applications, optimizing hyperparameters, architecture selections, decision thresholds, or feature representations against the test partition creates **subtle information leakage** and **optimistic performance bias**. 

To preserve the scientific integrity, reproducibility, and generalizability of this Stage 2 research prototype, the test set ($N=150$ patients, 1,800 pathology tiles, 2,402 longitudinal biomarker visits) is **strictly locked**.

---

## 2. Partition Roles & Boundaries

```
                 ┌────────────────────────────────────────────────┐
                 │           1,000 PATIENT COHORT                 │
                 └───────────────────────┬────────────────────────┘
                                         │
       ┌─────────────────────────────────┼────────────────────────────────┐
       ▼                                 ▼                                ▼
┌──────────────┐                 ┌──────────────┐                 ┌──────────────┐
│   TRAINING   │                 │  VALIDATION  │                 │ LOCKED TEST  │
│  700 Patients│                 │  150 Patients│                 │ 150 Patients │
│  8,400 Tiles │                 │  1,800 Tiles │                 │ 1,800 Tiles  │
└──────┬───────┘                 └──────┬───────┘                 └──────┬───────┘
       │                                │                                │
       │ Gradient Updates               │ Checkpoint Selection           │ Single Final
       │ Feature Normalization          │ Hyperparameter Tuning          │ Unbiased Run
       │ Augmentation Fitting           │ Temperature Scaling (ECE)      │ (No Feedback)
       ▼                                ▼                                ▼
[ Model Optimization ]          [ Model Selection ]              [ Final Reporting ]
```

### Partition Governance Rules:
1. **Training Split ($N=700$ patients):**
   - Strictly reserved for weight and parameter updates via gradient descent.
   - All empirical feature normalization parameters ($\mu, \sigma$) must be computed exclusively on this partition.
   - Training augmentations (geometric, color, noise) are applied exclusively here.
2. **Validation Split ($N=150$ patients):**
   - Used for early stopping, model selection, loss weight balancing ($\lambda_{\text{cls}}$), and post-hoc calibration (temperature scaling).
   - All architecture comparison metrics (ResNet vs ViT, BiLSTM vs Transformer, Fusion mechanisms) are evaluated here to choose primary candidates.
3. **Locked Test Split ($N=150$ patients):**
   - **LOCKED FROM MODEL TUNING:** Never use test metrics to revise learning rates, modify architectures, adjust loss functions, or select checkpoints.
   - Evaluated **only once** at final milestone completion to report unbiased benchmark figures.
   - Any test evaluation failure or degraded metric must be reported honestly as an experimental finding, rather than followed by iterative re-tuning.

---

## 3. Anti-Leakage Checkpoints & Code Enforcement

All dataset classes and training scripts must enforce the following assertions:

1. **Patient Disjointness Assertion:**
   $$\text{Set}(\text{Patients}_{\text{Train}}) \cap \text{Set}(\text{Patients}_{\text{Val}}) = \emptyset$$
   $$\text{Set}(\text{Patients}_{\text{Train}}) \cap \text{Set}(\text{Patients}_{\text{Test}}) = \emptyset$$
   $$\text{Set}(\text{Patients}_{\text{Val}}) \cap \text{Set}(\text{Patients}_{\text{Test}}) = \emptyset$$

2. **Temporal Window Assertion:**
   $$\forall \text{ observation } i \in \text{InputSequence}: \text{days\_from\_baseline}_i \le 90$$
   Any target feature ($t_{120}$ ctDNA VAF or progression) must never enter the model encoder.

3. **Normalization Boundary:**
   `TemporalSequenceDataset(split='test', norm_params=train_dataset.norm_params)`
   Test normalization statistics must be passed directly from training, never re-estimated on test data.

---

## 4. Scientific Disclaimer

This model is a research prototype evaluated on synthetic data. Preservation of test partition integrity simulates clinical validation rigor in-silico, but performance on this synthetic dataset does not establish real-world clinical safety, efficacy, or diagnostic validity.
