# Stage 2 Deep Learning (DL) Module

## Overview
This module contains the deep learning models, training scripts, checkpoints, and inference engines for Stage 2 of the project:

**"Personalized Precision Medicine for Oncology Treatment Optimization"**

It implements two decoupled deep learning architectures:
1. **Model A — Pathology Tile Classifier (CNN):** Classifies $224 \times 224 \times 3$ RGB whole-slide biopsy tiles into **Benign**, **Malignant**, or **Inflammation**.
2. **Model B — Longitudinal Biomarker Forecaster (BiLSTM):** Processes patient biomarker trajectories during the induction window (Days 0–90) to predict 30-day forward ctDNA VAF and disease progression risk.

---

> [!IMPORTANT]
> **Mandatory Clinical & Synthetic Data Notice:**  
> **This model was developed using synthetic data and has not been clinically validated. Performance on this dataset does not establish clinical effectiveness, safety, or real-world patient benefit.**  
> $$\text{Synthetic data} \ne \text{Real patient evidence} \ne \text{Clinical validation}$$  
> All model checkpoints and metrics report technical prototyping convergence, not real medical diagnosis or treatment guidance.

---

## Directory Layout

```text
stage-2-dl/dl/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Paths, constants, seed=42, class mappings, normalization parameters
│   ├── datasets.py               # PyTorch Dataset classes for image tiles and temporal sequences
│   ├── image_model.py            # ResNet-18 transfer learning and custom CNN architectures
│   ├── temporal_model.py         # Multi-task BiLSTM and Transformer sequence forecasters
│   ├── utils.py                  # Metrics, checkpoint serializer, seed initialization, curve plotting
│   ├── train_image.py            # Reproducible CNN training script with validation early stopping
│   ├── train_temporal.py         # Reproducible BiLSTM training script with multi-task loss
│   └── inference.py              # Standalone inference predictors for image and temporal models
├── checkpoints/
│   ├── best_pathology_cnn.pt     # Best validation checkpoint for pathology CNN (weights, config, metrics)
│   └── best_temporal_lstm.pt     # Best validation checkpoint for temporal BiLSTM (weights, scaler, metrics)
├── figures/
│   ├── image_training_curves.png # Loss and macro F1 progression over training epochs
│   ├── temporal_training_curves.png # Multi-task loss, ctDNA MAE, and progression F1 curves
│   └── validation_confusion_matrix.png # Confusion matrix on the validation split
├── reports/
│   └── dl_report.md              # 18-section comprehensive DL engineering and handoff report
├── tests/
│   ├── __init__.py
│   └── test_dl.py                # Automated unit test suite covering loading, shapes, leakage, and inference
├── requirements.txt              # Module dependencies
└── README.md                     # This file
```

---

## What the Models Do

### Model A: Pathology Tile Classifier (CNN)
- **Input:** $224 \times 224 \times 3$ RGB pathology tile.
- **Architecture:** `ResNet-18` backbone with transfer learning weights and a custom classification head (`Linear(512, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 3)`).
- **Target:** Predicts probability across 3 classes:
  - `0`: Benign
  - `1`: Malignant
  - `2`: Inflammation
- **Output:** Predicted class, confidence score, and class probability distribution.

### Model B: Longitudinal Biomarker Forecaster (BiLSTM)
- **Input:** A sequence of clinical visits during the patient's first 90 days of treatment ($t \le 90$).
- **Features per Visit (13):** ctDNA VAF, CEA, CA-125, LDH, CRP, ctDNA 30-day velocity, delta days, days from baseline, and 5 companion missingness masks.
- **Architecture:** 2-layer Bidirectional LSTM with unpadded last-visit latent state extraction and dual multi-task output heads.
- **Targets:**
  - **Head A (Continuous Regression):** Predicts future ctDNA VAF at approximately $t + 30$ days (`future_ctDNA_30d_target`).
  - **Head B (Binary Classification):** Predicts disease progression/recurrence risk (`future_progression_trend`: 0 = Stable, 1 = Progression).

---

## How Data Leakage is Prevented

1. **Patient Split Isolation:** All splits are partitioned strictly by `patient_id`. Zero patient overlap exists between training, validation, and test sets.
2. **Temporal Window Segregation:** The sequence dataset strictly filters to `is_input_window == 1` and `days_from_baseline <= 90`. Observations occurring after Day 90 are never fed into the sequence model.
3. **Target Isolation:** Ground truth prediction targets (`future_ctDNA_30d_target` and `future_progression_trend`) are strictly extracted as supervised training targets and are excluded from input feature tensors.
4. **Training-Only Normalization:** Feature standardization parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) are computed exclusively on training patients and applied to validation/test data.
5. **Locked Test Split:** The 150 test patients are **completely untouched** by the DL module. The Evaluation Engineer owns the final test-set evaluation.

---

## Installation & Setup

Ensure you have a Python 3.10+ environment. Install dependencies:
```powershell
pip install -r stage-2-dl/dl/requirements.txt
```

---

## How to Run Tests

Run the automated DL test suite:
```powershell
python stage-2-dl/dl/tests/test_dl.py -v
```

---

## How to Train Models

### 1. Train Pathology CNN
```powershell
# Full training (3 epochs on CPU with frozen backbone transfer learning)
python stage-2-dl/dl/src/train_image.py --epochs 3 --batch-size 64

# Quick smoke-test (1 epoch on a mini subset)
python stage-2-dl/dl/src/train_image.py --smoke-test
```

### 2. Train Temporal BiLSTM
```powershell
# Full multi-task training (25 epochs on CPU)
python stage-2-dl/dl/src/train_temporal.py --epochs 25 --batch-size 32

# Quick smoke-test (2 epochs on a mini subset)
python stage-2-dl/dl/src/train_temporal.py --smoke-test
```

---

## How to Run Inference

### Image Tile Inference
```python
import sys
sys.path.insert(0, 'stage-2-dl/dl/src')
from inference import PathologyImagePredictor

predictor = PathologyImagePredictor()
result = predictor.predict("path/to/tile.png")

print(result)
# {
#   "prediction": "malignant",
#   "class_idx": 1,
#   "confidence": 0.9998,
#   "probabilities": {
#     "benign": 0.0001,
#     "malignant": 0.9998,
#     "inflammation": 0.0001
#   },
#   "disclaimer": "..."
# }
```

### Temporal Sequence Inference
```python
import sys
sys.path.insert(0, 'stage-2-dl/dl/src')
import pandas as pd
from inference import TemporalBiomarkerPredictor

predictor = TemporalBiomarkerPredictor()

# Load patient induction records (t <= 90 days)
patient_history = pd.read_csv("patient_visits.csv")
result = predictor.predict(patient_history)

print(result)
# {
#   "predicted_ctDNA_30d_vaf": 1.75,
#   "predicted_progression_risk": 0.999,
#   "predicted_progression": 1,
#   "input_sequence_length": 7,
#   "max_days_from_baseline": 83,
#   "disclaimer": "..."
# }
```

---

## Handoff to Evaluation Engineer

The DL module has completed development-time validation.

> [!NOTE]
> **Locked-Test Evaluation Policy:**  
> Final locked-test evaluation is intentionally not included in this DL module. The Evaluation Engineer owns the final test-set evaluation.  
> Checkpoint files `best_pathology_cnn.pt` and `best_temporal_lstm.pt` are preserved in `stage-2-dl/dl/checkpoints/` and ready for evaluation benchmarking.

### Recommended Investigation for Evaluation Engineer:
Validation performance is exceptionally high (CNN: 100%, Progression F1: 100%, ctDNA R?: 0.869). The Evaluation Engineer should investigate whether synthetic generation artifacts create unusually high separability, perform feature attribution (e.g., Grad-CAM, permutation importance), and test perturbation robustness on the locked test set.
