# Stage 2: Deep Learning Multimodal Progression Prediction

This directory contains the complete end-to-end implementation of **Stage 2** for the Oncology Precision Medicine project.

---

## 📁 Directory Structure

```text
stage-2-dl/
├── data-engineering/    # Multimodal dataset generation (1,000 pts, 12,000 tiles, 16,012 temporal obs)
├── eda/                 # Spatial & temporal exploratory data analysis
├── dl/                  # CNN vision network & LSTM/Transformer sequence forecasting models
├── evaluation/          # Confusion matrices, visual benchmark reels, and progression metrics
└── integration/         # Unified multimodal clinical alert API
```

---

## 🔄 End-to-End Workflow

1. **Data Engineering** (`data-engineering/`):
   - Generates and validates synthetic visual pathology tiles (224x224 RGB, 12,000 tiles) with anti-shortcut stroma decoupling.
   - Generates longitudinal biomarker sequences (16,012 observations across 1,000 patients) spanning >= 180 days across 5 trajectory patterns.
   - Implements strict patient-level splitting (700 Train / 150 Val / 150 Test) with 0% patient leakage.
   - Formulates strict forecasting windows (Days 0–90 Historical Input vs Days 91+ Future Prediction).
   - Enforces training-only data augmentation.
2. **Exploratory Data Analysis** (`eda/`):
   - Computes tissue feature distributions, inspects saliency maps, and evaluates biomarker trajectories.
3. **Deep Learning Pipeline** (`dl/`):
   - Trains CNN vision networks for histopathology tile classification (benign vs malignant vs inflammation).
   - Trains LSTM / Transformer sequence models predicting 30-day forward ctDNA progression trends.
4. **Independent Evaluation** (`evaluation/`):
   - Evaluates trained vision and sequence models on the locked test cohort (150 patients / 1,800 test tiles).
   - Generates confusion matrices and visual benchmark reels.
5. **Inference Integration** (`integration/`):
   - Fuses vision and time-series scores into a single unified clinical alert API.

---

## 🧪 Testing

To run the full test suite for Stage 2 Data Engineering (11 tests):
```bash
python stage-2-dl/data-engineering/tests/test_data_pipeline.py -v
```
