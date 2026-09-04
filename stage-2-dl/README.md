# Stage 2 Deep Learning Dataset (Version 2) — Data Engineer Handoff Package

**Project:** Personalized Precision Medicine for Oncology Treatment Optimization  
**Stage:** Stage 02 — Deep Learning (Visual & Temporal Data Engineering)  
**Version:** Version 2 (`stage-2-dl/data/v2/`)  
**Role:** Stage 2 Data Engineer  
**Handoff Target:** Stage 2 Deep Learning Engineer  

> [!IMPORTANT]
> **MANDATORY CLINICAL & SYNTHETIC DATA NOTICE:**  
> **Synthetic data created for deep-learning research and pipeline prototyping. These data are not real patient records and are not clinically validated.**  
> $$\text{Synthetic data} \ne \text{Real patient evidence} \ne \text{Clinical validation}$$  
> This dataset has been engineered deterministically for machine learning model development, dataloader benchmarking, and multi-modal pipeline architecture validation. Never present downstream model evaluations as real clinical efficacy.

---

## 1. Team Responsibilities & Boundaries

```
[COMPLETED] Stage 2 Data Engineer
   ├── Deterministic synthetic generation (1,000 patients, 12,000 pathology tiles, ~16,000 temporal records)
   ├── Patient-level partitioning (Train: 700, Val: 150, Test: 150) with 0% patient leakage
   ├── Anti-shortcut image generation (randomized stroma temperature, exposure, and stain gains)
   ├── Image standardization (224x224x3 RGB PNGs, class folder organization, metadata manifest)
   ├── Forecasting sequence engineering (Days 0-90 input window vs Days 91+ prediction window)
   ├── Training-only data augmentation pipeline (H&E stain jitter, D4 symmetries, optical artifacts)
   ├── Automated 19-point validation suite & dataset_statistics.csv export
   └── PyTorch Dataset loaders & comprehensive handoff documentation

[NEXT STEP] Stage 2 Deep Learning Engineer
   ├── Task 1 (Computer Vision): Train CNN (ResNet, EfficientNet, ConvNeXt) to classify pathology tiles (benign vs malignant vs inflammation)
   ├── Task 2 (Sequence Modeling): Train LSTM / Transformer to forecast future ctDNA progression from 90-day history
   ├── Multi-modal fusion experiments & hyperparameter tuning
   └── Model evaluation and benchmark comparison reel
```

---

## 2. Directory Structure

```
stage-2-dl/
├── data/
│   ├── (existing v1 prototype preserved)
│   └── v2/
│       ├── raw/
│       │   ├── images/                     # 12,000 raw procedural tiles + manifest
│       │   └── biomarkers/                 # Raw longitudinal observations
│       ├── processed/
│       │   ├── pathology_tiles/            # Standardized 224x224 RGB tiles in class folders
│       │   │   ├── benign/                 # 4,000 tiles
│       │   │   ├── malignant/              # 4,000 tiles
│       │   │   └── inflammation/           # 4,000 tiles
│       │   ├── image_metadata.csv          # Master tile manifest with patient/split index
│       │   └── biomarkers_processed.csv    # Chronological series with forecasting windows
│       └── splits/
│           ├── train_patients.csv          # 700 Train patients
│           ├── validation_patients.csv     # 150 Validation patients
│           └── test_patients.csv           # 150 Test patients
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Global constants, paths, seed=42, disclaimers
│   ├── create_splits.py                # 700/150/150 stratified patient partition
│   ├── data_generation.py              # Vectorized anti-shortcut tile & biomarker generator
│   ├── image_preparation.py            # QA filtering, class sorting, metadata compilation
│   ├── biomarker_preparation.py        # Sequence monotonicity & forecasting window labeling
│   ├── augmentation.py                 # Training-only stain jitter and D4 transformations
│   ├── validation.py                   # Automated 19-point validation & audit suite
│   └── pipeline.py                     # Single-command end-to-end reproduction runner
│
├── reports/
│   ├── data_engineering_report.md      # Comprehensive audit report for v2
│   ├── data_dictionary.md              # Detailed schema, feature roles, and forecasting protocol
│   ├── dataset_statistics.csv          # Consolidated metrics and summary table
│   └── augmentation_samples_comparison.png # Visual QC inspection grid (training-only)
│
├── tests/
│   ├── __init__.py
│   └── test_data_pipeline.py           # Automated unit test suite (11 comprehensive tests)
│
└── README.md                           # This handoff guide
```

---

## 3. Quickstart & Reproducibility

The entire v2 dataset can be deterministically reproduced from scratch using a single command:

```powershell
# Run the complete end-to-end v2 pipeline (generation -> processing -> validation)
python stage-2-dl/src/pipeline.py
```

To run the automated test suite:
```powershell
python stage-2-dl/tests/test_data_pipeline.py -v
```

---

## 4. Dataset Overview & Key Statistics

### A) Visual Histopathology Dataset (for CNN)
* **Total Tiles:** Exactly 12,000 tiles ($224 \times 224 \times 3$ RGB PNG format).
* **Cohort Coverage:** 1,000 unique patients (Exactly 12 tiles per patient across 2 biopsy slides).
* **Class Balance:** Exactly 4,000 benign (33.33%), 4,000 malignant (33.33%), 4,000 inflammation (33.33%).
* **Split Allocation:**
  * **Train:** 8,400 tiles (700 patients)
  * **Validation:** 1,800 tiles (150 patients)
  * **Test:** 1,800 tiles (150 patients)
* **Anti-Shortcut Decoupling:** Global background stroma color temperature (warm, neutral, cool), brightness ($0.92 - 1.08$), contrast ($0.90 - 1.10$), and H&E gains are randomized independently of the class. Only cellular morphology differentiates the classes.

### B) Longitudinal Biomarker Dataset (for LSTM / Transformer)
* **Total Observations:** ~16,000 observations across 1,000 patients (14–18 visits per patient).
* **Longitudinal Span:** Every patient spans $\ge 180$ days from baseline intake.
* **Split Allocation:**
  * **Train:** ~11,200 observations (700 patients)
  * **Validation:** ~2,400 observations (150 patients)
  * **Test:** ~2,400 observations (150 patients)
* **Biomarkers:** ctDNA VAF (%), CEA (ng/mL), CA-125 (U/mL), LDH (U/L), CRP (mg/L).
* **Controlled Missingness:** 3.0% – 8.0% missing values tracked via binary indicator masks (`ctDNA_missing`, etc.).
* **Forecasting Structure:**
  * **Historical Input Window:** `days_from_baseline <= 90` (`is_input_window = 1`)
  * **Future Prediction Window:** `days_from_baseline > 90` (`is_input_window = 0`)
  * **Forward Targets:** `future_ctDNA_30d_target` and `future_progression_trend`.

---

## 5. Critical Leakage Prevention Guarantee

Data partitioning is performed strictly at the **patient level**:
$$\text{Train Patients} \cap \text{Validation Patients} = \emptyset$$
$$\text{Train Patients} \cap \text{Test Patients} = \emptyset$$
$$\text{Validation Patients} \cap \text{Test Patients} = \emptyset$$

* **Zero Image Leakage:** All 12 tiles of a patient belong exclusively to that patient's designated split.
* **Zero Temporal Leakage:** A patient's entire longitudinal sequence belongs exclusively to their assigned split.
* **Zero Forecast Feature Leakage:** Future targets (`future_ctDNA_30d_target`, `future_progression_trend`) are strictly prediction targets and must NEVER be passed as input features to sequence models.

---

## 6. Training-Only Augmentation Specification

Histopathology sections lack anatomical orientation. The following transformations are available:
1. **H&E Stain Jitter:** Independent channel gain variation ($\pm 7\%$).
2. **Scanner Exposure:** Brightness ($\pm 10\%$) and Contrast ($\pm 10\%$) adjustments.
3. **Dihedral $D_4$ Symmetries:** Random discrete $90^\circ, 180^\circ, 270^\circ$ rotations and horizontal/vertical flips.
4. **Slide Artifacts:** Subtle Gaussian focal blur and lens vignetting.

> [!WARNING]
> **Strict Policy:** Augmentation is implemented **ONLY** for the `train` split. Validation and test sets remain strictly unaugmented for deterministic, uncorrupted evaluation.

---

## 7. PyTorch Dataset & DataLoader Guide for the DL Engineer

### A) Visual CNN Dataloader (`PathologyTileDatasetV2`)

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from stage_2_dl.src.augmentation import HistopathologyTrainAugmentation

LABEL_MAP = {"benign": 0, "malignant": 1, "inflammation": 2}

class PathologyTileDatasetV2(Dataset):
    def __init__(self, metadata_path="stage-2-dl/data/v2/processed/image_metadata.csv", split="train"):
        df = pd.read_csv(metadata_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.split = split
        self.aug = HistopathologyTrainAugmentation()
        
        # Standard normalization
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        
        # Apply augmentation ONLY if training split
        if self.split == "train":
            img = self.aug(img, split="train")
            
        tensor_img = self.to_tensor(img)
        label = LABEL_MAP[row["class_label"]]
        return tensor_img, torch.tensor(label, dtype=torch.long), row["patient_id"]

# Example:
# train_loader = DataLoader(PathologyTileDatasetV2(split="train"), batch_size=32, shuffle=True)
# val_loader   = DataLoader(PathologyTileDatasetV2(split="validation"), batch_size=32, shuffle=False)
```

### B) Temporal Forecasting LSTM / Transformer Dataloader (`TemporalForecastingDatasetV2`)

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

class TemporalForecastingDatasetV2(Dataset):
    def __init__(self, data_path="stage-2-dl/data/v2/processed/biomarkers_processed.csv", split="train"):
        df = pd.read_csv(data_path)
        self.split_df = df[df["split"] == split].reset_index(drop=True)
        self.patient_ids = self.split_df["patient_id"].unique()
        self.feature_cols = [
            "days_from_baseline", "delta_days", "ctDNA_vaf_percent",
            "cea_ng_ml", "ca125_u_ml", "ldh_u_l", "crp_mg_l", "ctDNA_velocity_30d"
        ]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        p_data = self.split_df[self.split_df["patient_id"] == pid].sort_values("days_from_baseline")
        
        # RESTRICT INPUT STRICTLY TO HISTORICAL INPUT WINDOW (days <= 90)
        input_history = p_data[p_data["is_input_window"] == 1]
        
        # Features with past forward-fill
        feat_vals = input_history[self.feature_cols].fillna(method="ffill").fillna(0.0).values
        seq_tensor = torch.tensor(feat_vals, dtype=torch.float32)
        
        # Targets: Future 30-day ctDNA and overall future progression trend
        # Note: Targets are taken from future window or sequence endpoints
        future_trend = int(p_data["future_progression_trend"].iloc[0])
        
        # 30-day forecast target from last historical input visit
        target_30d = input_history["future_ctDNA_30d_target"].dropna()
        target_30d_val = float(target_30d.iloc[-1]) if len(target_30d) > 0 else float(input_history["ctDNA_vaf_percent"].dropna().iloc[-1])
        
        return seq_tensor, torch.tensor(target_30d_val, dtype=torch.float32), torch.tensor(future_trend, dtype=torch.long), len(input_history)

def collate_temporal_batch(batch):
    sequences, targets_30d, trends, lengths = zip(*batch)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    batch_size, max_len, _ = padded_seqs.shape
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        attention_mask[i, :l] = 1
        
    return padded_seqs, torch.stack(targets_30d), torch.stack(trends), attention_mask

# Example:
# train_dataset = TemporalForecastingDatasetV2(split="train")
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_temporal_batch)
```

---

## 8. Handoff Checklist Completed

- [x] 1,000 unique patients partitioned with zero cross-split leakage (`stage-2-dl/data/v2/splits/`)
- [x] Exactly 12,000 pathology tiles organized by class (`stage-2-dl/data/v2/processed/pathology_tiles/`)
- [x] Master image metadata manifest (`stage-2-dl/data/v2/processed/image_metadata.csv`)
- [x] ~16,000 temporal biomarker sequence records (`stage-2-dl/data/v2/processed/biomarkers_processed.csv`)
- [x] Forecasting structure separating Historical Input (<=90d) and Future Prediction (>90d)
- [x] Anti-shortcut stroma and exposure randomization implemented and verified
- [x] Controlled missingness (3–8%) with binary indicator masks
- [x] Training-only augmentation pipeline with visual preview (`reports/augmentation_samples_comparison.png`)
- [x] 19-point automated validation audit completed and certified (`reports/data_engineering_report.md`)
- [x] Dataset statistics exported (`reports/dataset_statistics.csv`)
- [x] Comprehensive data dictionary with schemas and feature roles (`reports/data_dictionary.md`)
- [x] Automated unit test suite passing 100% (`tests/test_data_pipeline.py`)
- [x] Existing v1 data preserved untouched

The Stage 2 v2 dataset is certified ready for downstream deep learning model development by the Deep Learning Engineer.
