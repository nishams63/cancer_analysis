# Stage 2 (v2) Data Dictionary: Schemas, Feature Roles & Forecasting Protocol

> [!IMPORTANT]
> **MANDATORY CLINICAL & SYNTHETIC DATA NOTICE:**  
> **Synthetic data created for deep-learning research and pipeline prototyping. These data are not real patient records and are not clinically validated.**  
> $$\text{Synthetic data} \ne \text{Real patient evidence} \ne \text{Clinical validation}$$  
> All records, images, and time-series sequences in this dataset are procedurally synthesized for machine learning engineering, batching verification, and model development. Do not cite these values as real clinical benchmarks.

---

## 1. Patient Cohort & Split Manifests (`stage-2-dl/data/v2/splits/`)

Files:
- `train_patients.csv` (700 unique patients)
- `validation_patients.csv` (150 unique patients)
- `test_patients.csv` (150 unique patients)

| Column Name | Data Type | Role | Allowed Values / Range | Description | Historical Usable? |
|---|---|---|---|---|---|
| `patient_id` | String | Metadata | `PAT-0001` to `PAT-1000` | Primary key and unit of isolation. | Yes |
| `cancer_type` | String | Metadata / Feature | `NSCLC` | Non-Small Cell Lung Cancer cohort identifier. | Yes |
| `clinical_stage` | String | Metadata / Feature | `Stage IIIA`, `Stage IIIB`, `Stage IV` | Intake clinical stage. | Yes |
| `age` | Integer | Feature | 45 – 79 | Patient age at study intake. | Yes |
| `biomarker_trajectory` | String | Metadata / Target | 5 trajectory archetypes | Ground truth trajectory class for stratification & multi-task targets. | No (Overall target) |
| `split` | String | Metadata | `train`, `validation`, `test` | Stratified partition with verified 0% overlap. | No |
| `data_source` | String | Metadata | `synthetic` | Explicit synthetic provenance marker. | Yes |
| `disclaimer` | String | Metadata | Standard text | Regulatory notice. | Yes |

---

## 2. Visual Histopathology Tiles Manifest (`stage-2-dl/data/v2/processed/image_metadata.csv`)

Total Records: 12,000 tiles (Exactly 12 tiles per patient)  
Directory: `stage-2-dl/data/v2/processed/pathology_tiles/{class_label}/`

| Column Name | Data Type | Role | Allowed Values / Range | Description | Anti-Shortcut Purpose |
|---|---|---|---|---|---|
| `patient_id` | String | Metadata | `PAT-0001` to `PAT-1000` | Links tile to patient. Strictly matches patient split. | N/A |
| `tile_id` | String | Metadata | `PAT-XXXX_SLIDE_YY_TILE_ZZ` | Unique tile identifier. | N/A |
| `slide_id` | String | Metadata | `PAT-XXXX_SLIDE_01`, `02` | Biopsy whole-slide identifier (2 slides/patient). | N/A |
| `class_label` | String | **Primary Target** | `benign`, `malignant`, `inflammation` | Morphological classification target for CNN. | Target |
| `split` | String | Metadata | `train`, `validation`, `test` | Train: 8,400, Val: 1,800, Test: 1,800. | Disjoint |
| `image_path` | String | Metadata | Absolute Path | Absolute path to 224x224x3 PNG on disk. | N/A |
| `relative_path` | String | Metadata | Relative Path | Relative repository path. | N/A |
| `file_name` | String | Metadata | String | Image file name. | N/A |
| `height`, `width` | Integer | Metadata | 224, 224 | Image dimensions. Standard across all classes. | Eliminates size shortcut |
| `channels` | Integer | Metadata | 3 | Standard RGB uint8 color channels. | Eliminates band shortcut |
| `background_temperature` | String | Generation Meta | `warm`, `neutral`, `cool` | Stroma tint randomized independently across all classes. | Prevents color tint shortcut |
| `brightness_factor`| Float | Generation Meta | 0.92 – 1.08 | Exposure variance randomized across all classes. | Prevents exposure shortcut |
| `contrast_factor` | Float | Generation Meta | 0.90 – 1.10 | Contrast variance randomized across all classes. | Prevents contrast shortcut |
| `stain_variation` | Float | Generation Meta | 0.78 – 1.28 | Ratio of hematoxylin to eosin gains. | Prevents staining shortcut |
| `augmentation_status` | String | Metadata | `original_unaugmented` | Confirms base tiles are uncorrupted. | Auditing |
| `qc_status` | String | Metadata | `PASS` | Quality filter check (non-blank, non-corrupt). | QA |
| `data_source` | String | Metadata | `synthetic` | Explicit synthetic provenance marker. | N/A |

---

## 3. Longitudinal Biomarkers & Forecasting Schema (`stage-2-dl/data/v2/processed/biomarkers_processed.csv`)

Total Records: ~16,000 observations (14–18 timepoints per patient)

| Column Name | Data Type | Modeling Role | Permitted Range | Description | Usable During Input Window (<=90d)? |
|---|---|---|---|---|---|
| `patient_id` | String | Metadata | `PAT-0001` to `PAT-1000` | Sequence grouping identifier. | Yes |
| `split` | String | Metadata | `train`, `validation`, `test` | Inherited from patient partition. | Yes |
| `timepoint_index` | Integer | Feature | 0 – 17 | Visit index ($t_0, t_1, \dots$). | Yes |
| `timestamp` | Date | Feature / Meta | `2026-01-01` to `2026-08-30` | Date of clinical assessment. | Yes |
| `days_from_baseline` | Integer | Feature | 0 – 220+ | Days elapsed since baseline ($t_0 = 0$). Monotonic. | Yes |
| `delta_days` | Integer | Feature | 0, 11 – 16 | Days since previous visit ($\Delta t$). | Yes |
| `ctDNA_vaf_percent` | Float / NaN | **Input & Target** | 0.05% – 15.0% | Circulating Tumor DNA Variant Allele Frequency. | Yes (in input window) |
| `cea_ng_ml` | Float / NaN | Input Feature | 0.5 – 95.0 ng/mL | Carcinoembryonic Antigen level. | Yes (in input window) |
| `ca125_u_ml` | Float / NaN | Input Feature | 5.0 – 120.0 U/mL | Cancer Antigen 125 level. | Yes (in input window) |
| `ldh_u_l` | Float / NaN | Input Feature | 120.0 – 800.0 U/L | Lactate Dehydrogenase level. | Yes (in input window) |
| `crp_mg_l` | Float / NaN | Input Feature | 0.5 – 25.0 mg/L | C-Reactive Protein inflammatory level. | Yes (in input window) |
| `ctDNA_velocity_30d` | Float | Input Feature | -12.0 to +15.0 | Rate of change per 30 days: $((\text{ctDNA}_t - \text{ctDNA}_{t-1})/\Delta t) \times 30$. | Yes (in input window) |
| `ctDNA_missing` | Integer (0/1) | Input Feature / Mask | 0 or 1 | 1 indicates assay missing/failed; value is NaN. | Yes |
| `cea_missing` | Integer (0/1) | Input Feature / Mask | 0 or 1 | Missingness mask for CEA. | Yes |
| `ca125_missing` | Integer (0/1) | Input Feature / Mask | 0 or 1 | Missingness mask for CA-125. | Yes |
| `ldh_missing` | Integer (0/1) | Input Feature / Mask | 0 or 1 | Missingness mask for LDH. | Yes |
| `crp_missing` | Integer (0/1) | Input Feature / Mask | 0 or 1 | Missingness mask for CRP. | Yes |
| `window_type` | String | Protocol Tag | `historical_input`, `future_prediction` | Forecasting window marker. Boundary at Day 90. | Yes |
| `is_input_window` | Integer (0/1) | Protocol Tag | 1 (<=90d), 0 (>90d) | Binary indicator for input window. | Yes |
| `future_ctDNA_30d_target` | Float / NaN | **Forecast Target** | 0.05% – 15.0% | Actual ctDNA VAF at approximately $t + 30$ days. **TARGET ONLY**. | **NO — NEVER USE AS INPUT** |
| `future_progression_trend` | Integer (0/1) | **Forecast Target** | 0 (Stable/Response) or 1 (Progression) | Future progression status. **TARGET ONLY**. | **NO — NEVER USE AS INPUT** |
| `trajectory_pattern` | String | Meta / Target | 5 archetypes | Ground truth archetype label. | **NO — Evaluation only** |
| `data_source` | String | Metadata | `synthetic` | Explicit synthetic provenance marker. | Yes |

---

## 4. Forecasting Protocol & Anti-Leakage Rules

### Temporal Horizon Definition:
$$\text{Historical Input Window}: \quad \text{days\_from\_baseline} \le 90 \quad (\text{is\_input\_window} = 1)$$
$$\text{Future Prediction Window}: \quad \text{days\_from\_baseline} > 90 \quad (\text{is\_input\_window} = 0)$$

### Strict Modeling Rules for the DL Engineer:
1. **Inputs strictly bounded:** When training an LSTM or Transformer to predict 30-day forward ctDNA or future progression, only observations with `is_input_window == 1` may be fed as sequence inputs.
2. **Target Isolation:** The columns `future_ctDNA_30d_target` and `future_progression_trend` are **ground-truth prediction targets**. They must NEVER be concatenated into input feature tensors.
3. **Missing Value Handling:** Missing values (~3–8%) are represented as `NaN` with companion binary missingness indicator masks (`ctDNA_missing == 1`). DL models should use masked attention or imputation based strictly on past observed values.
