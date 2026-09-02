# Data Dictionary — Master Patient Dataset

**Dataset File**: `data/processed/master_patient_dataset.csv`  
**Stage**: Stage 1 — Machine Learning  
**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  

---

| Column Name | Data Type | Description | Example Value | ML Role | Missing Value Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `patient_id` | `string` | Synthetic unique patient identifier | `PT-001895` | Identifier | None (Required) |
| `encounter_id` | `string` | Synthetic unique encounter / visit identifier | `ENC85214069` | Identifier | None (Required) |
| `observation_date` | `string` | Observation date formatted as `YYYY-MM-DD` | `2024-04-16` | Metadata / Feature | Forward-filled per patient encounter |
| `age` | `float` | Patient age in years | `62.0` | Feature | Median Imputation |
| `sex` | `category` | Biological sex (`Male`, `Female`, `Unknown`) | `Female` | Feature | Categorical `'Unknown'` |
| `cancer_type` | `category` | Primary cancer diagnosis (e.g. `NSCLC`, `Colorectal Cancer`) | `NSCLC` | Feature | Categorical `'Unknown'` |
| `cancer_stage` | `category` | Cancer staging (`Stage I`, `Stage II`, `Stage III`, `Stage IV`) | `Stage III` | Feature | Categorical `'Unknown'` |
| `smoking_history` | `category` | Smoking status (`Current`, `Former`, `Never`, `Unknown`) | `Former` | Feature | Categorical `'Unknown'` |
| `mutation_primary` | `category` | Primary driver genomic mutation (e.g. `EGFR`, `KRAS`, `TP53`) | `EGFR` | Feature | Explicit `'None/Unknown'` |
| `mutation_secondary` | `category` | Secondary co-occurring mutation | `TP53` | Feature | Explicit `'None/Unknown'` |
| `mutation_burden` | `float` | Tumor mutational burden (mutations/Mb) | `6.82` | Feature | Median Imputation |
| `gene_expression_score` | `float` | Gene expression composite score | `49.82` | Feature | Median Imputation |
| `ctdna_level` | `float` | Circulating tumor DNA concentration (ng/mL) | `1.40` | Feature | Median Imputation |
| `tumor_marker_level` | `float` | Serum tumor marker concentration | `24.73` | Feature | Median Imputation |
| `inflammation_marker` | `float` | Systemic inflammation marker level | `4.22` | Feature | Median Imputation |
| `biomarker_trend` | `category` | Serial biomarker trajectory (`Increasing`, `Stable`, `Decreasing`) | `Stable` | Feature | Categorical `'Unknown'` |
| `heart_rate` | `float` | Resting heart rate (beats per minute) | `79.0` | Feature | Median Imputation |
| `systolic_bp` | `float` | Systolic blood pressure (mmHg) | `128.0` | Feature | Median Imputation |
| `diastolic_bp` | `float` | Diastolic blood pressure (mmHg) | `80.0` | Feature | Median Imputation |
| `oxygen_saturation` | `float` | Blood oxygen saturation percentage (%) | `96.0` | Feature | Median Imputation |
| `hemoglobin` | `float` | Hemoglobin blood concentration (g/dL) | `12.5` | Feature | Median Imputation |
| `white_blood_cell_count` | `float` | White blood cell count (k/µL) | `7.41` | Feature | Median Imputation |
| `platelet_count` | `float` | Blood platelet count (k/µL) | `231.0` | Feature | Median Imputation |
| `creatinine_level` | `float` | Serum creatinine level (mg/dL) | `1.00` | Feature | Median Imputation |
| `liver_function_marker` | `float` | Liver enzyme / function marker level | `17.3` | Feature | Median Imputation |
| `treatment_type` | `category` | Modality of therapy (`Chemotherapy`, `Immunotherapy`, etc.) | `Targeted Therapy` | Feature | Categorical `'Unknown'` |
| `drug_name` | `category` | Administered drug or regimen name | `Erlotinib` | Feature | Categorical `'Unknown'` |
| `drug_dose` | `float` | Administered treatment dosage (mg) | `150.3` | Feature | Median Imputation |
| `treatment_cycle` | `int` | Current treatment cycle number | `4` | Feature | None (Complete) |
| `previous_treatment_count` | `int` | Count of prior lines of therapy | `1` | Feature | None (Complete) |
| `previous_adverse_event` | `boolean` | History of prior treatment adverse event (`True`/`False`) | `False` | Feature | Conservative `False` Imputation |
| `previous_toxicity_grade` | `float` | Grade of prior toxicity (0 to 4) | `0.0` | Feature | Median Imputation |
| `comorbidity_count` | `float` | Count of patient comorbidities | `1.0` | Feature | Median Imputation |
| `treatment_response` | `category` | Best overall response (`Complete Response`, `Partial Response`, etc.) | `Partial Response` | Feature | Categorical `'Unknown'` |
| `toxicity_risk` | `category` | ML Target: Treatment Toxicity Risk (`Low`, `Moderate`, `High`) | `Low` | ML Target | None (Target is 100% complete) |

---
