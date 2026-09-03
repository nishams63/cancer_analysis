# Stage 1 ML Integration Layer — Oncology Toxicity Risk Prediction Service

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Component**: Stage 1 Machine Learning — Inference Integration Layer  
**Frozen Model**: **Candidate Model V4** (`Regularized LightGBM`, `W = [1.0, 1.05, 1.05]`)  
**Interface**: Lightweight FastAPI REST Service  

---

## 1. System Architecture

```text
       Data Engineering Pipeline
                  ↓
       Master Patient Dataset
                  ↓
       Frozen Candidate V4 ML Model (Trained & Evaluated)
                  ↓
       Integration Layer (V4InferenceEngine)
                  ↓
       FastAPI REST Service (/health, /predict, /predict/batch)
                  ↓
       Predictions: Low / Moderate / High (+ Probabilities)
```

The Integration Engineer is responsible **only** for encapsulating the already-trained and evaluated Stage 1 ML model into a clean, reusable, production-ready inference interface.

> 🚨 **FROZEN MODEL POLICY**: Candidate V4 is completely frozen. The integration layer does not retrain, tune hyperparameters, modify feature calculations, or optimize decision rules.

---

## 2. Directory Layout

```text
stage-1-ml/integration/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── app.py               # FastAPI application with /health, /predict, /predict/batch
│   ├── predictor.py         # V4InferenceEngine encapsulating frozen artifacts
│   └── schemas.py           # Pydantic validation models for requests and responses
├── tests/
│   ├── test_api.py          # API endpoint tests (/health, /predict, /predict/batch)
│   ├── test_predictor.py    # Engine unit tests (loading, determinism, compatibility)
│   └── test_validation.py   # Schema validation tests (missing fields, bad types, novel categories)
├── examples/
│   ├── sample_request.json  # Valid 30-feature raw patient encounter JSON
│   └── sample_response.json # Example structured prediction response
├── requirements.txt         # Production and testing dependencies
└── README.md                # Comprehensive integration guide
```

---

## 3. Frozen V4 Artifact Locations

The integration layer automatically resolves paths relative to the repository root. No machine-specific hardcoded paths are used:

| Artifact | Repository Path | Description |
|:---|:---|:---|
| **Model** | `stage-1-ml/ml/models/best_model/model.joblib` | Frozen `ThresholdAdjustedClassifier` wrapping regularized LightGBM |
| **Preprocessor** | `stage-1-ml/ml/artifacts/preprocessor/preprocessor.joblib` | Frozen `PreprocessingArtifactManager` (ColumnTransformer with 113 features) |
| **Target Mapping** | `stage-1-ml/ml/artifacts/encoders/target_mapping.json` | Explicit target mapping (`Low: 0`, `Moderate: 1`, `High: 2`) |
| **Model Config** | `stage-1-ml/ml/results/v4_candidate_config.json` | Candidate V4 hyperparameter, feature, and decision-rule metadata |

*Note*: Paths can be overridden via environment variables: `V4_MODEL_PATH`, `V4_PREPROC_PATH`, `V4_ENCODERS_PATH`, `V4_CONFIG_PATH`.

---

## 4. End-to-End Inference Flow

```text
RAW PATIENT / ENCOUNTER INPUT (30 fields)
                 ↓
      INPUT VALIDATION (Pydantic)
                 ↓
EXISTING V4 FEATURE ENGINEERING (FeatureEngineer)
                 ↓
   EXISTING V4 PREPROCESSING (preprocessor.joblib)
                 ↓
      FROZEN V4 MODEL (model.joblib)
                 ↓
EXISTING V4 DECISION RULE (W = [1.0, 1.05, 1.05])
                 ↓
       Low / Moderate / High
                 ↓
STRUCTURED JSON RESPONSE (+ Calibrated Probabilities)
```

The caller provides only the **30 raw patient fields**. Engineered features (`cumulative_treatment_load`, `organ_impairment_index`, `vital_instability_score`, `genomic_instability_score`, `biomarker_severity_weight`) are computed automatically by the pipeline.

---

## 5. Required Input Fields (30 Raw Predictor Features)

### Numerical Features (20)
| Field Name | Type | Unit / Description | Valid Range |
|:---|:---:|:---|:---:|
| `age` | `float` | Patient age in years | $0 - 125$ |
| `mutation_burden` | `float` | Tumor mutational burden | $\ge 0$ |
| `gene_expression_score` | `float` | Gene expression score | $\ge 0$ |
| `ctdna_level` | `float` | Circulating tumor DNA level (ng/mL) | $\ge 0$ |
| `tumor_marker_level` | `float` | Serum tumor marker level (ng/mL) | $\ge 0$ |
| `inflammation_marker` | `float` | Inflammatory marker level (CRP/ESR) | $\ge 0$ |
| `heart_rate` | `float` | Resting heart rate (bpm) | $20 - 260$ |
| `systolic_bp` | `float` | Systolic blood pressure (mmHg) | $40 - 300$ |
| `diastolic_bp` | `float` | Diastolic blood pressure (mmHg) | $20 - 200$ |
| `oxygen_saturation` | `float` | Blood oxygen saturation SpO2 (%) | $50 - 100$ |
| `hemoglobin` | `float` | Hemoglobin level (g/dL) | $1.0 - 25.0$ |
| `white_blood_cell_count`| `float` | White blood cell count (k/uL) | $0.1 - 120.0$ |
| `platelet_count` | `float` | Platelet count (k/uL) | $5.0 - 2000.0$ |
| `creatinine_level` | `float` | Serum creatinine level (mg/dL) | $0.1 - 25.0$ |
| `liver_function_marker` | `float` | Liver enzymes ALT/AST (U/L) | $1.0 - 1500.0$ |
| `drug_dose` | `float` | Administered oncology drug dose (mg) | $\ge 0$ |
| `treatment_cycle` | `int` | Current treatment cycle number | $\ge 1$ |
| `previous_treatment_count`| `int` | Prior oncology treatment regimens | $\ge 0$ |
| `previous_toxicity_grade` | `float` | Prior adverse event toxicity grade | $0.0 - 5.0$ |
| `comorbidity_count` | `float` | Baseline comorbidity count | $\ge 0$ |

### Categorical Features (10)
| Field Name | Type | Description | Example Values |
|:---|:---:|:---|:---|
| `sex` | `str` | Biological sex | `"Female"`, `"Male"` |
| `cancer_type` | `str` | Primary cancer type | `"NSCLC"`, `"Breast Cancer"`, `"Colorectal"` |
| `cancer_stage` | `str` | Staging designation | `"Stage I"`, `"Stage II"`, `"Stage III"`, `"Stage IV"` |
| `smoking_history` | `str` | Tobacco use history | `"Never"`, `"Former"`, `"Current"` |
| `mutation_primary` | `str` | Primary detected mutation | `"EGFR"`, `"KRAS"`, `"BRAF"`, `"None"` |
| `mutation_secondary` | `str` | Secondary detected mutation | `"TP53"`, `"PIK3CA"`, `"None"` |
| `biomarker_trend` | `str` | Trajectory of tumor biomarkers | `"Increasing"`, `"Stable"`, `"Decreasing"` |
| `treatment_type` | `str` | Treatment modality | `"Chemotherapy"`, `"Immunotherapy"`, `"Targeted Therapy"` |
| `drug_name` | `str` | Generic drug name | `"Erlotinib"`, `"Pembrolizumab"`, `"Cisplatin"` |
| `previous_adverse_event` | `bool` | History of adverse events | `true`, `false` |

---

## 6. API Endpoints & Usage

### 1. `GET /health`
Verifies service status and confirms all frozen V4 artifacts are loaded into memory.

**Response (200 OK)**:
```json
{
  "status": "ok",
  "model_version": "V4",
  "artifacts_loaded": true,
  "timestamp": "2026-09-03 11:45:00"
}
```

---

### 2. `POST /predict`
Generates a toxicity risk prediction for a single patient encounter.

**Example Request**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @stage-1-ml/integration/examples/sample_request.json
```

**Example Payload**:
```json
{
  "age": 62.0,
  "sex": "Female",
  "cancer_type": "NSCLC",
  "cancer_stage": "Stage III",
  "smoking_history": "Former",
  "mutation_primary": "EGFR",
  "mutation_secondary": "TP53",
  "mutation_burden": 6.8,
  "gene_expression_score": 49.8,
  "ctdna_level": 1.4,
  "tumor_marker_level": 24.7,
  "inflammation_marker": 4.2,
  "biomarker_trend": "Stable",
  "heart_rate": 79.0,
  "systolic_bp": 128.0,
  "diastolic_bp": 80.0,
  "oxygen_saturation": 96.0,
  "hemoglobin": 12.5,
  "white_blood_cell_count": 7.41,
  "platelet_count": 231.0,
  "creatinine_level": 1.0,
  "liver_function_marker": 17.3,
  "treatment_type": "Targeted Therapy",
  "drug_name": "Erlotinib",
  "drug_dose": 150.0,
  "treatment_cycle": 4,
  "previous_treatment_count": 1,
  "previous_adverse_event": false,
  "previous_toxicity_grade": 0.0,
  "comorbidity_count": 1.0
}
```

**Example Response (200 OK)**:
```json
{
  "predicted_toxicity_risk": "Low",
  "probabilities": {
    "Low": 0.6753,
    "Moderate": 0.2445,
    "High": 0.0803
  },
  "model_version": "V4",
  "disclaimer": "This prediction is generated by a research decision-support prototype and does not constitute a clinical diagnosis or treatment recommendation."
}
```

---

### 3. `POST /predict/batch`
Evaluates a list of patient encounters in a single call.

**Example Request**:
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[{...}, {...}]'
```

**Example Response (200 OK)**:
```json
{
  "predictions": [
    {
      "predicted_toxicity_risk": "Low",
      "probabilities": { "Low": 0.6753, "Moderate": 0.2445, "High": 0.0803 },
      "model_version": "V4",
      "disclaimer": "..."
    }
  ],
  "total_records": 1,
  "model_version": "V4"
}
```

---

## 7. Input Validation & Error Handling

If any required field is missing or an incompatible data type is supplied, the API rejects the request with `HTTP 422 Unprocessable Content` and returns an actionable error description:

**Example Error Response (Missing `hemoglobin` field)**:
```json
{
  "error": "Input validation failed",
  "details": [
    "Field 'hemoglobin': Field required"
  ]
}
```

**Handling Unknown Categorical Values**:
If a patient has a previously unseen category (e.g. `cancer_type: "RareSarcomaSubtype"`), the frozen `OneHotEncoder` safely encodes that feature as all zeros (`handle_unknown='ignore'`), allowing graceful inference without service interruption.

---

## 8. Installation, Running, & Testing

### Installation
```bash
cd stage-1-ml/integration
pip install -r requirements.txt
```

### Running the API Server
```bash
# Start uvicorn server on port 8000 with auto-reload
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at: `http://localhost:8000/docs` (Swagger UI).

### Running Integration Tests
```bash
# Run all integration tests
pytest tests/ -v

# Run the complete project test suite (40/40 tests)
pytest stage-1-ml/ -v
```

---

## 9. Clinical Safety Disclaimer

This system is a **machine learning research decision-support prototype**. It has not been approved by regulatory agencies for autonomous clinical diagnosis, treatment selection, or chemotherapy dosage determination. All predictions must be reviewed by qualified oncology professionals in accordance with institutional clinical protocols.
