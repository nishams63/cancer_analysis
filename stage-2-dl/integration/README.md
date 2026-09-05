# Stage 2 Deep Learning: Multimodal Inference Integration

> [!IMPORTANT]
> **MANDATORY CLINICAL & SYNTHETIC DATA NOTICE:**  
> **This model was developed using synthetic data and has not been clinically validated. Performance on this dataset does not establish clinical effectiveness, safety, or real-world patient benefit.**  
> $$\text{Synthetic data} \ne \text{Real patient evidence} \ne \text{Clinical validation}$$  
> This integration system is a synthetic-data research prototype developed for deep learning engineering, API verification, and autonomous agent staging. It does NOT prescribe medical treatments and must NEVER be used for clinical decision-making.

---

## 1. Overview & Architecture

The **Integration Module** (`stage-2-dl/integration/`) unites the two decoupled Stage 2 Deep Learning models into a single, patient-level multimodal inference service:

```text
                                +-----------------------------------+
                                |    Patient Case (patient_id)      |
                                +-----------------+-----------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
        +---------------------------------+               +---------------------------------+
        |   Biopsy Pathology Tiles (WSI)  |               |  Longitudinal Biomarkers (<=90d)|
        +----------------+----------------+               +----------------+----------------+
                         |                                                 |
                         v                                                 v
        +---------------------------------+               +---------------------------------+
        | Pathology CNN (ResNet-18 Frozen)|               |  Temporal BiLSTM (Multi-Task)   |
        |  `best_pathology_cnn.pt`        |               |   `best_temporal_lstm.pt`       |
        +----------------+----------------+               +----------------+----------------+
                         |                                                 |
                         v                                                 v
        +---------------------------------+               +---------------------------------+
        |  Patient Tile Aggregation       |               | Dual Sequence Heads:            |
        |  (mean / median / max)          |               | - Progression Risk: P_prog      |
        |  -> P_malignant in [0, 1]       |               | - Forecast 30d ctDNA VAF (%)    |
        +----------------+----------------+               +----------------+----------------+
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
                               +------------------------------------+
                               |     Multimodal Fusion Engine       |
                               |  - Weighted Linear (default)       |
                               |  - Rule-Based Decision Logic       |
                               |  - Missing Modality Fallbacks      |
                               +------------------+-----------------+
                                                  |
                                                  v
                               +------------------------------------+
                               |    Standardized Patient Result     |
                               | - Prototype Multimodal Risk Score  |
                               | - Prototype Alert Level            |
                               | - Non-Causal Engineering Provenance|
                               +------------------------------------+
```

---

## 2. Component Contributions

### What Does the Pathology Model Contribute?
- Analyzes multiple whole-slide biopsy tiles ($224 \times 224 \times 3$ RGB) sampled from biopsy specimens.
- Identifies cellular morphology patterns (`benign`, `malignant`, `inflammation`).
- Aggregates tile-level probabilities across all patient biopsy tiles into a single patient-level **$P_{\text{malignant}}$** probability.

### What Does the Temporal Model Contribute?
- Processes historical longitudinal blood biomarker time-series strictly within the induction window ($\text{days} \le 90$).
- Tracks ctDNA VAF, CEA, CA-125, LDH, CRP, delta days, and backward-looking ctDNA velocity across 13 input channels.
- Forecasts two key forward signals:
  1. **Disease Progression Probability ($P_{\text{progression}}$):** Risk of progression beyond Day 90.
  2. **Quantitative 30-Day ctDNA VAF:** Forward continuous prediction of circulating tumor DNA allele fraction (%).

### How Does Multimodal Fusion Work?
The pipeline scales ctDNA VAF into a normalized $[0, 1]$ risk index and computes the **prototype multimodal risk score**:
$$\text{prototype\_multimodal\_risk\_score} = w_1 \cdot P_{\text{malignant}} + w_2 \cdot P_{\text{progression}} + w_3 \cdot \text{ctDNA\_risk}$$

Default prototype engineering weights:
- $w_1 = 0.35$ (Pathology Malignancy)
- $w_2 = 0.40$ (Temporal Progression Risk)
- $w_3 = 0.25$ (Normalized ctDNA VAF Index)

The score maps to a prototype engineering alert level:
- **`LOW (Prototype)`**: Score $< 0.35$
- **`MODERATE (Prototype)`**: Score between $0.35$ and $0.70$
- **`HIGH (Prototype)`**: Score $\ge 0.70$

> [!CAUTION]
> **Fusion Validation Rule:** These weights and alert thresholds are **prototype engineering parameters only** and are **NOT** clinically validated. They were not optimized on the locked test set.

---

## 3. Missing Modality Handling

In real clinical settings, a patient may not have both a biopsy and longitudinal biomarker tests available at the same time. The integration pipeline explicitly supports partial modalities:

| Available Data | Modality Status | Fusion Behavior |
|---|:---:|---|
| **Pathology + Temporal** | `FULL_MULTIMODAL` | Evaluates complete 3-component weighted score. |
| **Pathology Only** | `PATHOLOGY_ONLY` | Single-modality score equals $P_{\text{malignant}}$. Tagged explicitly as partial. |
| **Temporal Only** | `TEMPORAL_ONLY` | Re-weighted score: $0.60 \cdot P_{\text{progression}} + 0.40 \cdot \text{ctDNA\_risk}$. Tagged explicitly as partial. |
| **Neither** | `INSUFFICIENT_DATA` | Score is `None`. Tagged `INSUFFICIENT_DATA`. |

---

## 4. Quick Start & Execution Commands

### 1. Run Command-Line Inference for a Patient
Evaluate any patient from the development cohort:
```powershell
python stage-2-dl/integration/src/integration_pipeline.py --patient-id PAT-0001
```
Options:
- `--aggregation [mean|median|max]`: Tile aggregation method (default: `mean`).
- `--fusion [weighted_linear|rule_based]`: Fusion strategy (default: `weighted_linear`).

### 2. Launch the Interactive Research Dashboard
```powershell
streamlit run stage-2-dl/integration/dashboard/app.py
```
*Features:* Patient selector, Whole-Slide tile gallery, longitudinal trajectory plots, interactive weight sliders, and missing-modality toggles.

### 3. Generate a Standalone Offline HTML Report
Generate an interactive report without a web server:
```powershell
python stage-2-dl/integration/dashboard/standalone_viewer.py --patient-id PAT-0001
```
Saved to: `stage-2-dl/integration/reports/patient_PAT-0001_report.html`.

### 4. Launch the REST API Service
```powershell
python stage-2-dl/integration/src/api.py
```
*Endpoints:*
- `GET http://localhost:8000/health`: Health status and loaded model paths.
- `GET http://localhost:8000/info`: Active weights and thresholds.
- `POST http://localhost:8000/predict/patient`: Full multimodal inference payload.
- `POST http://localhost:8000/predict/sample/PAT-0001`: Demo endpoint.

### 5. Run the Automated Test Suite
```powershell
python -m pytest stage-2-dl/integration/tests/test_integration.py -v
```

---

## 5. Directory Structure

```text
stage-2-dl/integration/
├── dashboard/
│   ├── __init__.py
│   ├── app.py                     # Interactive Streamlit dashboard
│   └── standalone_viewer.py       # Offline HTML report generator
├── reports/
│   ├── integration_report.md      # Comprehensive technical integration report
│   └── patient_PAT-0001_report.html # Sample patient HTML report
├── src/
│   ├── __init__.py
│   ├── config.py                  # Checkpoint paths, prototype weights, disclaimers
│   ├── validation.py              # Anti-leakage checks (days <= 90), target exclusion
│   ├── patient_aggregation.py     # Tile-level inference and aggregation (mean/median/max)
│   ├── fusion.py                  # Multimodal fusion strategies & missing modality logic
│   ├── integration_pipeline.py    # Main MultimodalPatientPipeline class & CLI
│   └── api.py                     # FastAPI REST service with Pydantic validation
├── tests/
│   ├── __init__.py
│   └── test_integration.py        # 12 automated unit tests
├── requirements.txt
└── README.md
```

---

## 6. Downstream Handoff Toward Stage 6 Agentic AI

The integrated multimodal pipeline is packaged for **Stage 6 Agentic Reasoning**:
- **Callable Interface:**
  ```python
  from stage_2_dl.integration.src.integration_pipeline import run_patient_inference
  result = run_patient_inference(patient_id="PAT-0001", pathology_tiles=tiles, temporal_history=bio_df)
  ```
- **Audited Outputs:** Stage 6 agents consume standardized JSON containing `malignant_probability`, `progression_probability`, `predicted_ctDNA_30d_vaf`, and `prototype_multimodal_risk_score`.
- **Constraint:** Agents must cite evidence objectively and must never treat prototype alert categories as validated clinical diagnoses.
