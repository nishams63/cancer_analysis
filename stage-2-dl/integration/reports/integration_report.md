# Stage 2 Deep Learning: Multimodal Inference Integration Report

**Role:** Stage 2 Integration Engineer  
**Status:** `INTEGRATION COMPLETE — READY FOR STAGE 6 AGENTIC REASONING`  
**Date:** September 5, 2026  
**Artifact Directory:** `stage-2-dl/integration/`  

---

> [!IMPORTANT]
> **MANDATORY CLINICAL & SYNTHETIC DATA NOTICE:**  
> **This model was developed using synthetic data and has not been clinically validated. Performance on this dataset does not establish clinical effectiveness, safety, or real-world patient benefit.**  
> $$\text{Synthetic data} \ne \text{Real patient evidence} \ne \text{Clinical validation}$$  
> This integration system is a synthetic-data research prototype and has not been clinically validated. All outputs are prototype engineering scores designed for pipeline verification, API batching, and autonomous agent input staging. Never present this system as an autonomous clinical decision-support tool or treatment prescriber.

---

## 1. Objective

The objective of Stage 2 Multimodal Integration is to unite the two decoupled, frozen Stage 2 Deep Learning models into a single, cohesive, patient-level inference pipeline:
1. **Model A (Pathology CNN):** ResNet-18 whole-slide biopsy tile classifier predicting cellular morphology probabilities (`benign`, `malignant`, `inflammation`).
2. **Model B (Temporal BiLSTM):** Longitudinal multi-task recurrent sequence forecaster predicting forward 30-day ctDNA VAF (`predicted_ctDNA_30d_vaf`) and disease progression risk (`predicted_progression_risk`).

The integrated pipeline produces a standardized patient-level output combining visual malignancy, longitudinal progression risk, and ctDNA kinetics into an audited **prototype multimodal risk score** and **prototype alert category** for downstream Stage 6 agentic reasoning.

---

## 2. Frozen Models Used & Checkpoint Provenance

The integration layer operates strictly as an orchestration and inference consumer over the frozen checkpoints trained in Stage 2 DL and evaluated in Stage 2 Evaluation:
- **Pathology Vision Checkpoint:** `stage-2-dl/dl/checkpoints/best_pathology_cnn.pt` (45.5 MB, ResNet-18 transfer learning backbone with custom 3-way classification head).
- **Temporal BiLSTM Checkpoint:** `stage-2-dl/dl/checkpoints/best_temporal_lstm.pt` (1.8 MB, 2-layer Bidirectional LSTM, hidden dimension 64, dual regression and classification heads).
- **Immutability Guarantee:** Neither checkpoint was modified, retrained, calibrated, or fine-tuned. Checkpoint files remain 100% bit-identical to their frozen states.

---

## 3. Model Input & Output Contracts

### Pathology CNN Contract
- **Input:** $224 \times 224 \times 3$ uint8 RGB image tensor (normalized with ImageNet statistics: $\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
- **Direct Output:** 3-class softmax probability distribution $[P_{\text{benign}}, P_{\text{malignant}}, P_{\text{inflammation}}]$.
- **Integrated Target:** Patient-level aggregated $P_{\text{malignant}} \in [0.0, 1.0]$.

### Temporal BiLSTM Contract
- **Input:** Longitudinal sequence tensor $\mathbf{X} \in \mathbb{R}^{B \times L \times 13}$ where $L \le 10$ and all observations strictly satisfy $\text{days\_from\_baseline} \le 90$.
  - 8 Continuous Features: `ctDNA_vaf_percent`, `cea_ng_ml`, `ca125_u_ml`, `ldh_u_l`, `crp_mg_l`, `ctDNA_velocity_30d`, `delta_days`, `days_from_baseline` (standardized using frozen training moments).
  - 5 Binary Missingness Masks: `ctDNA_missing`, `cea_missing`, `ca125_missing`, `ldh_missing`, `crp_missing`.
- **Direct Output:**
  - Head A: Predicted 30-day forward ctDNA VAF $\hat{y}_{\text{ctDNA}} \ge 0.0\%$ (continuous float).
  - Head B: Logits $\rightarrow$ Sigmoid progression probability $P_{\text{progression}} \in [0.0, 1.0]$ and binary class $\hat{z} \in \{0, 1\}$.

---

## 4. Patient-Level Alignment & Primary Keys

Integration is strictly enforced at the **patient level** using `patient_id` as the primary key:
- Every pathology tile is mapped to its parent `patient_id` via biopsy slide metadata.
- Every longitudinal biomarker visit is chronologically grouped by `patient_id`.
- The pipeline verifies patient alignment across modalities before executing fusion, rejecting foreign tiles or misaligned time-series.
- Cross-patient contamination is prevented through isolated execution contexts.

---

## 5. Tile Aggregation Strategy

In clinical practice, a single biopsy specimen yields multiple Whole-Slide Image (WSI) tiles. In Stage 2 Data Engineering, each patient is represented by 12 tiles (4 benign, 4 malignant, 4 inflammation per baseline protocol).

The integration layer avoids naive row-level concatenation and implements structured patient-level aggregation:
$$P_{\text{malignant\_patient}} = \text{Aggregate}\left(P(\text{malignant} \mid \text{tile}_1), \dots, P(\text{malignant} \mid \text{tile}_K)\right)$$

### Supported Aggregation Methods:
1. **`mean` (Default Baseline):**
   Computes the arithmetic mean of class probabilities across all $K$ tiles:
   $$P_{\text{class\_patient}} = \frac{1}{K} \sum_{k=1}^K P(\text{class} \mid \text{tile}_k)$$
   Provides a stable, balanced estimate of overall tumor burden across sampled fields.
2. **`median`:**
   Computes the median class probability, offering robustness against outlier tiles or staining artifacts.
3. **`max`:**
   Extracts the maximum malignant probability across tiles, representing a "worst-case field" surveillance strategy.

### Dispersion & Audit Tracking:
The aggregator computes tile-level variance and dispersion ($\sigma_{\text{mal}}, \min_{\text{mal}}, \max_{\text{mal}}$) and retains individual tile predictions in `tile_details` for complete clinical auditability.

---

## 6. Temporal Input Handling & Anti-Leakage Enforcement

The temporal model receives strictly historical observations:
$$\forall t, \quad \text{days\_from\_baseline} \le 90 \quad (\text{is\_input\_window} = 1)$$

### Anti-Leakage Safeguards:
1. **Horizon Boundary Assertion:** The validator (`validation.py`) programmatically inspects the incoming DataFrame and raises a `ValidationError` if any observation exceeds Day 90.
2. **Forbidden Target Exclusion:** Input columns are checked against forbidden target names (`future_ctDNA_30d_target`, `future_progression_trend`, `trajectory_pattern`). If present, they are either stripped or rejected.
3. **ctDNA Velocity Verification:** The `ctDNA_velocity_30d` feature is verified as strictly backward-looking:
   $$\text{velocity}_t = \frac{\text{ctDNA}_t - \text{ctDNA}_{t-1}}{\text{days}_t - \text{days}_{t-1}} \times 30.0$$
   At $t=0$, velocity is $0.0$. Zero future information is admitted into the input tensor.

---

## 7. Multimodal Fusion Design

The integration pipeline unifies the three core signals:
1. Patient-level aggregated malignant probability $P_{\text{mal}} \in [0, 1]$.
2. Longitudinal disease progression probability $P_{\text{prog}} \in [0, 1]$.
3. Normalized ctDNA VAF risk index $\text{ctDNA\_risk} \in [0, 1]$.

### ctDNA Risk Normalization:
Continuous ctDNA VAF (%) is normalized into a $[0, 1]$ index using linear dynamic range scaling:
$$\text{ctDNA\_risk} = \text{clip}\left(\frac{\hat{y}_{\text{ctDNA}} - \text{VAF}_{\min}}{\text{VAF}_{\max} - \text{VAF}_{\min}}, 0.0, 1.0\right)$$
where $\text{VAF}_{\min} = 0.0\%$ and $\text{VAF}_{\max} = 5.0\%$ (matching the 99th percentile of the synthetic distribution).

### Fusion Strategy A: `weighted_linear` (Default Baseline)
$$\text{prototype\_multimodal\_risk\_score} = w_1 \cdot P_{\text{mal}} + w_2 \cdot P_{\text{prog}} + w_3 \cdot \text{ctDNA\_risk}$$
Default prototype weights:
$$w_1 = 0.35 \quad (\text{Pathology}), \quad w_2 = 0.40 \quad (\text{Progression}), \quad w_3 = 0.25 \quad (\text{ctDNA VAF})$$

### Fusion Strategy B: `rule_based`
A conservative heuristic logic:
- `HIGH (Prototype)`: if both $P_{\text{mal}} \ge 0.70$ and $P_{\text{prog}} \ge 0.70$.
- `LOW (Prototype)`: if both $P_{\text{mal}} < 0.35$ and $P_{\text{prog}} < 0.35$ and $\text{ctDNA\_risk} < 0.35$.
- `MODERATE (Prototype)`: otherwise.

---

## 8. Fusion Weights, Thresholds & Test-Set Protection

> [!CAUTION]
> **PROTOTYPE ENGINEERING PARAMETERS NOTICE:**
> - The default fusion weights ($0.35, 0.40, 0.25$) and alert thresholds ($0.35, 0.70$) are **prototype engineering parameters only**.
> - They are **NOT** clinically validated.
> - They were **NOT** optimized, tuned, or fitted on the locked test set (`test_patients.csv`).
> - The categories `LOW (Prototype)`, `MODERATE (Prototype)`, and `HIGH (Prototype)` represent technical engineering groupings, **NOT** clinical treatment thresholds.
> - Any future calibration or learned meta-model must be fitted strictly on development/training splits, preserving the locked test set for final independent evaluation.

### Alert Threshold Mapping:
| Score Interval | Prototype Alert Category | Engineering Meaning |
|:---:|:---:|---|
| $[0.00, 0.35)$ | **`LOW (Prototype)`** | Low visual malignancy and stable/decreasing longitudinal biomarker trajectory. |
| $[0.35, 0.70)$ | **`MODERATE (Prototype)`** | Mixed signals (e.g. high visual malignancy but stable biomarkers, or vice versa). |
| $[0.70, 1.00]$ | **`HIGH (Prototype)`** | Concordant high visual malignancy and rising longitudinal biomarker trajectory. |

---

## 9. Missing-Modality Handling

Clinical real-world scenarios frequently present incomplete data. The integration pipeline explicitly handles all four modality availability states without hallucinating fake data:

| Availability Case | Modality Status | Fusion Behavior | Resulting Output |
|---|:---:|---|---|
| **Pathology + Temporal** | `FULL_MULTIMODAL` | Evaluates all 3 components using default or custom weights. | Full multimodal score $[0, 1]$ |
| **Pathology Only** | `PATHOLOGY_ONLY` | Evaluates single-modality score equal to $P_{\text{malignant}}$. Tagged explicitly as partial. | Single-modality score $[0, 1]$ |
| **Temporal Only** | `TEMPORAL_ONLY` | Re-weights temporal signals: $0.60 \cdot P_{\text{prog}} + 0.40 \cdot \text{ctDNA\_risk}$. Tagged explicitly as partial. | Single-modality score $[0, 1]$ |
| **Neither Modality** | `INSUFFICIENT_DATA` | Score is `None`. Alert level is `INSUFFICIENT_DATA`. | Rejected / Null score |

---

## 10. Standardized Integration Output Schema

The pipeline returns a standardized JSON/dictionary conforming to the following structure:

```json
{
  "patient_id": "PAT-0001",
  "modality_availability": {
    "pathology": true,
    "temporal": true
  },
  "modality_status": "FULL_MULTIMODAL",
  "pathology_summary": {
    "available": true,
    "num_tiles_analyzed": 12,
    "aggregation_method": "mean",
    "malignant_probability": 0.3344,
    "benign_probability": 0.3342,
    "inflammation_probability": 0.3314,
    "primary_patient_class": "malignant",
    "tile_predictions_breakdown": {
      "benign": 4,
      "malignant": 4,
      "inflammation": 4
    },
    "malignant_dispersion": {
      "std": 0.4706,
      "min": 0.0,
      "max": 1.0
    },
    "status": "SUCCESS"
  },
  "temporal_summary": {
    "available": true,
    "input_sequence_length": 7,
    "max_days_from_baseline": 83,
    "progression_probability": 0.0015,
    "predicted_progression": 0,
    "predicted_ctDNA_30d_vaf": 0.3366,
    "normalized_ctdna_risk": 0.0673,
    "status": "SUCCESS"
  },
  "multimodal_fusion": {
    "fusion_method": "weighted_linear",
    "weights_used": {
      "pathology_malignant": 0.35,
      "temporal_progression": 0.40,
      "ctdna_vaf_risk": 0.25
    },
    "prototype_multimodal_risk_score": 0.1345,
    "prototype_alert_level": "LOW (Prototype)",
    "normalized_ctdna_risk": 0.0673,
    "score_components": {
      "pathology_contribution": 0.1170,
      "progression_contribution": 0.0006,
      "ctdna_contribution": 0.0168
    }
  },
  "engineering_explanation": "Multimodal score (0.1345) computed via prototype weighted sum: Pathology P_malignant=0.3344 (weight 0.35), Temporal P_progression=0.0015 (weight 0.40), ctDNA Risk Index=0.0673 (weight 0.25, raw forecasted VAF 0.34%). Mapped to prototype alert level: LOW (Prototype).",
  "provenance": {
    "timestamp": "2026-09-05T19:55:00Z",
    "data_source": "synthetic",
    "clinical_validation_status": "NOT_CLINICALLY_VALIDATED",
    "pathology_model_checkpoint": "stage-2-dl/dl/checkpoints/best_pathology_cnn.pt",
    "temporal_model_checkpoint": "stage-2-dl/dl/checkpoints/best_temporal_lstm.pt",
    "mandatory_disclaimer": "This model was developed using synthetic data and has not been clinically validated. Performance on this dataset does not establish clinical effectiveness, safety, or real-world patient benefit.",
    "equivalence_disclaimer": "Synthetic data != Real patient evidence != Clinical validation"
  }
}
```

---

## 11. REST API & Inference Service

A high-performance FastAPI service is implemented in [`src/api.py`](file:///c:/Users/nallu/Desktop/Personalized_prediction/stage-2-dl/integration/src/api.py):
- `GET /health`: Returns service health, model paths, and regulatory disclaimers.
- `GET /info`: Exposes active fusion weights, alert thresholds, and schema definitions.
- `POST /predict/patient`: Accepts patient ID, tile file paths, and biomarker visit observations; returns standardized multimodal inference result.
- `POST /predict/sample/{patient_id}`: Demonstration endpoint loading sample data directly from the development cohort.
- `Error Handling`: Traps anti-leakage boundary violations and invalid inputs with HTTP 400.

---

## 12. Research Prototype Dashboard

An interactive Streamlit application is implemented in [`dashboard/app.py`](file:///c:/Users/nallu/Desktop/Personalized_prediction/stage-2-dl/integration/dashboard/app.py):
1. **Mandatory Red Top Banner:** Prominently displays `SYNTHETIC RESEARCH PROTOTYPE — NOT CLINICALLY VALIDATED`.
2. **Interactive Controls:** Allows selecting patient cases (`PAT-0001` through `PAT-0020`), toggling tile aggregation methods (`mean`, `median`, `max`), adjusting fusion weights via sliders, and simulating partial-modality outages.
3. **Visual Modality Inspection:** Displays Whole-Slide tile gallery with individual tile predictions and confidence scores.
4. **Temporal Biomarker Trajectory:** Renders longitudinal ctDNA VAF curve up to Day 90 with forward 30-day forecasted point.
5. **Multimodal Gauge & Decomposition:** Renders score breakdown bar chart and engineering narrative.
6. **Standalone HTML Generator:** [`dashboard/standalone_viewer.py`](file:///c:/Users/nallu/Desktop/Personalized_prediction/stage-2-dl/integration/dashboard/standalone_viewer.py) provides offline HTML report generation without a web server.

---

## 13. Explainability & Provenance Architecture

For every integrated prediction, the pipeline generates an **engineering contribution explanation**:
- Documents exact mathematical contributions ($w_i \cdot x_i$) to the total risk score.
- Strictly avoids causal clinical claims (e.g. stating *"The temporal model assigned a progression probability of 0.84"* rather than *"The patient has disease progression because biomarker X increased"*).
- Attaches full provenance: UTC timestamp, checkpoint file paths, data source ("synthetic"), validation status ("NOT_CLINICALLY_VALIDATED"), and regulatory disclaimers.

---

## 14. Validation & Automated Test Suite

A comprehensive test suite of 12 unit tests is implemented in [`tests/test_integration.py`](file:///c:/Users/nallu/Desktop/Personalized_prediction/stage-2-dl/integration/tests/test_integration.py):
1. `test_frozen_checkpoints_loading`: Validates that frozen `.pt` models load cleanly without errors.
2. `test_patient_level_alignment`: Verifies that tiles and time-series map to identical patient ID.
3. `test_tile_aggregation_methods`: Tests mean, median, and max tile aggregation accuracy.
4. `test_temporal_historical_window_boundary`: Asserts `ValidationError` is raised if observations exceed Day 90.
5. `test_future_target_exclusion`: Asserts `ValidationError` is raised if future target columns are supplied.
6. `test_missing_modality_pathology_only`: Validates single-modality pathology behavior.
7. `test_missing_modality_temporal_only`: Validates single-modality temporal behavior.
8. `test_missing_modality_neither`: Validates `INSUFFICIENT_DATA` response when no modalities are present.
9. `test_fusion_score_calculation`: Verifies exact mathematical computation of the weighted linear formula.
10. `test_standard_output_schema`: Asserts that all required keys, types, and value ranges exist.
11. `test_provenance_and_disclaimer_fields`: Checks presence of regulatory synthetic disclaimers.
12. `test_frozen_components_immutability`: Confirms that prior stage modules remain 100% unmodified.

---

## 15. Performance & Engineering Observations

- **Singleton Model Caching:** Deep learning backbones (ResNet-18 and BiLSTM) are loaded once upon pipeline instantiation, enabling fast subsequent inferences (<250 ms per patient on standard CPU).
- **Batching Tile Inference:** Biopsy tiles are processed sequentially or batched without memory leaks.
- **Dynamic Sequence Handling:** Variable-length historical sequences (6 to 8 visits) are processed with unpadded last-hidden-state indexing, ensuring padding tokens never contaminate sequence representations.

---

## 16. Known Limitations

1. **Synthetic Data Substrate:** The entire pipeline was developed and verified on synthetic benchmark data. Class separability is mathematically orthogonal and does not reflect real-world clinical noise.
2. **Heuristic Fusion Weights:** Weights ($0.35, 0.40, 0.25$) are baseline engineering choices and have not undergone clinical calibration.
3. **Single Time-Horizon Forecast:** The temporal model forecasts specifically at $t + 30$ days from Day 90 landmark; it does not model arbitrary continuous-time trajectories.
4. **No Autonomous Clinical Prescribing:** This prototype does not and cannot recommend treatments, drugs, or interventions.

---

## 17. Downstream Handoff Toward Stage 6 Agentic AI

The integrated multimodal pipeline is packaged and ready for downstream **Stage 6 Agentic Reasoning**:
- **Interface Entry Point:**
  ```python
  from stage_2_dl.integration.src.integration_pipeline import run_patient_inference
  result = run_patient_inference(patient_id="PAT-0001", pathology_tiles=tiles, temporal_history=bio_df)
  ```
- **Agentic Evidence Consumption:**
  Stage 6 agents can consume:
  - `pathology_summary.malignant_probability` (tissue-level morphology evidence)
  - `temporal_summary.progression_probability` (longitudinal disease dynamics evidence)
  - `temporal_summary.predicted_ctDNA_30d_vaf` (quantitative molecular tumor burden forecast)
  - `multimodal_fusion.prototype_multimodal_risk_score` (integrated synthesis)
- **Constraint for Stage 6 Agents:**
  Agents must cite the evidence fields objectively and must NEVER interpret prototype alert levels as clinically approved treatment directives.
