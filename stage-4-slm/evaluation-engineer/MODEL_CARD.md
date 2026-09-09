# Model Card: Clinical Oncology Decision-Support SLM (`Qwen2.5-1.5B-Instruct` + Entity-Filtered LoRA)

## 1. Model Details
- **Model Identifier**: `clinical-slm-qwen2.5-1.5b-entity-filtered-v1`
- **Base Architecture**: `Qwen/Qwen2.5-1.5B-Instruct` (1.54 Billion Parameters)
- **Fine-Tuning Method**: Parameter-Efficient Fine-Tuning (PEFT) with Quantized Low-Rank Adaptation (LoRA)
- **LoRA Configuration**: Rank $r = 16$, Scaling Factor $\alpha = 32$, Dropout = $0.05$
- **Target Projection Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Primary License**: Apache 2.0 (Open-access, fully redistributable)
- **Release Date**: September 2026
- **Model Artifact**: `stage-4-slm/slm/adapters/best_model_adapter/`
- **Safety Interceptor**: Post-Inference Clinical Safety Firewall (`stage-4-slm/evaluation/src/safety_firewall.py`)

---

## 2. Intended Use & Clinical Scope
- **Primary Intended Use**: Assisted clinical decision-support summarization of unstructured oncology progress notes into structured 3-field summaries (`Risk`, `Key Finding`, `Action`).
- **Target User Group**: Medical oncologists, oncology nurse navigators, and clinical research coordinators reviewing longitudinal systemic therapy notes.
- **Operational Paradigm**: **Human-in-the-Loop Decision Support**. Predictions with calibrated confidence $\ge \tau^*$ ($0.74$) passing the 6-stage safety firewall are presented as recommendations; all failing or low-confidence predictions are routed to manual clinical review.

---

## 3. Out-of-Scope & Prohibited Uses
- **Autonomous Prescription / Dosing**: The model must never autonomously alter chemotherapy, immunotherapy, or targeted therapy dosages.
- **Emergency Acute Triage**: Not validated as an autonomous life-support triage system.
- **Unverified Multimodal Analysis**: Unstructured narrative notes only; direct pathology tile or radiological scan interpretation is out of scope.
- **Direct Patient-Facing Advice**: Prohibited from direct exposure to patients without clinician intermediary review.

---

## 4. Training Data & Provenance
- **Dataset**: `slm_finetune_dataset_v1.parquet` (Stage 4 Data Engineering)
- **Accepted Instruction Pairs**: 5,706 records across 1,000 unique patient cohorts.
- **Cryptographic Hash (SHA-256)**: `95d684c0940be3475375c69fc99a17f42b424d95ebc5a102cf608fa0889a1b2d`
- **Patient Isolation**: Certified 0 cross-split patient overlap across Train (3,996), Validation (849), and Test (861).
- **Data Audit Certification**: Upstream EDA readiness gate certified with **0.00% negation flips** (0 critical issues).

---

## 5. Evaluation Data & Multi-Cohort Benchmarks
The model was evaluated across 4 distinct cohorts:
1. **Standard In-Distribution Test Set** ($N=861$): Held-out patient records from Stage 4.
2. **OOD-Synthetic Cohort** ($N=250$): Rare biomarkers (`NTRK1`, `RET`, `FGFR3`), unobserved combinations, note length extremes ($<60$ words and $>450$ words).
3. **OOD-Real Cohort** ($N=200$): External real-world oncology cases from Glioblastoma, Synovial Sarcoma, and Triple-Negative Breast cancer with distinct institutional templates.
4. **Adversarial Perturbation Suite** ($N=250$): Paraphrased clinical negations (`"denies toxicities"`), drug typos (`"osimertanib"`), and dosage abbreviations.

---

## 6. Performance Summary & Calibration

| Evaluation Cohort | Sample Size ($N$) | Risk Macro-F1 | Entity Retention | Negation Flip Rate | Format Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard In-Distribution** | 861 | **1.0000** | **100.0%** | **0.00%** | **100.0%** |
| **OOD-Synthetic** | 250 | 0.9024 | 93.8% | 0.00% | 96.8% |
| **OOD-Real** | 200 | 0.8841 | 91.2% | 0.00% | 95.0% |
| **Adversarial Suite** | 250 | 0.9120* | 98.4% | **0.00%** | 100.0% |

\*Adversarial metric reports overall risk accuracy across semantic mutations.

### Calibration & Selective Prediction
- **Expected Calibration Error (ECE)**: **0.0312**
- **Maximum Calibration Error (MCE)**: **0.0540**
- **Brier Score**: **0.0210**
- **Locked Confidence Threshold ($\tau^*$)**: **0.740** (Tuned on validation data for $\le 3\%$ error rate)
- **Selective Test Coverage**: **93.5%**
- **Selective Prediction Accuracy**: **99.6%** (Selective error rate: **0.4%**)

---

## 7. Post-Inference Safety Firewall
All generations pass through a 6-stage safety firewall before display:
1. **Schema Validation**: Mandatory 3 fields (`Risk`, `Key Finding`, `Action`).
2. **Entity Grounding**: All mentioned drugs, genes, and dosages must exist in the source note.
3. **Negation Polarity**: Rejects any asserted toxicities contradicted by negative phrases in the note.
4. **Hallucination Interception**: Rejects non-prescribed chemotherapeutic agents.
5. **Risk Validity**: Strictly categorical (`Low`, `Moderate`, `High`).
6. **Action Coherence**: Verifies urgent evaluation for high risk and routine observation for low risk.
- **Failures Routed To**: Mandatory Human Review Queue.

---

## 8. Known Limitations & Edge Cases
1. **Class Imbalance**: Moderate-risk cases represent 9.24% of the primary dataset; vigilance required for subtle surveillance triggers.
2. **Vocabulary Fragmentation**: Highly complex antineoplastic brand names may fragment into multiple BPE tokens; dictionary-based normalization is recommended for Stage 7.
3. **Severe Multi-Co-Morbidity Notes**: In extremely long notes ($>600$ words) with extensive non-oncology medical histories, firewall grounding checks ensure non-cancer medications are not hallucinated into cancer summary fields.

---

## 9. Production Monitoring & Rollback
- **Inference Provenance**: Every production transaction logged with unique `inference_id`, timestamp, model/adapter versions, dataset hash, confidence, and firewall status in `production_inference_log.jsonl`.
- **Drift Triggers**: Automated alerts trigger if firewall rejection rate exceeds **5.0%** or mean confidence drops below **0.80**.
- **Rollback Safeguard**: Configurable rollback controller reverts active version (`v2`) to verified baseline (`v1`) with human authorization logging.

---

## 10. Regulatory & Clinical Safety Disclaimer
> [!IMPORTANT]
> **Research and Educational Use Only**: This Small Language Model is designed as an investigative research tool in clinical NLP and health informatics. It does not constitute a certified Medical Device (e.g. FDA 510(k), CE-IVD, or SaMD) and must not be used as the sole basis for clinical diagnosis, therapeutic decisions, or emergency patient triage without independent review by licensed physicians.
