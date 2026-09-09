# Model Card: Qwen2.5-1.5B-Instruct Clinical Decision-Support LoRA

**Model Name**: `Qwen2.5-1.5B-Instruct-Clinical-LoRA`  
**Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`  
**Task**: Clinical Oncology Progress Note Summarization & Risk Categorization  
**Version**: 2.0.0 (Stage 5 Real SLM Rebuild)  
**License**: Apache 2.0 (Open Access)  

---

## 1. Model Summary & Architecture

This model is a Parameter-Efficient Fine-Tuned (PEFT) adapter built on top of the Alibaba Cloud `Qwen/Qwen2.5-1.5B-Instruct` causal language model. It specializes in summarizing complex, unstructured clinical oncology progress notes into structured clinical decision briefings:
- **Categorical Risk Tier**: `Low`, `Moderate`, or `High`
- **Key Clinical Finding**: Synthesized summary of primary diagnosis, genomic alterations (e.g. EGFR, KRAS, BRAF), antineoplastic therapy (e.g. Osimertinib, Docetaxel), and observed adverse events.
- **Recommended Action**: Clinically aligned therapeutic guidance (e.g. continue standard therapy, hold medication, dose reduce, increase monitoring).

### Architectural Specifications:
- **Base Parameters**: 1,562,179,072
- **Hidden Layers**: 28 transformer decoder blocks
- **Hidden Dimension**: 1536
- **Attention Heads**: 12 query heads, 2 KV heads (GQA)
- **Vocabulary Size**: 151,665 tokens
- **Context Length**: 32,768 (operationalized at 1024 / 512 for summarization)

---

## 2. LoRA Fine-Tuning Specifications

- **PEFT Method**: Low-Rank Adaptation (LoRA)
- **Rank ($r$)**: 16
- **Alpha ($\alpha$)**: 32 (Scaling Factor: 2.000)
- **Dropout**: 0.05
- **Target Modules**: All 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Trainable Parameters**: 18,464,768 (1.1820% of total)
- **Adapter Weight Footprint**: ~35.2 MB (safetensors format)

---

## 3. Training Data & Split Methodology

- **Dataset**: `slm_finetune_dataset_v1.parquet` (Stage 4 Data Engineering)
- **Total Records**: 5,706 accepted oncology consultation notes
- **Unique Patients**: 1,000 synthetic patients
- **Splits**:
  - `TRAIN`: 3,996 records (700 patients)
  - `VALIDATION`: 849 records (150 patients)
  - `TEST`: 861 records (150 patients)
- **Patient Isolation**: Strict patient-level grouping. Zero patient overlap across splits.

---

## 4. Intended Use & Clinical Scope

### Intended Use:
- **Clinical Decision Support (CDS)**: Assisting oncology providers by drafting rapid structured briefings from lengthy electronic health record progress notes.
- **Provider Time Savings**: Accelerating chart review during multidisciplinary oncology tumor boards.

### Prohibited & Out-of-Scope Uses:
- **Autonomous Prescription / Dosing**: The model must never autonomously order medications, alter dosages, or discharge patients.
- **Direct-to-Consumer Diagnosis**: Unsupervised patient interaction is strictly prohibited.
- **Unreviewed Triage**: Outputs must not bypass licensed clinician review.

---

## 5. Safety Limitations & Mandatory Human Review

> [!WARNING]
> **MANDATORY CLINICIAN REVIEW REQUIREMENT**  
> This model is a research decision-support prototype. It is **NOT clinically validated** by the FDA or medical boards. All generated summaries and risk classifications must be reviewed and countersigned by a board-certified physician.

### Known Limitations:
1. **Hallucination Risk**: Small language models can occasionally hallucinate unprescribed antineoplastics or fabricate toxicities under rare genomic contexts.
2. **Negation Sensitivity**: Complex nested negations (e.g. *"not completely free of symptoms"*) require validation through the Stage 6 post-inference clinical safety firewall.
3. **Hardware Requirement for Training**: Full fine-tuning requires a CUDA-enabled GPU (T4/A100). Local CPU execution is limited to inference and validation.

---

## 6. Training & Reproducibility Command

To train on CUDA GPU:
```bash
python -m stage-4-slm.slm-engineer.src.train \
    --config stage-4-slm/slm-engineer/configs/training.yaml \
    --lora-config stage-4-slm/slm-engineer/configs/lora.yaml
```
Or execute the Google Colab notebook: [`notebooks/train_qwen_colab.ipynb`](file:///C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage-4-slm/slm-engineer/notebooks/train_qwen_colab.ipynb).
