# Stage 5 — SLM Engineer: Real Qwen2.5-1.5B-Instruct LoRA Engineering

**Subsystem**: `stage-4-slm/slm-engineer/`  
**Base Architecture**: `Qwen/Qwen2.5-1.5B-Instruct` (1,562,179,072 parameters)  
**Tokenizer**: `Qwen2TokenizerFast` (151,665 vocabulary tokens)  
**LoRA Adapter**: PEFT LoRA ($r=16, \alpha=32$) on 7 linear projection targets across 28 layers (18,464,768 trainable parameters, 1.1820%)  
**Fine-Tuning Dataset**: `stage-4-slm/data-engineer/data/slm_finetune_dataset_v1.parquet` (5,706 records, SHA-256: `95d684c0940...`)  
**Reality Audit Status**: `NOT VERIFIED — REAL TRAINING REQUIRED` (Rule 1 Compliant; CUDA GPU Required)  

---

## 1. Overview & Rebuild Mandate

This directory contains the production-grade rebuild of **Stage 5 (SLM Engineer)**. Following the Stage 7 independent verification audit, all legacy simulation artifacts (including the previous 36-byte placeholder adapter) were archived in `legacy_simulation/` to eliminate simulation from the primary pipeline.

The rebuilt system provides:
1. **Real Model & Architecture**: Native Hugging Face `Qwen2ForCausalLM` loading and parameter verification.
2. **Real Fast Tokenizer**: Official Qwen ChatML template formatting with empirical sequence length auditing (0.00% truncation at length 512).
3. **Real PEFT LoRA**: Verified target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) injecting 18.46M trainable parameters.
4. **Instruction Masking**: PyTorch data collator masking prompt tokens with `-100` to focus gradients exclusively on assistant responses.
5. **Real Autoregressive Generation**: Native `model.generate()` inference engine. No regex answer synthesis.
6. **Strict Rule 1 Compliance**: Automatic hardware probing cleanly halts on CPU (`TRAINING STATUS: BLOCKED`) without fabricating weights.
7. **1-Click Google Colab Notebook**: Production notebook (`notebooks/train_qwen_colab.ipynb`) for training on free T4/A100 GPUs.

---

## 2. Directory Structure

```text
stage-4-slm/slm-engineer/
├── configs/
│   ├── model.yaml              # Base model configuration (Qwen2.5-1.5B-Instruct)
│   ├── lora.yaml               # PEFT LoRA configuration (r=16, alpha=32, 7 targets)
│   ├── training.yaml           # Training hyperparameters and scheduler
│   └── qwen.yaml               # Backward-compatible model config
│
├── data/
│   ├── manifests/
│   │   └── dataset_manifest.json # SHA-256 and token distribution profile
│   └── README.md
│
├── src/
│   ├── dataset.py              # ClinicalDataset and masking collator
│   ├── tokenizer.py            # Real Qwen tokenizer loader and length auditor
│   ├── model.py                # Base model loader and parameter counter
│   ├── lora.py                 # PEFT LoRA injector and adapter auditor
│   ├── prompts.py              # ChatML prompt builder and output contract parser
│   ├── train.py                # Genuine Hugging Face Trainer execution script
│   ├── evaluate.py             # Empirical clinical metrics calculator
│   ├── inference.py            # Autoregressive generation engine (model.generate)
│   ├── utils.py                # Hardware detection, SHA-256, and logging
│   └── __init__.py
│
├── notebooks/
│   └── train_qwen_colab.ipynb  # Self-contained Google Colab GPU training notebook
│
├── run_gpu_training.sh         # Shell script for Linux / Cloud GPU servers
│
├── adapters/
│   ├── real_qwen_lora/         # Destination for real ~35.2 MB trained adapter weights
│   └── best_model_adapter/     # Preserved legacy adapter for integration compatibility
│
├── reports/
│   ├── STAGE5_REALITY_AUDIT.md # Evidence-based answers to 11 reality questions
│   ├── TRAINING_REPORT.md      # Environment, dataset, and parameter report
│   └── MODEL_CARD.md           # Model specifications, safety, and limitations
│
├── tests/                      # 18 comprehensive unit tests (all passing)
│   ├── conftest.py
│   ├── test_real_tokenizer.py
│   ├── test_real_lora_config.py
│   ├── test_dataset_and_leakage.py
│   ├── test_output_schema_and_prompt.py
│   ├── test_clinical_metrics.py
│   ├── test_no_placeholder_weights.py
│   └── test_no_simulation_primary_path.py
│
├── legacy_simulation/          # Archived historical simulation artifacts
└── README.md
```

---

## 3. How to Execute Training

### Preferred (Google Colab / Cloud GPU):
1. Open `notebooks/train_qwen_colab.ipynb` in Google Colab.
2. Select Runtime $\to$ Change runtime type $\to$ T4 or A100 GPU.
3. Run all cells to execute 3 epochs of genuine LoRA training and download `adapter_model.safetensors` (~35.2 MB).

### Local Environment Probe:
```bash
python -m stage-4-slm.slm-engineer.src.train --check-env
```
In a CPU-only environment, this cleanly outputs:
```text
TRAINING STATUS: BLOCKED (CUDA GPU REQUIRED)
```

---

## 4. Test Suite Execution

To run all 18 Stage 5 unit tests:
```bash
python -m pytest stage-4-slm/slm-engineer/tests -v
```
All 18 tests pass with 100% genuine assertion logic (no `assert True` stubs).
