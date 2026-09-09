# Real Qwen2.5-1.5B LoRA Adapter Directory

This directory is the designated destination for the **genuine trained LoRA adapter weights** (`adapter_model.safetensors`, `adapter_config.json`, and tokenizer files) produced by real GPU training.

## Adapter Specifications (Rule 1 Compliant)
- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (28 transformer layers $\times$ 7 projections = 196 targets)
- **LoRA Rank ($r$)**: `16`
- **LoRA Alpha ($\alpha$)**: `32`
- **Trainable Parameters**: `18,464,768` (1.1820% of 1,562,179,072 base parameters)
- **Expected Weight File Size**: `~35.2 MB` in FP16 / BF16

## Local Training Status
- **Current Environment**: Windows 11 AMD64 (CPU Only, CUDA `False`)
- **Status**: `TRAINING STATUS: BLOCKED (CUDA GPU REQUIRED)`
- **Rule 1 Compliance**: No placeholder weights, fake `.safetensors`, or simulated random tensors have been created.

## How to Produce Real Adapter Weights
Execute the complete fine-tuning pipeline on a CUDA GPU (e.g. Google Colab T4 / A100):
```bash
# 1. Open notebooks/train_qwen_colab.ipynb in Google Colab
# 2. Or run on any Linux/Windows server equipped with NVIDIA CUDA:
python -m stage-4-slm.slm-engineer.src.train --config stage-4-slm/slm-engineer/configs/training.yaml
```
Once real training finishes, the validated adapter files will be deposited here and audited with `verify_adapter_integrity()`.
