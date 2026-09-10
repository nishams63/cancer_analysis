#!/usr/bin/env bash
# ==============================================================================
# Script: run_gpu_training.sh
# Purpose: Executes genuine PEFT LoRA fine-tuning of Qwen2.5-1.5B-Instruct on CUDA GPU
# ==============================================================================

set -euo pipefail

echo "=================================================================="
echo "Starting Stage 5 Real Qwen2.5-1.5B-Instruct LoRA Fine-Tuning"
echo "=================================================================="

# Check NVIDIA GPU presence
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Real training requires a CUDA GPU."
    exit 1
fi

nvidia-smi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python -m stage-4-slm.slm-engineer.src.train \
    --config stage-4-slm/slm-engineer/configs/training.yaml \
    --lora-config stage-4-slm/slm-engineer/configs/lora.yaml \
    --dataset stage-4-slm/data-engineer/data/slm_finetune_dataset_v1.parquet \
    --output-adapter-dir stage-4-slm/slm-engineer/adapters/real_qwen_lora

echo "Fine-tuning completed successfully."
