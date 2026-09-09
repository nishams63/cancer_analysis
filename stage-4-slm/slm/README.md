# Stage 4 — SLM Engineer: Small Language Model Fine-Tuning

## Role Overview
The **SLM Engineer** in Stage 4 is responsible for fine-tuning compact, parameter-efficient Small Language Models (SLMs) on the validated clinical instruction-tuning dataset (`slm_finetune_dataset_v1.parquet`).

## Objectives & Deliverables
1. **Model Architecture Selection**:
   - Parameter-efficient base SLMs (e.g. BioGPT, Clinical-Llama-3-8B, Gemma-2-2B/9B, Mistral-7B, Qwen2.5-7B).
2. **Fine-Tuning Strategy**:
   - Quantized Low-Rank Adaptation (QLoRA, 4-bit/8-bit NF4) or full-rank LoRA.
   - Target projection modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
3. **Training & Hyperparameters**:
   - Cosine learning rate schedule with warmup, gradient checkpointing, and gradient accumulation.
   - Supervised fine-tuning loss on prompt-completion tokens strictly masking instruction inputs.
4. **Artifact Checkpointing**:
   - LoRA adapter weights, tokenizer configs, and merged inference weights.
