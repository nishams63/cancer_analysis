"""
Official Training Orchestration Script for Stage 5 SLM Engineering.
Trains Qwen2.5-1.5B-Instruct with PEFT LoRA on CUDA GPU.
Enforces Rule 1: Automatically halts with 'TRAINING STATUS: BLOCKED'
if executed in a CPU-only environment without attempting fake simulation.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
from transformers import TrainingArguments, Trainer

from .utils import setup_logger, detect_hardware, load_yaml_config, compute_sha256
from .tokenizer import load_tokenizer
from .dataset import ClinicalDataset, DataCollatorForCausalLMWithMasking
from .model import load_base_model, count_parameters
from .lora import build_lora_config, apply_lora_to_model, verify_adapter_integrity

logger = setup_logger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-1.5B Clinical LoRA Fine-Tuning")
    parser.add_argument("--config", type=str, default="stage-4-slm/slm-engineer/configs/training.yaml", help="Path to training YAML config")
    parser.add_argument("--lora-config", type=str, default="stage-4-slm/slm-engineer/configs/lora.yaml", help="Path to LoRA YAML config")
    parser.add_argument("--dataset", type=str, default="stage-4-slm/data-engineer/data/slm_finetune_dataset_v1.parquet", help="Path to fine-tune parquet dataset")
    parser.add_argument("--output-adapter-dir", type=str, default="stage-4-slm/slm-engineer/adapters/real_qwen_lora", help="Destination directory for trained adapter")
    parser.add_argument("--check-env", action="store_true", help="Probes hardware environment and exits")
    return parser.parse_args()


def run_training():
    args = parse_args()
    logger.info("==================================================================")
    logger.info("STAGE 5: REAL QWEN2.5-1.5B-INSTRUCT LoRA FINE-TUNING")
    logger.info("==================================================================")

    # 1. Hardware Detection & Rule 1 Gate
    hw = detect_hardware()
    logger.info(f"Hardware Environment: OS={hw['os']}, Python={hw['python_version']}, PyTorch={hw['pytorch_version']}")
    logger.info(f"CUDA Available: {hw['cuda_available']} (Device: {hw['device']})")

    if args.check_env:
        print(json.dumps(hw, indent=2))
        return

    if not hw["cuda_available"]:
        logger.error("******************************************************************")
        logger.error("TRAINING STATUS: BLOCKED (CUDA GPU REQUIRED)")
        logger.error("Rule 1 Compliance: Fake CPU simulation or random weights strictly forbidden.")
        logger.error("Reason: Fine-tuning a 1.56B parameter model on CPU is impractical.")
        logger.error("To execute real training, run in Google Colab or on a CUDA GPU:")
        logger.error("  -> See notebooks/train_qwen_colab.ipynb or run on a CUDA server.")
        logger.error("******************************************************************")
        sys.exit(2)

    # 2. Load Configurations
    train_cfg = load_yaml_config(args.config)
    lora_cfg_dict = load_yaml_config(args.lora_config)
    dataset_path = Path(args.dataset)

    dataset_sha = compute_sha256(dataset_path)
    logger.info(f"Verified Stage 4 Dataset: {dataset_path} (SHA: {dataset_sha[:16]}...)")

    # 3. Load Tokenizer & Model
    tokenizer = load_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
    base_model = load_base_model(
        model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )

    if train_cfg.get("gradient_checkpointing", True):
        base_model.gradient_checkpointing_enable()

    # 4. Inject PEFT LoRA
    lora_config = build_lora_config(
        r=lora_cfg_dict.get("r", 16),
        lora_alpha=lora_cfg_dict.get("lora_alpha", 32),
        lora_dropout=lora_cfg_dict.get("lora_dropout", 0.05),
        target_modules=lora_cfg_dict.get("target_modules")
    )
    model = apply_lora_to_model(base_model, lora_config)

    # 5. Prepare Datasets & Masking Collator
    train_dataset = ClinicalDataset(dataset_path, split="TRAIN")
    val_dataset = ClinicalDataset(dataset_path, split="VALIDATION")
    data_collator = DataCollatorForCausalLMWithMasking(tokenizer, max_length=train_cfg.get("max_seq_length", 512))

    # 6. Configure Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        logging_dir=train_cfg.get("logging_dir", "training/logs"),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        fp16=(not torch.cuda.is_bf16_supported()),
        bf16=torch.cuda.is_bf16_supported(),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 50),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 50),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_cfg.get("greater_is_better", False),
        logging_steps=train_cfg.get("logging_steps", 10),
        report_to="none",
        seed=train_cfg.get("seed", 42)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # 7. Execute Genuine Training
    logger.info("Starting genuine LoRA fine-tuning...")
    train_result = trainer.train()

    # 8. Save Real LoRA Adapter
    adapter_out = Path(args.output_adapter_dir)
    adapter_out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving real LoRA adapter to {adapter_out}...")
    model.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))

    # 9. Verify Real Adapter Integrity
    audit_res = verify_adapter_integrity(adapter_out)
    if not audit_res["valid"]:
        raise RuntimeError(f"Saved adapter failed integrity audit: {audit_res.get('error')}")

    logger.info("==================================================================")
    logger.info("REAL LoRA FINE-TUNING COMPLETED SUCCESSFULLY")
    logger.info(f"Adapter: {adapter_out} (Size: {audit_res['file_size_mb']} MB, SHA: {audit_res['sha256'][:16]}...)")
    logger.info("==================================================================")


if __name__ == "__main__":
    run_training()
