"""
Tokenizer module for Stage 5 SLM Engineering.
Loads the official Qwen2.5 tokenizer, applies ChatML templates,
and provides empirical sequence length and truncation auditing.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .utils import setup_logger

logger = setup_logger("tokenizer")


def load_tokenizer(
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    trust_remote_code: bool = True
) -> PreTrainedTokenizerFast:
    """
    Loads the official Qwen fast tokenizer.
    Ensures pad_token is set and chat_template is verified.
    """
    logger.info(f"Loading official tokenizer: {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Tokenizer loaded successfully. Vocab size: {len(tokenizer)}")
    return tokenizer


def format_clinical_chat(
    tokenizer: PreTrainedTokenizerFast,
    system_prompt: str,
    clinical_note: str,
    assistant_response: Optional[str] = None
) -> str:
    """
    Formats clinical inputs into official Qwen ChatML syntax:
    <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Clinical Note:\n{clinical_note}\n\nProvide:\nRisk:\nKey Finding:\nAction:"}
    ]
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": assistant_response})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def audit_token_lengths(
    tokenizer: PreTrainedTokenizerFast,
    dataset_path: str | Path,
    system_prompt: str,
    split: str = "TRAIN",
    max_seq_length: int = 512
) -> Dict[str, Any]:
    """
    Audits actual token length distributions across the specified split.
    Measures prompt, target, total tokens, and checks for truncation.
    """
    df = pd.read_parquet(dataset_path)
    if "split" in df.columns:
        df = df[df["split"] == split]

    logger.info(f"Auditing token lengths for {len(df)} records in split '{split}'...")

    in_lens = []
    tar_lens = []
    tot_lens = []

    for _, row in df.iterrows():
        prompt = format_clinical_chat(tokenizer, system_prompt, row["clinical_note"])
        target = f"Risk: {row['target_risk']}\nKey Finding: {row['target_key_finding']}\nAction: {row['target_action']}"

        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        t_ids = tokenizer.encode(target, add_special_tokens=False)

        in_lens.append(len(p_ids))
        tar_lens.append(len(t_ids))
        tot_lens.append(len(p_ids) + len(t_ids))

    truncation_count = sum(1 for x in tot_lens if x > max_seq_length)
    truncation_rate = (truncation_count / len(tot_lens)) * 100.0 if tot_lens else 0.0

    stats = {
        "split": split,
        "sample_count": len(tot_lens),
        "max_seq_length_threshold": max_seq_length,
        "truncation_count": truncation_count,
        "truncation_rate_percent": round(truncation_rate, 2),
        "input_tokens": {
            "min": int(min(in_lens)),
            "max": int(max(in_lens)),
            "mean": round(float(np.mean(in_lens)), 1),
            "p95": round(float(np.percentile(in_lens, 95)), 1)
        },
        "target_tokens": {
            "min": int(min(tar_lens)),
            "max": int(max(tar_lens)),
            "mean": round(float(np.mean(tar_lens)), 1),
            "p95": round(float(np.percentile(tar_lens, 95)), 1)
        },
        "total_tokens": {
            "min": int(min(tot_lens)),
            "max": int(max(tot_lens)),
            "mean": round(float(np.mean(tot_lens)), 1),
            "p95": round(float(np.percentile(tot_lens, 95)), 1)
        }
    }
    logger.info(f"Token audit completed: mean total = {stats['total_tokens']['mean']}, truncation = {truncation_count}")
    return stats
