"""
Clinical Instruction Dataset Module for Stage 4 SLM Fine-Tuning.
Formats clinical oncology notes and targets into prompt-completion pairs,
strictly masking prompt tokens (-100 label masking) to focus loss on target generation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("stage4.slm.dataset")

DEFAULT_SYSTEM_PROMPT = (
    "You are an oncology clinical decision support assistant. "
    "Analyze the clinical note and generate structured Risk, Key Finding, and Action recommendations."
)


def format_instruction_prompt(instruction: str, clinical_note: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Standardizes prompt formatting for causal language model fine-tuning."""
    return (
        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"Instruction: {instruction}\n\n"
        f"Clinical Note:\n{clinical_note}\n\n"
        f"Provide structured Risk, Key Finding, and Action: [/INST]"
    )


def format_completion(target_risk: str, target_key_finding: str, target_action: str) -> str:
    """Formats the target completion triad."""
    return (
        f" Risk: {target_risk}\n"
        f"Key Finding: {target_key_finding}\n"
        f"Action: {target_action}</s>"
    )


class ClinicalInstructionDataset(Dataset):
    """PyTorch Dataset for Clinical SLM Instruction Tuning."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any = None,
        max_length: int = 1024,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        split: Optional[str] = None
    ):
        if split is not None:
            self.df = df[df["split"] == split].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        logger.info(f"Initialized ClinicalInstructionDataset with {len(self.df)} records (split={split}).")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        note = str(row.get("clinical_note", ""))
        inst = str(row.get("instruction", "Analyze the clinical note and provide the patient's Risk, Key Finding, and Action."))
        t_risk = str(row.get("target_risk", ""))
        t_kf = str(row.get("target_key_finding", ""))
        t_act = str(row.get("target_action", ""))

        prompt_str = format_instruction_prompt(inst, note, self.system_prompt)
        completion_str = format_completion(t_risk, t_kf, t_act)
        full_text = prompt_str + completion_str

        sample = {
            "patient_id": str(row.get("patient_id", "")),
            "note_id": str(row.get("note_id", "")),
            "clinical_note": note,
            "instruction": inst,
            "target_risk": t_risk,
            "target_key_finding": t_kf,
            "target_action": t_act,
            "prompt_text": prompt_str,
            "completion_text": completion_str,
            "full_text": full_text
        }

        if self.tokenizer is not None:
            # Tokenize prompt and full text
            prompt_enc = self.tokenizer(
                prompt_str,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False
            )
            full_enc = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )

            input_ids = full_enc["input_ids"].squeeze(0)
            attention_mask = full_enc["attention_mask"].squeeze(0)

            # Mask prompt tokens with -100 so cross-entropy loss is computed ONLY on completion
            labels = input_ids.clone()
            prompt_len = len(prompt_enc["input_ids"])
            labels[:prompt_len] = -100
            # Also mask padding tokens
            labels[attention_mask == 0] = -100

            sample["input_ids"] = input_ids
            sample["attention_mask"] = attention_mask
            sample["labels"] = labels
            sample["prompt_length"] = prompt_len

        return sample
