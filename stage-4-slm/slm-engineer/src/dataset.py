"""
PyTorch Dataset and DataCollator module for Stage 5 SLM Fine-Tuning.
Implements instruction-masking data collation so loss is calculated
strictly on assistant target tokens, preserving model generation capacity.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import PreTrainedTokenizerFast

from .prompts import build_clinical_prompt, DEFAULT_SYSTEM_PROMPT
from .utils import setup_logger

logger = setup_logger("dataset")


class ClinicalDataset(Dataset):
    """Encapsulates clinical note pairs from the frozen Stage 4 parquet dataset."""

    def __init__(
        self,
        parquet_path: str | Path,
        split: str = "TRAIN",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ):
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.parquet_path}")

        df = pd.read_parquet(self.parquet_path)
        if "split" in df.columns:
            self.df = df[df["split"] == split].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        self.split = split
        self.system_prompt = system_prompt
        logger.info(f"Loaded {len(self.df)} records for split '{split}' from {self.parquet_path.name}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        note = row["clinical_note"]
        target = f"Risk: {row['target_risk']}\nKey Finding: {row['target_key_finding']}\nAction: {row['target_action']}"

        return {
            "patient_id": row.get("patient_id", f"PT-{idx}"),
            "note_id": row.get("note_id", f"N-{idx}"),
            "clinical_note": note,
            "target": target,
            "target_risk": row.get("target_risk", "Low"),
            "system_prompt": self.system_prompt,
            "ner_genes": row.get("ner_genes", []),
            "ner_drugs": row.get("ner_drugs", []),
            "ner_dosages": row.get("ner_dosages", []),
            "ner_adverse_events": row.get("ner_adverse_events", [])
        }


class DataCollatorForCausalLMWithMasking:
    """
    Data collator that pads input sequences and applies label masking (-100)
    to the system and user prompt tokens so cross-entropy loss is computed
    only on the assistant's clinical decision response.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        pad_to_multiple_of: Optional[int] = 8
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for item in batch:
            # 1. Tokenize prompt only (to locate prompt end boundary)
            prompt_str = build_clinical_prompt(
                self.tokenizer,
                clinical_note=item["clinical_note"],
                system_prompt=item["system_prompt"]
            )
            prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)

            # 2. Tokenize full sequence with assistant response
            full_str = prompt_str + item["target"] + self.tokenizer.eos_token
            full_ids = self.tokenizer.encode(full_str, add_special_tokens=False)

            # Truncate if exceeding max length
            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]

            # 3. Create labels: mask prompt with -100, keep response tokens
            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]

            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            attention_mask_list.append(torch.ones(len(full_ids), dtype=torch.long))

        # Pad sequences to max length in current batch
        pad_id = self.tokenizer.pad_token_id or 151643
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=pad_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100
        )
        padded_attention = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0
        )

        return {
            "input_ids": padded_inputs,
            "attention_mask": padded_attention,
            "labels": padded_labels
        }
