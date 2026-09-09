"""
Tests verifying dataset schema, split isolation, patient separation,
and instruction-masking data collation.
"""

from pathlib import Path
import pandas as pd
import torch
from src.utils import compute_sha256
from src.tokenizer import load_tokenizer
from src.dataset import ClinicalDataset, DataCollatorForCausalLMWithMasking


def test_dataset_integrity_and_hash():
    """Verifies dataset existence, 5,706 record count, and exact SHA-256."""
    dataset_path = Path("stage-4-slm/data-engineer/data/slm_finetune_dataset_v1.parquet")
    assert dataset_path.exists()
    sha = compute_sha256(dataset_path)
    assert sha == "95d684c0940be3475375c69fc99a17f42b424d95ebc5a102cf608fa0889a1b2d"

    df = pd.read_parquet(dataset_path)
    assert len(df) == 5706
    assert set(df["split"].unique()) == {"TRAIN", "VALIDATION", "TEST"}


def test_cross_split_patient_isolation():
    """Verifies 0 patient leakage between train, validation, and test splits."""
    dataset_path = Path("stage-4-slm/data-engineer/data/slm_finetune_dataset_v1.parquet")
    df = pd.read_parquet(dataset_path)

    train_pts = set(df[df["split"] == "TRAIN"]["patient_id"].unique())
    val_pts = set(df[df["split"] == "VALIDATION"]["patient_id"].unique())
    test_pts = set(df[df["split"] == "TEST"]["patient_id"].unique())

    assert len(train_pts.intersection(val_pts)) == 0
    assert len(train_pts.intersection(test_pts)) == 0
    assert len(val_pts.intersection(test_pts)) == 0
    assert len(train_pts) + len(val_pts) + len(test_pts) == 1000


def test_instruction_masking_collator():
    """Verifies that prompt tokens are masked with -100 while response tokens are preserved."""
    tok = load_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
    dataset_path = Path("stage-4-slm/data-engineer/data/slm_finetune_dataset_v1.parquet")
    ds = ClinicalDataset(dataset_path, split="TRAIN")

    collator = DataCollatorForCausalLMWithMasking(tok, max_length=512)
    sample_batch = [ds[0], ds[1]]
    batch = collator(sample_batch)

    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch

    labels = batch["labels"]
    # Check that prompt begins with -100 masking
    assert (labels[0][:10] == -100).all()
    # Check that assistant response tokens have valid token IDs (>= 0)
    assert (labels[0] >= 0).any()
