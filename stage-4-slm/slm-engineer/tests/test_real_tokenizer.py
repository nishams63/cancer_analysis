"""
Tests verifying the official Qwen2.5-1.5B-Instruct tokenizer and ChatML formatting.
"""

from pathlib import Path
from src.tokenizer import load_tokenizer, format_clinical_chat, audit_token_lengths
from src.prompts import DEFAULT_SYSTEM_PROMPT


def test_real_tokenizer_loading():
    """Verifies that the official Qwen tokenizer loads with expected 151k+ vocabulary."""
    tok = load_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
    assert tok is not None
    assert len(tok) == 151665
    assert tok.pad_token is not None
    assert tok.eos_token is not None


def test_chatml_formatting():
    """Verifies that clinical notes are formatted with official Qwen ChatML syntax."""
    tok = load_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
    formatted = format_clinical_chat(
        tok,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        clinical_note="Patient with EGFR mutation on Osimertinib 80mg daily."
    )
    assert "<|im_start|>system" in formatted
    assert "<|im_start|>user" in formatted
    assert "<|im_start|>assistant" in formatted
    assert "Osimertinib 80mg daily" in formatted


def test_token_length_audit():
    """Verifies empirical token distribution measurement across Stage 4 dataset."""
    tok = load_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
    dataset_path = Path("stage-4-slm/data-engineer/data/slm_finetune_dataset_v1.parquet")
    assert dataset_path.exists()

    stats = audit_token_lengths(tok, dataset_path, DEFAULT_SYSTEM_PROMPT, split="VALIDATION", max_seq_length=512)
    assert stats["sample_count"] == 849
    assert stats["truncation_count"] == 0
    assert stats["total_tokens"]["max"] <= 512
    assert stats["total_tokens"]["mean"] > 100
