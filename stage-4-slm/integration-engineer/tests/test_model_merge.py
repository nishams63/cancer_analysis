"""
Unit Tests for Section 3: Model Merging and Deterministic LoRA Export.
Verifies adapter integrity, architecture compatibility, parameter scaling, and artifact output.
"""

import json
from pathlib import Path
import pytest
from merge_lora import LoRAMerger


def test_lora_merger_input_verification():
    merger = LoRAMerger()
    info = merger.verify_inputs()
    assert "adapter_config" in info
    assert "adapter_weights_sha256" in info
    assert info["adapter_config"]["base_model_name_or_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert info["adapter_config"]["r"] == 16
    assert info["adapter_config"]["lora_alpha"] == 32
    assert len(info["adapter_weights_sha256"]) == 64


def test_lora_merger_execution(tmp_path):
    out_dir = tmp_path / "test_merged"
    merger = LoRAMerger(output_dir=str(out_dir))
    manifest = merger.execute_merge()

    assert manifest["merge_status"] == "SUCCESS"
    assert (out_dir / "config.json").exists()
    assert (out_dir / "generation_config.json").exists()
    assert (out_dir / "tokenizer_config.json").exists()
    assert (out_dir / "merge_manifest.json").exists()

    with open(out_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["model_type"] == "qwen2"
    assert cfg["hidden_size"] == 1536
    assert cfg["num_hidden_layers"] == 28
    assert cfg["merged_adapter"]["scaling_factor"] == 2.0
    assert len(cfg["merged_adapter"]["target_modules"]) == 7


def test_scaling_factor_exact_value():
    merger = LoRAMerger()
    info = merger.verify_inputs()
    cfg = info["adapter_config"]
    scale = cfg["lora_alpha"] / cfg["r"]
    assert scale == 2.0
