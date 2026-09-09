"""
Tests verifying PEFT LoRA target modules and exact parameter scaling on Qwen2.5-1.5B.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from src.lora import build_lora_config, apply_lora_to_model
from src.model import count_parameters


def test_real_lora_parameters():
    """
    Verifies that LoRA config with r=16 on all 7 projection modules
    yields exactly 18,464,768 trainable parameters (1.1820%) on Qwen2.5-1.5B.
    """
    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    assert cfg.model_type == "qwen2"
    assert cfg.num_hidden_layers == 28
    assert cfg.hidden_size == 1536

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)

    lora_cfg = build_lora_config(r=16, lora_alpha=32, lora_dropout=0.05)
    assert lora_cfg.r == 16
    assert lora_cfg.lora_alpha == 32
    assert "q_proj" in lora_cfg.target_modules
    assert "down_proj" in lora_cfg.target_modules

    peft_model = apply_lora_to_model(model, lora_cfg)
    param_info = count_parameters(peft_model)

    assert param_info["total_parameters"] == 1562179072
    assert param_info["trainable_parameters"] == 18464768
    assert abs(param_info["trainable_percent"] - 1.1820) < 0.001
