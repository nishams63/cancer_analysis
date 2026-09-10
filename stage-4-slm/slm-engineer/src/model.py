"""
Model management module for Stage 5 SLM Engineering.
Handles Qwen2.5-1.5B-Instruct architecture loading, device placement,
and exact parameter accounting.
"""

from typing import Dict, Any, Optional, Tuple
import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from .utils import setup_logger

logger = setup_logger("model")


def load_model_config(
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    trust_remote_code: bool = True
) -> AutoConfig:
    """Loads and inspects the base model configuration."""
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    logger.info(
        f"Base config: {cfg.model_type}, layers={cfg.num_hidden_layers}, "
        f"hidden_size={cfg.hidden_size}, ffn_dim={cfg.intermediate_size}, "
        f"num_heads={cfg.num_attention_heads}, num_kv_heads={cfg.num_key_value_heads}"
    )
    return cfg


def load_base_model(
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    low_cpu_mem_usage: bool = True
) -> PreTrainedModel:
    """
    Loads the real Qwen2.5-1.5B-Instruct model from Hugging Face or local cache.
    Selects appropriate device (CUDA or CPU).
    """
    logger.info(f"Loading base causal LM: {model_name_or_path} (dtype={torch_dtype}, device_map={device_map})...")

    # If CUDA is unavailable, ensure device_map is cpu
    if not torch.cuda.is_available() and device_map == "auto":
        device_map = "cpu"
        # On CPU, default to float32 if bfloat16 causes issues
        if torch_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage
    )

    logger.info("Base model loaded successfully.")
    return model


def count_parameters(model: torch.nn.Module) -> Dict[str, Any]:
    """Computes exact parameter counts and trainable fraction."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percent = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "trainable_percent": round(trainable_percent, 4)
    }
