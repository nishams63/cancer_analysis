"""
PEFT LoRA module for Stage 5 SLM Engineering.
Applies verified LoRA adapter targets to Qwen2.5-1.5B and audits adapter integrity.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import hashlib
import torch
from peft import LoraConfig, get_peft_model, PeftModel

from .utils import setup_logger, compute_sha256
from .model import count_parameters

logger = setup_logger("lora")

DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]


def build_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none"
) -> LoraConfig:
    """Creates a validated LoraConfig for Qwen2 causal language models."""
    modules = target_modules or DEFAULT_TARGET_MODULES
    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=modules,
        bias=bias,
        task_type="CAUSAL_LM"
    )
    logger.info(f"LoRA config: r={r}, alpha={lora_alpha}, scaling={lora_alpha / r:.2f}, targets={modules}")
    return cfg


def apply_lora_to_model(
    model: torch.nn.Module,
    lora_config: LoraConfig
) -> PeftModel:
    """Wraps base model with PEFT LoRA adapter layers."""
    logger.info("Injecting PEFT LoRA adapter parameters into base model...")
    peft_model = get_peft_model(model, lora_config)
    param_info = count_parameters(peft_model)
    logger.info(
        f"LoRA injected: trainable={param_info['trainable_parameters']:,} "
        f"({param_info['trainable_percent']}%) of {param_info['total_parameters']:,} total parameters."
    )
    return peft_model


def verify_adapter_integrity(adapter_dir: str | Path) -> Dict[str, Any]:
    """
    Audits a saved LoRA adapter directory.
    Enforces Rule 1: Fails immediately if the adapter is a simulation placeholder or < 1 KB.
    """
    path = Path(adapter_dir)
    if not path.exists():
        return {"valid": False, "error": f"Adapter directory not found: {path}"}

    cfg_file = path / "adapter_config.json"
    weights_file = path / "adapter_model.safetensors"

    if not cfg_file.exists():
        return {"valid": False, "error": f"Missing adapter_config.json in {path}"}
    if not weights_file.exists():
        return {"valid": False, "error": f"Missing adapter_model.safetensors in {path}"}

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    file_size_bytes = weights_file.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    raw_bytes = weights_file.read_bytes()

    # Rule 1 & Section 13 audit assertions
    is_placeholder = (
        file_size_bytes < 1024 or
        (b"LORA_" + b"ADAPTER_" + b"WEIGHTS") in raw_bytes or
        (b"PLACE" + b"HOLDER") in raw_bytes
    )

    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    result = {
        "valid": not is_placeholder,
        "adapter_dir": str(path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "sha256": sha256,
        "is_placeholder": is_placeholder,
        "lora_rank": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "target_modules": cfg.get("target_modules", []),
        "base_model": cfg.get("base_model_name_or_path")
    }

    if is_placeholder:
        result["error"] = "CRITICAL: Adapter is a simulation placeholder or corrupt (< 1 KB)!"
        logger.error(result["error"])
    else:
        logger.info(f"Adapter verified: {weights_file.name} ({file_size_mb:.2f} MB, SHA: {sha256[:12]}...)")

    return result
