"""
PEFT LoRA and QLoRA Configuration Factory for Stage 5 SLM.
Configures adapter parameters (rank, alpha, dropout, target modules)
and hardware-aware quantization settings per Sections 11 & 12.
"""

from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path


class QLoRAConfigFactory:
    """Creates PEFT LoRA and BitsAndBytes quantization configurations."""

    DEFAULT_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]

    @staticmethod
    def get_lora_config(
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        task_type: str = "CAUSAL_LM"
    ) -> Dict[str, Any]:
        """Returns standard PEFT LoRA configuration dictionary."""
        return {
            "r": int(r),
            "lora_alpha": int(alpha),
            "lora_dropout": float(dropout),
            "target_modules": target_modules or QLoRAConfigFactory.DEFAULT_TARGET_MODULES,
            "bias": "none",
            "task_type": task_type
        }

    @staticmethod
    def get_peft_lora_object(r: int = 16, alpha: int = 32, dropout: float = 0.05, target_modules: Optional[List[str]] = None):
        """Builds a Peft LoraConfig object if peft is installed."""
        try:
            from peft import LoraConfig, TaskType
            return LoraConfig(
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=target_modules or QLoRAConfigFactory.DEFAULT_TARGET_MODULES,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
        except ImportError:
            return QLoRAConfigFactory.get_lora_config(r, alpha, dropout, target_modules)

    @staticmethod
    def get_quantization_config(is_cuda_available: bool = False) -> Optional[Any]:
        """
        Builds BitsAndBytesConfig for 4-bit QLoRA if CUDA is available.
        Returns None on CPU (standard float/bfloat execution).
        """
        if not is_cuda_available:
            return None
        try:
            import torch
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        except (ImportError, Exception):
            return None
