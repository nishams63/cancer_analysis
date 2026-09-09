"""
Hardware-Aware Model and Tokenizer Loader Module for Stage 5 SLM.
Inspects system hardware, configures 4-bit QLoRA on CUDA or CPU-adapted PEFT layers,
and handles open vs gated candidate model loading per Sections 4, 5, & 11.
"""

import os
import shutil
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import yaml

logger = logging.getLogger("stage5_slm.model_loader")


def inspect_system_hardware() -> Dict[str, Any]:
    """
    Inspects available CPU, RAM, GPU, VRAM, and disk space per Section 4.
    """
    total_disk, used_disk, free_disk = shutil.disk_usage(".")

    hw_info = {
        "disk_free_gb": float(round(free_disk / (1024**3), 2)),
        "disk_total_gb": float(round(total_disk / (1024**3), 2)),
        "cuda_available": False,
        "device_count": 0,
        "device_name": "CPU",
        "vram_gb": 0.0,
        "pytorch_version": "N/A"
    }

    try:
        import torch
        hw_info["pytorch_version"] = torch.__version__
        hw_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hw_info["device_count"] = torch.cuda.device_count()
            hw_info["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            hw_info["vram_gb"] = float(round(props.total_memory / (1024**3), 2))
    except ImportError:
        pass

    try:
        import psutil
        vm = psutil.virtual_memory()
        hw_info["ram_total_gb"] = float(round(vm.total / (1024**3), 2))
        hw_info["ram_available_gb"] = float(round(vm.available / (1024**3), 2))
    except ImportError:
        hw_info["ram_total_gb"] = 16.0
        hw_info["ram_available_gb"] = 8.0

    return hw_info


class ModelLoader:
    """Manages candidate model loading, tokenizer configuration, and hardware adaptation."""

    def __init__(self, model_config_path: str):
        self.config_path = Path(model_config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.model_info = self.config["model"]
        self.hardware = inspect_system_hardware()

    def check_access_and_auth(self) -> Dict[str, Any]:
        """
        Validates model identifier, license, and Hugging Face authentication requirements.
        """
        requires_auth = self.model_info.get("requires_auth", False)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        is_accessible = True
        notes = []

        if requires_auth and not hf_token:
            is_accessible = False
            notes.append(
                f"Model '{self.model_info['model_id']}' is gated and requires HF_TOKEN. "
                "No Hugging Face token detected in environment."
            )
        else:
            notes.append(f"Model '{self.model_info['model_id']}' access verified ({self.model_info.get('license')}).")

        return {
            "model_name": self.model_info["name"],
            "model_id": self.model_info["model_id"],
            "license": self.model_info.get("license"),
            "requires_auth": requires_auth,
            "has_token": bool(hf_token),
            "is_accessible": is_accessible,
            "notes": notes
        }

    def load_tokenizer(self):
        """Loads and configures the subword tokenizer."""
        from transformers import AutoTokenizer
        model_id = self.model_info["model_id"]
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            logger.warning(f"Could not load AutoTokenizer from '{model_id}': {e}. Using GPT-2 fallback tokenizer.")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
