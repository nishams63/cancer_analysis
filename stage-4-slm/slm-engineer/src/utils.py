"""
Utility functions for Stage 5 SLM Engineering.
Provides hardware environment detection, cryptographic hashing, and logging utilities.
"""

import os
import sys
import yaml
import logging
import hashlib
import platform
import psutil
from pathlib import Path
from typing import Dict, Any


def setup_logger(name: str = "slm_engineer", level: int = logging.INFO) -> logging.Logger:
    """Configures structured console logging."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def compute_sha256(filepath: str | Path) -> str:
    """Computes SHA-256 hash of a file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def detect_hardware() -> Dict[str, Any]:
    """
    Detects hardware capabilities (CUDA, MPS, CPU) and installed package versions.
    Enforces Rule 1: No fake GPU claims.
    """
    import torch
    import transformers
    import peft

    cuda_available = torch.cuda.is_available()
    hw_info = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu",
        "cpu_count": os.cpu_count() or 4,
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "disk_free_gb": round(psutil.disk_usage(".").free / (1024**3), 2),
    }

    if cuda_available:
        hw_info.update({
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "training_ready": True
        })
    else:
        hw_info.update({
            "gpu_count": 0,
            "gpu_name": "None (CPU Execution Only)",
            "vram_total_gb": 0.0,
            "training_ready": False,
            "blocked_reason": "CUDA GPU required for full 1.56B parameter fine-tuning. CPU training impractical."
        })

    return hw_info


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Loads YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
