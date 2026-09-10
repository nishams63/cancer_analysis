"""
Integration Configuration Loader Module.
Loads runtime.yaml and reads the frozen calibrated threshold from Stage 6.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str = "qwen2.5-1.5b-instruct-clinical-lora"
    path: str = "stage-4-slm/integration-engineer/runtime/models/merged-model-Q4_K_M.gguf"
    quantization: str = "Q4_K_M"
    architecture: str = "qwen2"
    context_length: int = 4096
    threads: int = 4
    batch_size: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 256


class ServiceConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    offline_mode: bool = True
    max_note_length_chars: int = 10000


class SafetyConfig(BaseModel):
    strict_mode: bool = True
    calibrated_threshold_source: str = "stage-4-slm/evaluation-engineer/reports/provenance_manifest.json"
    confidence_threshold: float = 0.500
    route_on_failure: str = "HUMAN_REVIEW"


class LoggingConfig(BaseModel):
    audit_log_path: str = "stage-4-slm/integration-engineer/artifacts/audit_log.jsonl"
    mask_phi: bool = True


class AppConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: str = "stage-4-slm/integration-engineer/runtime/config/runtime.yaml") -> AppConfig:
    """Loads configuration from YAML and resolves Stage 6 calibrated threshold."""
    path = Path(config_path)
    raw = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    config = AppConfig(**raw)

    # Resolve frozen calibrated threshold from Stage 6 artifact
    stage6_manifest_path = Path(config.safety.calibrated_threshold_source)
    if stage6_manifest_path.exists():
        try:
            with open(stage6_manifest_path, "r", encoding="utf-8") as f:
                s6_data = json.load(f)
                locked_threshold = s6_data.get("locked_confidence_threshold")
                if locked_threshold is not None:
                    config.safety.confidence_threshold = float(locked_threshold)
        except Exception:
            pass

    return config
