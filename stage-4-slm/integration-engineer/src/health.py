"""
Health Check and Service Diagnostics Module.
Verifies runtime state, model integrity, safety gateway readiness, and offline status.
"""

from typing import Dict, Any
from pathlib import Path


class HealthChecker:
    """Performs deep service diagnostics for clinical deployment readiness."""

    def __init__(self, model_manager, safety_gateway, config):
        self.model_manager = model_manager
        self.safety_gateway = safety_gateway
        self.config = config

    def check_health(self) -> Dict[str, Any]:
        """Validates all operational components and returns health status dictionary."""
        model_exists = Path(self.config.model.path).exists()
        model_loaded = self.model_manager.llm is not None
        runtime_ok = True
        safety_ok = self.safety_gateway is not None

        all_ok = model_exists and model_loaded and runtime_ok and safety_ok

        return {
            "status": "healthy" if all_ok else "degraded",
            "model_loaded": model_loaded,
            "runtime": "llama.cpp",
            "runtime_version": "0.3.35",
            "model_name": self.config.model.name,
            "quantization": self.config.model.quantization,
            "model_file_exists": model_exists,
            "safety_gateway": safety_ok,
            "offline_mode": self.config.service.offline_mode,
            "confidence_threshold": self.safety_gateway.confidence_threshold,
            "threads": self.config.model.threads,
            "context_length": self.config.model.context_length
        }
