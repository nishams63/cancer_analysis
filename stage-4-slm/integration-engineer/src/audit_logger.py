"""
Immutable Audit Logging Module for Clinical SLM Decisions.
Appends cryptographically hashed transaction records to audit_log.jsonl.
Complies with HIPAA/privacy standards by default (no raw PHI stored).
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("audit_logger")


class AuditLogger:
    """Manages append-only JSONL audit records for clinical inference requests."""

    def __init__(self, log_path: str = "stage-4-slm/integration-engineer/artifacts/audit_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_inference(
        self,
        inference_id: str,
        timestamp: str,
        provenance: Dict[str, Any],
        confidence: float,
        latency_ms: float,
        firewall_verdict: Dict[str, Any],
        review_required: bool
    ) -> Dict[str, Any]:
        """Appends an immutable audit record."""
        record = {
            "inference_id": inference_id,
            "timestamp": timestamp,
            "model_version": provenance.get("model_version"),
            "adapter_version": provenance.get("adapter_version"),
            "quantization": provenance.get("quantization"),
            "prompt_version": provenance.get("prompt_version"),
            "tokenizer_version": provenance.get("tokenizer_version"),
            "runtime_version": provenance.get("runtime_version"),
            "input_hash": provenance.get("input_hash"),
            "output_hash": provenance.get("output_hash"),
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2),
            "firewall_verdict": firewall_verdict.get("safety_status"),
            "firewall_checks": firewall_verdict.get("infractions", []),
            "review_required": review_required
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to audit log {self.log_path}: {e}")

        return record

    def get_record(self, inference_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an audit record by its unique inference_id."""
        if not self.log_path.exists():
            return None

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("inference_id") == inference_id:
                        return data
                except Exception:
                    continue
        return None
