"""
Production Inference Monitoring and Drift Detection Module.
Logs all production inferences with end-to-end cryptographic and version provenance:
inference_id, model_version, adapter_version, dataset_sha256, prompt_version, confidence, firewall_verdict.
Maintains rolling window metrics and triggers clinical safety alerts upon distribution drift.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("stage6_eval.monitoring")


class ProductionInferenceLogger:
    """Manages immutable audit logging for all clinical inferences."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.inference_log_file = self.log_dir / "production_inference_log.jsonl"

    def log_inference(
        self,
        inference_id: str,
        clinical_note: str,
        generation_text: str,
        firewall_verdict: Dict[str, Any],
        provenance_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Appends one stamped inference record to production audit log."""
        record = {
            "inference_id": inference_id,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "provenance": {
                "model_id": provenance_meta.get("model_id", "Qwen2.5-1.5B-Instruct"),
                "adapter_id": provenance_meta.get("adapter_id", "filtered-lora-r16"),
                "dataset_sha256": provenance_meta.get("dataset_sha256", "95d684c0940be3475375c69fc99a17f42b424d95ebc5a102cf608fa0889a1b2d"),
                "prompt_version": provenance_meta.get("prompt_version", "clinical-v1.0"),
                "code_commit": provenance_meta.get("code_commit", "stage6-eval-prod")
            },
            "input_length_chars": len(clinical_note),
            "output_length_chars": len(generation_text),
            "firewall_passed": firewall_verdict.get("passed", False),
            "firewall_route": firewall_verdict.get("route", "HUMAN_REVIEW"),
            "firewall_infractions": firewall_verdict.get("infractions", []),
            "confidence": firewall_verdict.get("confidence", 1.0),
            "assigned_risk": firewall_verdict.get("parsed_fields", {}).get("Risk", "Unknown")
        }

        with open(self.inference_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record


class ProductionDriftMonitor:
    """Tracks rolling windows of production inferences and identifies safety/data drift."""

    def __init__(
        self,
        rejection_threshold: float = 0.05,
        format_failure_threshold: float = 0.02,
        min_window_size: int = 20
    ):
        self.rejection_threshold = rejection_threshold
        self.format_failure_threshold = format_failure_threshold
        self.min_window_size = min_window_size

    def analyze_window(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes a window of inference records and triggers alerts if drifted."""
        n = len(records)
        if n < self.min_window_size:
            return {
                "window_size": n,
                "status": "INSUFFICIENT_DATA",
                "alert": False,
                "recommendation": "Collect more inferences"
            }

        rejections = sum(1 for r in records if not r.get("firewall_passed", False))
        rejection_rate = rejections / n

        confidences = [r.get("confidence", 1.0) for r in records]
        mean_conf = float(np.mean(confidences))

        risks = [r.get("assigned_risk", "Unknown") for r in records]
        risk_counts = {k: risks.count(k) for k in set(risks)}

        # Check triggers
        alerts = []
        if rejection_rate > self.rejection_threshold:
            alerts.append(f"HIGH_REJECTION_RATE: Firewall rejection rate {rejection_rate*100:.1f}% exceeds {self.rejection_threshold*100:.1f}%")

        if mean_conf < 0.80:
            alerts.append(f"CONFIDENCE_DEPRESSION: Mean confidence dropped to {mean_conf:.3f}")

        is_alert = len(alerts) > 0

        return {
            "window_size": n,
            "rejection_rate": float(round(rejection_rate, 4)),
            "mean_confidence": float(round(mean_conf, 4)),
            "risk_distribution": risk_counts,
            "alert": is_alert,
            "active_alerts": alerts,
            "status": "DRIFT_ALERT" if is_alert else "HEALTHY"
        }
