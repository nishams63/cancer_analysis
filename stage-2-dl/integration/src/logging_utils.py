"""
Stage 2 Deep Learning - Structured Observability & Logging Module

Logs structured JSON-like operational records:
  - Request ID & Patient ID
  - Inference Mode & Selected Model
  - Latency breakdown (validation, pathology, temporal, fusion, OOD)
  - OOD status and calibrated risk
  - Safe error categorization (no raw imagery or sensitive traces logged)
"""
import logging
import sys
import time
import uuid
from typing import Dict, Any, Optional

logger = logging.getLogger("oncology_integration")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class LatencyTracker:
    """Context manager for timing pipeline components."""
    def __init__(self):
        self.timings: Dict[str, float] = {}

    def time_block(self, name: str):
        class _Timer:
            def __init__(self, tracker, block_name):
                self.tracker = tracker
                self.block_name = block_name
                self.start = 0.0

            def __enter__(self):
                self.start = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed_ms = (time.perf_counter() - self.start) * 1000.0
                self.tracker.timings[self.block_name] = round(elapsed_ms, 2)

        return _Timer(self, name)


def log_inference_event(
    patient_id: str,
    inference_mode: str,
    model_version: str,
    risk_level: str,
    ood_status: str,
    latency_ms: float,
    request_id: Optional[str] = None,
    error: Optional[str] = None
):
    """Logs a structured event for production/research observability."""
    req_id = request_id or str(uuid.uuid4())[:8]
    status = "SUCCESS" if not error else "FAILED"
    msg = (
        f"req_id={req_id} patient_id={patient_id} status={status} "
        f"mode={inference_mode} model={model_version} risk={risk_level} "
        f"ood={ood_status} latency_ms={latency_ms:.1f}"
    )
    if error:
        msg += f" error={error}"
        logger.error(msg)
    else:
        logger.info(msg)
