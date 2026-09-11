from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum
import time

class StageStatus(str, Enum):
    SUCCESS = "SUCCESS"
    MODEL_FAILURE = "MODEL_FAILURE"
    PIPELINE_FAILURE = "PIPELINE_FAILURE"
    SKIPPED_INPUT_UNAVAILABLE = "SKIPPED_INPUT_UNAVAILABLE"
    STAGE_ERROR = "STAGE_ERROR"

class StageAdapterResult:
    def __init__(
        self,
        stage: str,
        scenario_id: str,
        status: str,
        prediction: Optional[Dict[str, Any]] = None,
        confidence: Optional[Dict[str, Any]] = None,
        raw_output: Optional[Dict[str, Any]] = None,
        model_version: str = "v1.0",
        latency_ms: float = 0.0,
        error: Optional[str] = None
    ):
        self.stage = stage
        self.scenario_id = scenario_id
        self.status = status
        self.prediction = prediction or {}
        self.confidence = confidence or {}
        self.raw_output = raw_output or {}
        self.model_version = model_version
        self.latency_ms = round(latency_ms, 2)
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "scenario_id": self.scenario_id,
            "status": self.status,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "raw_output": self.raw_output,
            "model_version": self.model_version,
            "latency_ms": self.latency_ms,
            "error": self.error
        }

class BaseStageAdapter(ABC):
    def __init__(self, stage_name: str, model_version: str = "v1.0"):
        self.stage_name = stage_name
        self.model_version = model_version

    @abstractmethod
    def run(self, scenario_id: str, scenario_data: Dict[str, Any]) -> StageAdapterResult:
        raise NotImplementedError
