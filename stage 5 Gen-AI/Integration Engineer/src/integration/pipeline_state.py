from typing import Dict, Any, List, Optional
import time

class ScenarioState:
    def __init__(self, scenario_id: str):
        self.scenario_id = scenario_id
        self.current_step = "INITIALIZED"
        self.completed_steps: List[str] = []
        self.failed_step: Optional[str] = None
        self.retry_count = 0
        self.last_error: Optional[str] = None
        self.step_latencies: Dict[str, float] = {}

    def mark_step_start(self, step: str):
        self.current_step = step

    def mark_step_complete(self, step: str, latency_ms: float = 0.0):
        self.completed_steps.append(step)
        self.step_latencies[step] = round(latency_ms, 2)
        self.current_step = f"COMPLETED_{step}"

    def mark_step_failed(self, step: str, error: str):
        self.failed_step = step
        self.last_error = error
        self.current_step = f"FAILED_{step}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "failed_step": self.failed_step,
            "retry_count": self.retry_count,
            "last_error": self.last_error,
            "step_latencies": self.step_latencies
        }

class PipelineState:
    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        self.scenario_states: Dict[str, ScenarioState] = {}

    def get_or_create_scenario(self, scenario_id: str) -> ScenarioState:
        if scenario_id not in self.scenario_states:
            self.scenario_states[scenario_id] = ScenarioState(scenario_id)
        return self.scenario_states[scenario_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "scenarios": {k: v.to_dict() for k, v in self.scenario_states.items()}
        }
