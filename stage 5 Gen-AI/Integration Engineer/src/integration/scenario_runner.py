import time
from typing import Dict, Any, Optional
from ..adapters import Stage1Adapter, Stage2Adapter, Stage3Adapter, Stage4Adapter
from .checkpoint_manager import CheckpointManager
from .pipeline_state import ScenarioState
from ..utils.logging import get_logger

logger = get_logger("ScenarioRunner")

class ScenarioRunner:
    def __init__(self, checkpoint_manager: CheckpointManager, continue_on_error: bool = True):
        self.checkpoint_manager = checkpoint_manager
        self.continue_on_error = continue_on_error
        self.stage1 = Stage1Adapter()
        self.stage2 = Stage2Adapter()
        self.stage3 = Stage3Adapter()
        self.stage4 = Stage4Adapter()

    def run_stages(self, batch_id: str, scenario_id: str, scenario_data: Dict[str, Any], state: ScenarioState) -> Dict[str, Any]:
        stages_output = {}

        # Stage 1
        state.mark_step_start("stage1")
        s1_res = self.stage1.run(scenario_id, scenario_data)
        stages_output["stage1"] = s1_res.to_dict()
        state.mark_step_complete("stage1", s1_res.latency_ms)

        # Stage 2
        state.mark_step_start("stage2")
        s2_res = self.stage2.run(scenario_id, scenario_data)
        stages_output["stage2"] = s2_res.to_dict()
        state.mark_step_complete("stage2", s2_res.latency_ms)

        # Stage 3
        state.mark_step_start("stage3")
        s3_res = self.stage3.run(scenario_id, scenario_data)
        stages_output["stage3"] = s3_res.to_dict()
        state.mark_step_complete("stage3", s3_res.latency_ms)

        # Stage 4
        state.mark_step_start("stage4")
        s4_res = self.stage4.run(scenario_id, scenario_data)
        stages_output["stage4"] = s4_res.to_dict()
        state.mark_step_complete("stage4", s4_res.latency_ms)

        return stages_output
