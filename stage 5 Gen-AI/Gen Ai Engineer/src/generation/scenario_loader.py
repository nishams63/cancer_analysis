"""Scenario loader: extracts approved scenario definitions from prompt library."""
import os
import yaml
from typing import Dict, Any, List, Optional
from src.prompts.prompt_schema import ScenarioDefinition


class ScenarioLoader:
    def __init__(self, prompt_lib_path: str = None, catalog_path: str = None):
        self.prompt_lib_path = prompt_lib_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "prompts", "prompt_library.yaml")
        self.catalog_path = catalog_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "prompts", "scenario_catalog.yaml")
        self._scenarios_cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        target_path = self.prompt_lib_path if os.path.exists(self.prompt_lib_path) else self.catalog_path
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Neither prompt library nor scenario catalog found at {target_path}")
        
        with open(target_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            scenarios = data.get("scenarios", [])
            for sc in scenarios:
                sid = sc.get("scenario_id")
                if sid:
                    self._scenarios_cache[sid] = sc

    def list_scenarios(self) -> List[str]:
        return list(self._scenarios_cache.keys())

    def get_scenario(self, scenario_id: str) -> Dict[str, Any]:
        if scenario_id not in self._scenarios_cache:
            raise KeyError(f"Scenario '{scenario_id}' not found in library. Available: {list(self._scenarios_cache.keys())}")
        return self._scenarios_cache[scenario_id]

    def get_all_scenarios(self) -> List[Dict[str, Any]]:
        return list(self._scenarios_cache.values())