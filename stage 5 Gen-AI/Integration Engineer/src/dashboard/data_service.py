from typing import Dict, Any, List, Optional
from ..storage.result_store import ResultStore
from ..storage.batch_store import BatchStore
from ..storage.scenario_store import ScenarioStore
from ..storage.failure_store import FailureStore
from ..utils.serialization import json_loads

class DashboardDataService:
    def __init__(self, result_store: Optional[ResultStore] = None):
        self.result_store = result_store or ResultStore()
        self.batch_store = BatchStore(self.result_store)
        self.scenario_store = ScenarioStore(self.result_store)
        self.failure_store = FailureStore(self.result_store)

    def get_batches(self) -> List[Dict[str, Any]]:
        return self.batch_store.list_batches()

    def get_scenarios(self, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.scenario_store.list_scenarios(batch_id)

    def get_scenario_detail(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        return self.scenario_store.get_scenario(scenario_id)

    def get_failures(self, batch_id: Optional[str] = None, failure_code: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.failure_store.list_failures(batch_id, failure_code)

    def get_rankings(self, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.failure_store.list_rankings(batch_id)

    def get_counterfactuals(self, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.result_store.get_connection() as conn:
            if batch_id:
                rows = conn.execute("SELECT * FROM counterfactual_results WHERE batch_id = ?", (batch_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM counterfactual_results").fetchall()
            return [dict(r) for r in rows]
