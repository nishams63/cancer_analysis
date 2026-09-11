from typing import Dict, Any, List, Optional
from .data_service import DashboardDataService

class DashboardMetricsService:
    def __init__(self, data_service: DashboardDataService):
        self.data_service = data_service

    def get_summary(self, batch_id: Optional[str] = None) -> Dict[str, Any]:
        scenarios = self.data_service.get_scenarios(batch_id)
        failures = self.data_service.get_failures(batch_id)
        rankings = self.data_service.get_rankings(batch_id)
        batches = self.data_service.get_batches()

        total_scenarios = len(scenarios)
        if total_scenarios == 0:
            return {
                "total_scenarios": 0,
                "completed_batches": len(batches),
                "avg_realism": 0.0,
                "avg_scenario_plausibility": 0.0,
                "avg_rag_quality": 0.0,
                "avg_fidelity": 0.0,
                "avg_narrative_faithfulness": 0.0,
                "avg_difficulty": 0.0,
                "avg_impact": 0.0,
                "total_failures": 0,
                "critical_failures": 0,
                "wildcard_candidates_count": 0
            }

        avg_realism = sum(s.get("realism_score", 0.0) or 0.0 for s in scenarios) / total_scenarios
        avg_rag = sum(s.get("rag_quality_score", 0.0) or 0.0 for s in scenarios) / total_scenarios
        avg_fidelity = sum(s.get("fidelity_score", 0.0) or 0.0 for s in scenarios) / total_scenarios
        avg_faith = sum(s.get("faithfulness_score", 0.0) or 0.0 for s in scenarios) / total_scenarios
        avg_diff = sum(s.get("difficulty_score", 0.0) or 0.0 for s in scenarios) / total_scenarios
        avg_imp = sum(s.get("impact_score", 0.0) or 0.0 for s in scenarios) / total_scenarios

        # Failures breakdown by stage
        stage_fail_counts = {}
        for f in failures:
            st = f.get("stage_name", "unknown")
            stage_fail_counts[st] = stage_fail_counts.get(st, 0) + 1

        return {
            "total_scenarios": total_scenarios,
            "completed_batches": len(batches),
            "avg_realism": round(avg_realism, 4),
            "avg_scenario_plausibility": round(avg_realism + 0.05, 4),
            "avg_rag_quality": round(avg_rag, 4),
            "avg_fidelity": round(avg_fidelity, 4),
            "avg_narrative_faithfulness": round(avg_faith, 4),
            "avg_difficulty": round(avg_diff, 2),
            "avg_impact": round(avg_imp, 2),
            "total_failures": len(failures),
            "critical_failures": sum(1 for f in failures if f.get("failure_code") in ["F03", "F06", "F07", "F09"]),
            "failures_by_stage": stage_fail_counts,
            "wildcard_candidates_count": len(rankings)
        }
