"""Causal and chronological temporal constraint rules."""
from typing import List, Dict, Any

class TemporalRulesBuilder:
    def build(self) -> List[Dict[str, Any]]:
        return [
            {
                "rule_id": "T001",
                "name": "treatment_after_diagnosis",
                "condition": "treatment_start_date >= diagnosis_date",
                "description": "Cancer treatment start timestamp must occur on or after the initial diagnosis date."
            },
            {
                "rule_id": "T002",
                "name": "followup_after_treatment",
                "condition": "followup_date >= treatment_start_date",
                "description": "Post-treatment surveillance observation must occur on or after treatment initiation."
            },
            {
                "rule_id": "T003",
                "name": "toxicity_during_active_therapy",
                "condition": "toxicity_onset_date >= treatment_start_date",
                "description": "Drug-induced adverse events must follow the administration of the causative agent."
            },
            {
                "rule_id": "T004",
                "name": "progression_time_window",
                "condition": "progression_date >= treatment_start_date + 14_days",
                "description": "Documented biological disease progression requires minimum observation window."
            }
        ]
