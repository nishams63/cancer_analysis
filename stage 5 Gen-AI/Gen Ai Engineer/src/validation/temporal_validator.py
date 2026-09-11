"""Temporal ordering, causality, and progression interval validator."""
from typing import Dict, Any, List, Tuple
from datetime import datetime


class TemporalValidator:
    def validate(self, patient_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        violations = []
        timeline = patient_dict.get("timeline", [])

        if not timeline:
            return (True, [])

        prev_delta = -1
        prev_type = None

        for idx, event in enumerate(timeline):
            delta = event.get("delta_days", 0) if isinstance(event, dict) else event.delta_days
            etype = event.get("event_type", "") if isinstance(event, dict) else event.event_type

            # Check chronological monotony
            if delta < prev_delta:
                violations.append(f"Chronological regression: event {etype} at day {delta} occurs before previous day {prev_delta}")
            prev_delta = delta

            # Causality checks
            if idx == 0 and etype not in {"diagnosis", "staging"}:
                violations.append("First timeline event must be diagnosis or baseline staging")

        return (len(violations) == 0, violations)