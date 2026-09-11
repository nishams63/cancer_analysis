"""Failure Classifier: maps observed errors to F01-F20 with explanations."""
from typing import Dict, Any, List


class FailureClassifier:
    """Categorizes failures into formal taxonomy codes F01 to F20."""

    def __init__(self, taxonomy: Dict[str, Any] = None):
        self.taxonomy = taxonomy or {}

    def classify_stage_failure(self, stage_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        codes = stage_result.get("failure_codes", [])
        stage = stage_result.get("stage", "unknown")
        sid = stage_result.get("scenario_id", "unknown")
        expected = stage_result.get("expected_output", "")
        actual = stage_result.get("actual_output", "")
        confidence = stage_result.get("confidence", 0.0)
        evidence = stage_result.get("evidence", "")

        classified = []
        for c in codes:
            classified.append({
                "failure_code": c,
                "stage": stage,
                "scenario_id": sid,
                "expected": str(expected),
                "actual": str(actual),
                "confidence": confidence,
                "evidence": evidence
            })
        return classified
