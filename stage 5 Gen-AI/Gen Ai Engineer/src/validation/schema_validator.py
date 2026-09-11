"""Schema validator for structured synthetic patients."""
import json
import os
from typing import Dict, Any, List, Tuple


class SchemaValidator:
    def __init__(self, schema_path: str = None):
        self.schema_path = schema_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "schemas", "patient_schema.json")
        self._schema = None
        if os.path.exists(self.schema_path):
            with open(self.schema_path, "r", encoding="utf-8") as f:
                self._schema = json.load(f)

    def validate(self, patient_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        # Mandatory top-level check
        if not patient_dict.get("synthetic"):
            errors.append("Patient must have 'synthetic' == true.")

        required_fields = [
            "scenario_id", "patient_id", "demographics", "mutations",
            "biomarkers", "treatments", "dosages", "adverse_events",
            "resistance", "timeline", "missing_fields", "provenance"
        ]
        for rf in required_fields:
            if rf not in patient_dict:
                errors.append(f"Missing required field: {rf}")

        return (len(errors) == 0, errors)