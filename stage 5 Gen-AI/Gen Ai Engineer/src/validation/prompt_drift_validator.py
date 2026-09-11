"""Prompt drift and scenario compliance validator."""
from typing import Dict, Any, List, Tuple
from src.prompts.drift_rules import PromptDriftEvaluator
from src.prompts.prompt_schema import ScenarioDefinition


class PromptDriftValidator:
    def validate(self, patient_dict: Dict[str, Any], scenario: Dict[str, Any]) -> Tuple[bool, List[str], float]:
        violations = []
        score = 1.0

        req_entities = scenario.get("required_entities", [])
        for req in req_entities:
            name = req.get("name")
            etype = req.get("entity_type")
            allowed = req.get("allowed_values", [])

            if etype == "mutation":
                patient_muts = patient_dict.get("mutations", [])
                if allowed:
                    all_muts_str = " ".join(patient_muts).lower()
                    matched = False
                    for av in allowed:
                        av_lower = av.lower()
                        if any(av_lower in pm.lower() or pm.lower() in av_lower for pm in patient_muts):
                            matched = True
                            break
                        tokens = [t.strip() for t in av_lower.replace("+", " ").split() if t.strip()]
                        if all(tok in all_muts_str for tok in tokens):
                            matched = True
                            break
                    if not matched:
                        violations.append(f"Prompt drift: Required mutation '{allowed}' missing from patient {patient_muts}")
            elif etype in {"lab", "biomarker", "vital"}:
                patient_bios = patient_dict.get("biomarkers", {})
                if name not in patient_bios:
                    # Might be in missing_fields if intentional
                    if name not in patient_dict.get("missing_fields", []):
                        violations.append(f"Prompt drift: Required lab/biomarker '{name}' missing")

        # Global drift checks
        age = patient_dict.get("demographics", {}).get("age", 50)
        sk_age = scenario.get("patient_skeleton", {}).get("age")
        if sk_age and abs(age - sk_age) > 2:
            # Check if overridden by required_entities
            has_age_override = any(r.get("name") == "age" for r in req_entities)
            if not has_age_override:
                violations.append(f"GLOBAL-DRIFT-02 violation: Age shifted from target skeleton {sk_age} to {age}")

        if violations:
            score = max(0.0, 1.0 - (len(violations) * 0.25))

        return (len(violations) == 0, violations, score)