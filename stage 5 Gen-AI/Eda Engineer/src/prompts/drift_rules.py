"""Prompt Drift and Scenario Compliance Rules Engine."""
import re
from typing import List, Dict, Any, Tuple
import yaml
from pydantic import BaseModel, Field
from .prompt_schema import ScenarioDefinition, RequiredEntity, EntityPresence, ForbiddenChange


class DriftCheckResult(BaseModel):
    scenario_id: str
    is_compliant: bool
    compliance_score: float = Field(..., ge=0.0, le=1.0)
    missing_required_entities: List[str] = Field(default_factory=list)
    forbidden_rule_violations: List[str] = Field(default_factory=list)
    range_violations: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


def get_default_drift_rules_catalog() -> Dict[str, Any]:
    """Catalog of generic and scenario-specific prompt drift rules."""
    return {
        "version": "1.0.0",
        "description": "Rules to detect semantic drift, condition leakage, and constraint violation in generated scenarios",
        "global_rules": [
            {
                "rule_id": "GLOBAL-DRIFT-01",
                "name": "no_fictional_hallucinated_mutations",
                "pattern": r"\\b(KRAS G12Z|EGFR T999M|TP53 XYZ)\\b",
                "description": "Generator must not hallucinate biologically impossible mutation nomenclature."
            },
            {
                "rule_id": "GLOBAL-DRIFT-02",
                "name": "preserve_specified_sex_and_age",
                "description": "Patient skeleton demographics must remain within +/- 2 years of target age and preserve biological sex."
            },
            {
                "rule_id": "GLOBAL-DRIFT-03",
                "name": "severity_inversion_prevention",
                "description": "Critical emergency scenarios (e.g. Sepsis, Calcium > 14) must not be rewritten as routine outpatient encounters."
            }
        ],
        "compliance_thresholds": {
            "min_acceptable_score": 0.85,
            "zero_tolerance_on_forbidden": True
        }
    }


def export_drift_rules_yaml(output_path: str) -> str:
    """Export drift rules to YAML."""
    import os
    rules = get_default_drift_rules_catalog()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(rules, f, default_flow_style=False, sort_keys=False)
    return output_path


class PromptDriftEvaluator:
    """Evaluates whether a generated clinical scenario or prompt complies with scenario specifications."""

    def __init__(self, scenario: ScenarioDefinition):
        self.scenario = scenario

    def evaluate(self, generated_content: str, extracted_entities: Dict[str, Any] = None) -> DriftCheckResult:
        """Run all compliance and drift checks against generated text."""
        if extracted_entities is None:
            extracted_entities = {}

        missing_req = []
        forbidden_violations = []
        range_violations = []

        total_checks = 0
        passed_checks = 0

        # 1. Check Required Entities (text search & value match)
        for req in self.scenario.required_entities:
            total_checks += 1
            matched = False
            if req.name in extracted_entities:
                val = extracted_entities[req.name]
                if req.min_value is not None and req.max_value is not None:
                    if isinstance(val, (int, float)) and req.min_value <= val <= req.max_value:
                        matched = True
                    else:
                        range_violations.append(f"{req.name}: value {val} outside [{req.min_value}, {req.max_value}]")
                elif req.allowed_values:
                    if any(av.lower() in str(val).lower() for av in req.allowed_values):
                        matched = True
                else:
                    matched = True
            else:
                # Text fallback search
                if req.allowed_values:
                    for av in req.allowed_values:
                        if av.lower() in generated_content.lower():
                            matched = True
                            break
                elif req.name.lower() in generated_content.lower():
                    matched = True

            if matched:
                passed_checks += 1
            else:
                missing_req.append(req.name)

        # 2. Check Forbidden Modifications
        for forbid in self.scenario.forbidden_modifications:
            total_checks += 1
            if re.search(forbid.forbidden_pattern, generated_content, re.IGNORECASE):
                forbidden_violations.append(f"{forbid.rule_id}: {forbid.description}")
            else:
                passed_checks += 1

        # Calculate Compliance Score
        score = (passed_checks / total_checks) if total_checks > 0 else 1.0
        # If any forbidden violation occurs, compliance drops significantly
        if forbidden_violations:
            score = min(score, 0.4)

        is_compliant = (len(missing_req) == 0) and (len(forbidden_violations) == 0) and (len(range_violations) == 0)

        return DriftCheckResult(
            scenario_id=self.scenario.scenario_id,
            is_compliant=is_compliant,
            compliance_score=round(score, 3),
            missing_required_entities=missing_req,
            forbidden_rule_violations=forbidden_violations,
            range_violations=range_violations,
            details={
                "total_checks": total_checks,
                "passed_checks": passed_checks
            }
        )
