"""Counterfactual Scenario Generator.

Generates paired scenarios where exactly ONE controlled variable is modified
while strictly locking all other patient features.
"""
import copy
from typing import Dict, Any, Tuple
from .patient_builder import StructuredSyntheticPatient


class CounterfactualPairResult:
    def __init__(self, original: StructuredSyntheticPatient, counterfactual: StructuredSyntheticPatient,
                 changed_variable: str, original_value: Any, counterfactual_value: Any,
                 is_valid: bool, unexpected_changes: int):
        self.original = original
        self.counterfactual = counterfactual
        self.changed_variable = changed_variable
        self.original_value = original_value
        self.counterfactual_value = counterfactual_value
        self.is_valid = is_valid
        self.unexpected_changes = unexpected_changes

    @property
    def target_variable(self) -> str:
        return self.changed_variable

    @property
    def counterfactual_patient(self) -> StructuredSyntheticPatient:
        return self.counterfactual

    @property
    def factual_patient(self) -> StructuredSyntheticPatient:
        return self.original

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counterfactual_id": f"{self.original.patient_id}-CF",
            "original_patient_id": self.original.patient_id,
            "counterfactual_patient_id": self.counterfactual.patient_id,
            "scenario_id": self.original.scenario_id,
            "changed_variable": self.changed_variable,
            "original_value": self.original_value,
            "counterfactual_value": self.counterfactual_value,
            "is_valid": self.is_valid,
            "unexpected_changes": self.unexpected_changes,
            "factual_patient": self.original.to_dict(),
            "counterfactual_patient": self.counterfactual.to_dict(),
            "traceability": {
                "target_variable": self.changed_variable,
                "original_value": self.original_value,
                "counterfactual_value": self.counterfactual_value
            }
        }


class CounterfactualGenerator:
    """Generates and validates counterfactual pairs with strict single-variable mutation."""

    def generate_counterfactual(self, patient: StructuredSyntheticPatient = None, target_variable: str = "treatments",
                                new_value: Any = None, factual_patient: StructuredSyntheticPatient = None,
                                counterfactual_value: Any = None) -> CounterfactualPairResult:
        patient = patient or factual_patient
        if new_value is None:
            new_value = counterfactual_value
        orig_dict = patient.to_dict()
        cf_dict = copy.deepcopy(orig_dict)

        # Update counterfactual ID
        cf_dict["patient_id"] = patient.patient_id + "-CF"
        cf_dict["provenance"]["is_counterfactual"] = True
        cf_dict["provenance"]["counterfactual_target"] = target_variable

        orig_val = None
        # Normalize target variable alias
        norm_map = {"dosage": "dosages", "treatment": "treatments", "mutation": "mutations"}
        actual_target = norm_map.get(target_variable, target_variable)

        # Apply the single controlled modification
        if actual_target == "treatments":
            orig_val = orig_dict["treatments"]
            cf_dict["treatments"] = new_value if isinstance(new_value, list) else [{"treatment_name": str(new_value), "line": 1}]
        elif actual_target == "dosages":
            orig_val = orig_dict["dosages"]
            cf_dict["dosages"] = new_value
        elif actual_target == "mutations":
            orig_val = orig_dict["mutations"]
            cf_dict["mutations"] = new_value if isinstance(new_value, list) else [new_value]
        elif actual_target == "resistance":
            orig_val = orig_dict["resistance"]["status"]
            cf_dict["resistance"]["status"] = new_value
        elif actual_target in orig_dict.get("biomarkers", {}):
            orig_val = orig_dict["biomarkers"].get(actual_target)
            cf_dict["biomarkers"][actual_target] = new_value
        elif actual_target == "age":
            orig_val = orig_dict["demographics"]["age"]
            cf_dict["demographics"]["age"] = new_value
        else:
            orig_val = orig_dict.get(actual_target, "N/A")
            cf_dict[actual_target] = new_value

        cf_patient = StructuredSyntheticPatient(**cf_dict)

        # Verification: count differences across top-level fields
        unexpected_changes = 0
        locked_fields = ["demographics", "mutations", "biomarkers", "treatments", "dosages", "adverse_events", "resistance"]
        for f in locked_fields:
            if f == actual_target or (actual_target in orig_dict.get(f, {}) if isinstance(orig_dict.get(f), dict) else False):
                continue
            if orig_dict[f] != cf_dict[f]:
                unexpected_changes += 1


        is_valid = (unexpected_changes == 0)

        return CounterfactualPairResult(
            original=patient,
            counterfactual=cf_patient,
            changed_variable=target_variable,
            original_value=orig_val,
            counterfactual_value=new_value,
            is_valid=is_valid,
            unexpected_changes=unexpected_changes
        )