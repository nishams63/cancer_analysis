"""Master Constraint Validator for structured synthetic oncology patients."""
from typing import Dict, Any, List, Tuple
from .schema_validator import SchemaValidator
from .range_validator import RangeValidator
from .mutation_validator import MutationValidator
from .treatment_validator import TreatmentValidator
from .dosage_validator import DosageValidator
from .temporal_validator import TemporalValidator
from .prompt_drift_validator import PromptDriftValidator


class ConstraintValidator:
    def __init__(self):
        self.schema_validator = SchemaValidator()
        self.range_validator = RangeValidator()
        self.mutation_validator = MutationValidator()
        self.treatment_validator = TreatmentValidator()
        self.dosage_validator = DosageValidator()
        self.temporal_validator = TemporalValidator()
        self.drift_validator = PromptDriftValidator()

    def validate_patient(self, patient_dict: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        violations = []

        # 1. Schema
        s_ok, s_errs = self.schema_validator.validate(patient_dict)
        violations.extend([{"rule_id": "SCHEMA_ERR", "field": "schema", "reason": e} for e in s_errs])

        # 2. Range
        r_ok, r_errs = self.range_validator.validate(patient_dict)
        violations.extend([{"rule_id": "RANGE_ERR", "field": "range", "reason": e} for e in r_errs])

        # 3. Mutation
        m_ok, m_errs = self.mutation_validator.validate(patient_dict)
        violations.extend([{"rule_id": "MUTATION_ERR", "field": "mutations", "reason": e} for e in m_errs])

        # 4. Treatment
        t_ok, t_errs = self.treatment_validator.validate(patient_dict, scenario)
        violations.extend([{"rule_id": "TREATMENT_ERR", "field": "treatment", "reason": e} for e in t_errs])

        # 5. Dosage
        d_ok, d_errs = self.dosage_validator.validate(patient_dict, scenario)
        violations.extend([{"rule_id": "DOSAGE_ERR", "field": "dosage", "reason": e} for e in d_errs])

        # 6. Temporal
        tm_ok, tm_errs = self.temporal_validator.validate(patient_dict)
        violations.extend([{"rule_id": "TEMPORAL_ERR", "field": "timeline", "reason": e} for e in tm_errs])

        # 7. Prompt Drift
        dr_ok, dr_errs, score = self.drift_validator.validate(patient_dict, scenario)
        violations.extend([{"rule_id": "DRIFT_ERR", "field": "prompt_drift", "reason": e} for e in dr_errs])

        is_valid = (len(violations) == 0)
        return {
            "valid": is_valid,
            "violations": violations,
            "compliance_score": score,
            "validator_version": "1.0.0"
        }