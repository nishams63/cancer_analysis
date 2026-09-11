"""Range and boundary validator for demographic and clinical labs."""
from typing import Dict, Any, List, Tuple


class RangeValidator:
    BOUNDS = {
        "age": (18, 105),
        "tumor_size_cm": (0.1, 25.0),
        "creatinine_level": (0.2, 15.0),
        "serum_creatinine": (0.2, 15.0),
        "cea_level": (0.1, 1000.0),
        "ca125_level": (1.0, 5000.0),
        "anc": (50.0, 15000.0),
        "lactate": (0.5, 20.0),
        "calcium_level": (5.0, 20.0),
        "weight_kg": (30.0, 250.0),
        "crcl_ml_min": (5.0, 200.0)
    }

    def validate(self, patient_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        violations = []
        demo = patient_dict.get("demographics", {})
        age = demo.get("age")
        if age is not None:
            min_a, max_a = self.BOUNDS["age"]
            if not (min_a <= age <= max_a):
                violations.append(f"Age {age} out of bounds [{min_a}, {max_a}]")

        bios = patient_dict.get("biomarkers", {})
        for k, v in bios.items():
            if isinstance(v, (int, float)) and k in self.BOUNDS:
                min_v, max_v = self.BOUNDS[k]
                if not (min_v <= v <= max_v):
                    violations.append(f"Biomarker '{k}' value {v} out of bounds [{min_v}, {max_v}]")

        return (len(violations) == 0, violations)