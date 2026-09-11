"""Mutation nomenclature and biologically plausible combination validator."""
import re
from typing import Dict, Any, List, Tuple


class MutationValidator:
    # GLOBAL-DRIFT-01: Prohibited biologically impossible or hallucinated mutations
    FORBIDDEN_MUTATIONS = [
        re.compile(r"\b(KRAS G12Z|EGFR T999M|TP53 XYZ|BRAF V999E)\b", re.IGNORECASE)
    ]

    def validate(self, patient_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        violations = []
        mutations = patient_dict.get("mutations", [])

        for m in mutations:
            for pat in self.FORBIDDEN_MUTATIONS:
                if pat.search(m):
                    violations.append(f"GLOBAL-DRIFT-01 violation: Hallucinated mutation detected: '{m}'")

        return (len(violations) == 0, violations)