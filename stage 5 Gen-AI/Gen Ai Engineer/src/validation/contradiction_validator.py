"""Internal clinical contradiction validator."""
import re
from typing import Dict, Any, List, Tuple


class ContradictionValidator:
    def validate(self, text: str) -> Tuple[bool, List[str]]:
        violations = []
        # Check for adjacent contradictory patterns within same narrative
        pairs = [
            (r"no evidence of MET amplification", r"MET amplification was detected"),
            (r"denies shortness of breath", r"marked dyspnea"),
            (r"normal creatinine clearance", r"severe renal failure"),
            (r"disease is stable", r"progressive disease confirmed on CT")
        ]
        for neg_pat, pos_pat in pairs:
            if re.search(neg_pat, text, re.IGNORECASE) and re.search(pos_pat, text, re.IGNORECASE):
                violations.append(f"Contradiction detected: Narrative contains both '{neg_pat}' and '{pos_pat}'")

        return (len(violations) == 0, violations)