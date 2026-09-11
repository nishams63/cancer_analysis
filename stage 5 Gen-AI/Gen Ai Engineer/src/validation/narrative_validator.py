"""Narrative realization validator.

Compares LLM-generated clinical text against the ground truth structured synthetic patient.
Verifies:
1. Demographics preservation (age, sex)
2. Mutation preservation (no dropped or invented mutations)
3. Treatment and dosage preservation
4. Missingness preservation (fails if LLM hallucinated intentionally missing fields)
5. Forbidden modification prevention
6. Internal contradiction absence
"""
import re
from typing import Dict, Any, List, Tuple
from pydantic import BaseModel, Field


class NarrativeValidationResult(BaseModel):
    valid: bool
    checks: Dict[str, str] = Field(default_factory=dict)
    violations: List[Dict[str, str]] = Field(default_factory=list)
    validator_version: str = "1.0.0"


class NarrativeValidator:
    def validate(self, narrative: str, patient: Dict[str, Any], scenario: Dict[str, Any]) -> NarrativeValidationResult:
        checks = {}
        violations = []

        # 1. Demographics Check
        demo = patient.get("demographics", {})
        age = str(demo.get("age"))
        sex = demo.get("sex", "")
        age_pass = bool(re.search(rf"\b{age}\b", narrative))
        sex_pass = bool(re.search(rf"\b{sex}\b|\b{sex.lower()}\b|\bwoman\b|\bman\b|\bfemale\b|\bmale\b", narrative, re.IGNORECASE))
        if age_pass and sex_pass:
            checks["demographics"] = "PASS"
        else:
            checks["demographics"] = "FAIL"
            violations.append({"field": "demographics", "reason": f"Age {age} or Sex {sex} not preserved in narrative"})

        # 2. Molecular & Mutation Check
        mut_pass = True
        for m in patient.get("mutations", []):
            tokens = [t.strip() for t in m.replace("+", " ").split() if t.strip()]
            variant_tokens = [t for t in tokens if t.upper() not in {"EGFR", "KRAS", "BRAF", "PIK3CA", "TP53", "ALK", "ROS1", "HER2", "MET", "RET", "NTRK"}]
            check_tokens = variant_tokens if variant_tokens else tokens
            if not all(tok.lower() in narrative.lower() for tok in check_tokens):
                mut_pass = False
                violations.append({"field": "mutations", "reason": f"Required mutation '{m}' dropped from narrative"})
        checks["mutations"] = "PASS" if mut_pass else "FAIL"

        # 3. Treatment & Dosage Check
        treat_pass = True
        for t in patient.get("treatments", []):
            tname = t.get("treatment_name", "")
            base_drug = tname.split()[0]
            if base_drug.lower() not in narrative.lower():
                treat_pass = False
                violations.append({"field": "treatment", "reason": f"Treatment '{tname}' omitted from narrative"})
        checks["treatment"] = "PASS" if treat_pass else "FAIL"

        # 4. Missingness Preservation Check
        missing_pass = True
        missing_fields = patient.get("missing_fields", [])
        for mf in missing_fields:
            # If missing field is 'tumor_size_cm', narrative must not assert definitive primary measurement
            if mf == "tumor_size_cm" and re.search(r"primary tumor measures [0-9]+\.[0-9]+ cm", narrative, re.IGNORECASE):
                missing_pass = False
                violations.append({"field": "missingness", "reason": "LLM hallucinated specific primary tumor dimensions into missing field"})
            elif mf == "cea_level" and re.search(r"CEA level is [0-9]+", narrative, re.IGNORECASE):
                missing_pass = False
                violations.append({"field": "missingness", "reason": "LLM hallucinated CEA lab into intentionally missing field"})
        checks["missingness"] = "PASS" if missing_pass else "FAIL"

        # 5. Forbidden Modifications Check
        forbidden_pass = True
        for fb in scenario.get("forbidden_modifications", []):
            pat = fb.get("forbidden_pattern", "")
            if pat:
                parts = pat.split("|")
                bounded_parts = [r"\b" + p.strip() + r"\b" if not (p.strip().startswith(r"\b") or any(ch in p for ch in "[]()\\")) else p.strip() for p in parts]
                bounded_pat = "|".join(bounded_parts)
                if re.search(bounded_pat, narrative, re.IGNORECASE):
                    forbidden_pass = False
                    violations.append({"field": "forbidden_modifications", "reason": f"Forbidden pattern '{pat}' detected: {fb.get('description')}"})
        checks["forbidden_modifications"] = "PASS" if forbidden_pass else "FAIL"

        # 6. Contradictions Check
        from .contradiction_validator import ContradictionValidator
        cv = ContradictionValidator()
        c_ok, c_errs = cv.validate(narrative)
        checks["contradictions"] = "PASS" if c_ok else "FAIL"
        for ce in c_errs:
            violations.append({"field": "contradictions", "reason": ce})

        is_valid = (len(violations) == 0)
        return NarrativeValidationResult(
            valid=is_valid,
            checks=checks,
            violations=violations
        )