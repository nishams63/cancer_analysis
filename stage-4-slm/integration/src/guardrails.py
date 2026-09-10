"""
Clinical Safety Guardrails for Stage 4 SLM Inference.
Verifies entity preservation, prevents toxic hallucinations, and intercepts clinical contradictions.
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Set

logger = logging.getLogger("stage4.integration.guardrails")

KNOWN_DRUGS = [
    "Docetaxel", "Cisplatin", "Carboplatin", "Paclitaxel", "Pemetrexed",
    "Nivolumab", "Durvalumab", "Atezolizumab", "Erlotinib", "Osimertinib",
    "Trastuzumab", "Olaparib", "Alectinib", "Warfarin", "Pembrolizumab"
]


class ClinicalSafetyGuardrails:
    """Multi-dimensional safety gate enforcing clinical reliability of generated recommendations."""

    def __init__(self):
        pass

    @staticmethod
    def extract_drugs(text: str) -> Set[str]:
        found = set()
        for d in KNOWN_DRUGS:
            if re.search(rf"\b{re.escape(d)}\b", text, re.IGNORECASE):
                found.add(d.capitalize())
        return found

    @staticmethod
    def extract_dosages(text: str) -> Set[str]:
        matches = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mg/m2|mg|mcg|g)\b", text, re.IGNORECASE)
        return set([re.sub(r"\s+", "", m.lower()) for m in matches])

    def audit(self, clinical_note: str, triad: Dict[str, str]) -> Tuple[bool, List[str], Dict[str, List[str]]]:
        """
        Audits generated triad against source clinical note.
        Returns (passed, list_of_warnings, preserved_entities_dict).
        """
        warnings = []
        note_drugs = self.extract_drugs(clinical_note)
        note_dosages = self.extract_dosages(clinical_note)

        pred_text = f"{triad.get('target_risk', '')} {triad.get('target_key_finding', '')} {triad.get('target_action', '')}"
        pred_drugs = self.extract_drugs(pred_text)
        pred_dosages = self.extract_dosages(pred_text)

        # 1. Check for omitted drugs
        omitted_drugs = note_drugs - pred_drugs
        if omitted_drugs:
            warnings.append(f"Entity Preservation: Note drug(s) omitted in generation: {list(omitted_drugs)}")

        # 2. Check for invented/hallucinated drugs
        invented_drugs = pred_drugs - note_drugs
        if invented_drugs:
            warnings.append(f"Safety Violation: Hallucinated drug(s) not in clinical note: {list(invented_drugs)}")

        # 3. Contradiction Detection: Critical note vs continuation action
        is_critical_note = bool(re.search(
            r"(acute respiratory distress|hypoxia|septic shock|febrile neutropenia|spO2 < 90|cardiac arrest|anaphylaxis|grade 4)",
            clinical_note,
            re.IGNORECASE
        ))
        action_continues = bool(re.search(
            r"(continue standard|maintain current|administer scheduled)",
            triad.get("target_action", ""),
            re.IGNORECASE
        ))
        if is_critical_note and action_continues:
            warnings.append("Clinical Contradiction: Patient exhibits critical toxicity but generated action maintains full regimen.")

        passed = (len(invented_drugs) == 0) and not (is_critical_note and action_continues)

        preserved = {
            "drugs": list(note_drugs.intersection(pred_drugs)),
            "dosages": list(note_dosages.intersection(pred_dosages))
        }

        return passed, warnings, preserved
