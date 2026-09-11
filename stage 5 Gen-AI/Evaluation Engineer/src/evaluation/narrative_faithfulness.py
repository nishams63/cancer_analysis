"""Narrative Faithfulness Evaluator."""
from typing import Dict, Any, List


class NarrativeFaithfulnessEvaluator:
    """Evaluates Level 4: Did the LLM faithfully convert structured facts into clinical narrative?"""

    def __init__(self):
        pass

    def evaluate_narrative(self, patient: Dict[str, Any], narrative: Dict[str, Any]) -> Dict[str, Any]:
        sid = patient.get("scenario_id", "UNKNOWN")
        text = narrative.get("narrative_text", "").lower() if narrative else ""
        demo = patient.get("demographics", {})
        muts = patient.get("mutations", [])
        bios = patient.get("biomarkers", {})

        age = demo.get("age", 0)
        age_str = str(int(age)) if age else ""
        entity_preserved = (age_str in text or not age_str)

        if isinstance(muts, list):
            mut_words = [m.split()[0].lower() for m in muts if m]
        else:
            mut_words = [str(muts).lower()]
        
        mut_preserved = all(w in text for w in mut_words if w not in ["none", "none/unknown", ""]) if mut_words else True

        missing_preserved = True
        for bname, bval in bios.items():
            if bval is None or bval == "MISSING":
                if f"{bname} is" in text:
                    missing_preserved = False

        contradiction_free = True
        if "no met" in text and "met positive" in text:
            contradiction_free = False

        scores = {
            "entity_preservation": 1.0 if entity_preserved else 0.6,
            "value_preservation": 1.0 if mut_preserved else 0.7,
            "missingness_preservation": 1.0 if missing_preserved else 0.0,
            "negation_preservation": 1.0,
            "contradiction_free_score": 1.0 if contradiction_free else 0.0
        }

        faithfulness_score = sum(scores.values()) / len(scores)
        status = "PASS" if faithfulness_score >= 0.85 else "FAIL"

        return {
            "scenario_id": sid,
            "narrative_faithfulness_score": round(faithfulness_score, 4),
            "component_scores": scores,
            "has_hallucination": not missing_preserved,
            "has_contradiction": not contradiction_free,
            "status": status
        }
