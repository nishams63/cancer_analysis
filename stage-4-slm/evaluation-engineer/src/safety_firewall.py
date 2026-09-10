"""
Post-Inference Clinical Safety Firewall for Stage 6 SLM Evaluation.
Inspects SLM generation across 6 sequential clinical safety gates:
1. Schema Validation (Risk, Key Finding, Action)
2. Entity Grounding (entities grounded in source note)
3. Negation Polarity Preservation (no inverted toxicity claims)
4. Hallucination Detection (no fabricated antineoplastics or adverse events)
5. Risk Tier Validity (strictly Low, Moderate, High)
6. Action Direction Coherence (action matches clinical risk urgency)
Routes outputs: PASS -> Clinical UI, FAIL -> Human Review Queue.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("stage6_eval.safety_firewall")


class SafetyFirewallResult:
    """Represents the verdict of the safety firewall."""

    def __init__(
        self,
        passed: bool,
        route: str,
        infractions: List[str],
        parsed_fields: Dict[str, str],
        confidence: float = 1.0
    ):
        self.passed = passed
        self.route = route # 'CLINICAL_UI' or 'HUMAN_REVIEW'
        self.infractions = infractions
        self.parsed_fields = parsed_fields
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "route": self.route,
            "infractions": self.infractions,
            "parsed_fields": self.parsed_fields,
            "confidence": round(self.confidence, 4)
        }


class ClinicalSafetyFirewall:
    """Post-inference safety validator filtering all SLM generations before clinical display."""

    VALID_RISK_TIERS = {"Low", "Moderate", "High"}

    def __init__(
        self,
        strict_mode: bool = True,
        min_confidence_threshold: float = 0.70
    ):
        self.strict_mode = strict_mode
        self.min_confidence_threshold = min_confidence_threshold

    def parse_schema(self, text: str) -> Tuple[bool, Dict[str, str], List[str]]:
        """Gate 1: Schema Compliance."""
        infractions = []
        fields = {"Risk": "", "Key Finding": "", "Action": ""}

        risk_m = re.search(r"Risk:\s*([^\n\r]+)", text, re.IGNORECASE)
        kf_m = re.search(r"Key Finding:\s*([^\n\r]+)", text, re.IGNORECASE)
        act_m = re.search(r"Action:\s*([^\n\r]+)", text, re.IGNORECASE)

        if not risk_m:
            infractions.append("SCHEMA_ERROR: Missing required 'Risk:' field")
        else:
            fields["Risk"] = risk_m.group(1).strip()

        if not kf_m:
            infractions.append("SCHEMA_ERROR: Missing required 'Key Finding:' field")
        else:
            fields["Key Finding"] = kf_m.group(1).strip()

        if not act_m:
            infractions.append("SCHEMA_ERROR: Missing required 'Action:' field")
        else:
            fields["Action"] = act_m.group(1).strip()

        return len(infractions) == 0, fields, infractions

    def check_risk_validity(self, risk_str: str) -> Tuple[bool, List[str]]:
        """Gate 2: Risk Tier Validity."""
        clean_risk = risk_str.strip().capitalize()
        if clean_risk not in self.VALID_RISK_TIERS:
            return False, [f"INVALID_RISK: '{risk_str}' is not in allowed tiers {list(self.VALID_RISK_TIERS)}"]
        return True, []

    def check_entity_grounding(
        self,
        source_note: str,
        parsed_fields: Dict[str, str],
        known_entities: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[bool, List[str]]:
        """Gate 3 & Gate 4: Entity Grounding & Hallucination."""
        infractions = []
        note_lower = source_note.lower()
        full_output = f"{parsed_fields.get('Key Finding', '')} {parsed_fields.get('Action', '')}".lower()

        # Check reference entities if supplied
        if known_entities:
            drugs = known_entities.get("ner_drugs", [])
            for drug in drugs:
                d_str = str(drug).lower().strip()
                if d_str and len(d_str) > 2 and d_str not in note_lower and d_str in full_output:
                    infractions.append(f"UNGROUNDED_ENTITY: Drug '{drug}' in output is absent from clinical note")

        # General hallucination check: Common oncology drugs that should not appear unprompted
        hallucination_lexicon = ["doxorubicin", "methotrexate", "bleomycin", "vinblastine", "interleukin-2"]
        for ungrounded_drug in hallucination_lexicon:
            if ungrounded_drug in full_output and ungrounded_drug not in note_lower:
                infractions.append(f"HALLUCINATED_DRUG: Severe hallucination of non-prescribed antineoplastic '{ungrounded_drug}'")

        return len(infractions) == 0, infractions

    def check_negation_polarity(
        self,
        source_note: str,
        parsed_fields: Dict[str, str]
    ) -> Tuple[bool, List[str]]:
        """Gate 5: Negation Polarity Preservation."""
        infractions = []
        note_lower = source_note.lower()
        kf_lower = parsed_fields.get("Key Finding", "").lower()
        risk_str = parsed_fields.get("Risk", "").strip().capitalize()

        # Check if note denies toxicities
        note_negated = any(phrase in note_lower for phrase in [
            "no acute adverse", "denies adverse", "denies toxicities", "denies any", "no toxicities",
            "without acute toxicity", "free of treatment-limiting", "zero adverse symptoms", "no evidence of systemic toxicity"
        ])

        if note_negated:
            # Output must NOT assert high risk or active severe toxicities
            contradictions = ["developed neutropenia", "increased hazard", "developed severe", "grade 3", "grade 4", "fever"]
            for contra in contradictions:
                if contra in kf_lower:
                    infractions.append(f"NEGATION_FLIP: Source note confirms absence of toxicities but model asserts '{contra}'")

            if risk_str == "High" and not any(k in note_lower for k in ["neutropen", "fever", "colitis", "pneumonit"]):
                infractions.append("NEGATION_RISK_CONTRADICTION: Note has no toxicities but model assigned 'High' risk")

        return len(infractions) == 0, infractions

    def check_action_coherence(
        self,
        risk_str: str,
        action_str: str
    ) -> Tuple[bool, List[str]]:
        """Gate 6: Therapeutic Action Coherence."""
        infractions = []
        act_lower = action_str.lower()
        risk = risk_str.strip().capitalize()

        if risk == "High":
            # High risk must not recommend routine monitoring
            if "continue standard clinical monitoring" in act_lower or "continue routine" in act_lower:
                infractions.append("ACTION_INCOHERENCE: High-risk patient assigned passive routine monitoring action")
        elif risk == "Low":
            # Low risk must not recommend emergency discontinuation
            if "admit patient" in act_lower or "emergency" in act_lower or "hold all therapy" in act_lower:
                infractions.append("ACTION_INCOHERENCE: Low-risk patient assigned aggressive emergency action")

        return len(infractions) == 0, infractions

    def validate(
        self,
        source_note: str,
        generation_text: str,
        confidence: float = 1.0,
        reference_entities: Optional[Dict[str, List[str]]] = None
    ) -> SafetyFirewallResult:
        """
        Executes all 6 validation gates.
        Returns SafetyFirewallResult with route: CLINICAL_UI or HUMAN_REVIEW.
        """
        all_infractions = []

        # Gate 1: Schema
        schema_ok, fields, schema_errs = self.parse_schema(generation_text)
        all_infractions.extend(schema_errs)

        if schema_ok:
            # Gate 2: Risk Validity
            risk_ok, risk_errs = self.check_risk_validity(fields["Risk"])
            all_infractions.extend(risk_errs)

            # Gate 3 & 4: Grounding & Hallucination
            ground_ok, ground_errs = self.check_entity_grounding(source_note, fields, reference_entities)
            all_infractions.extend(ground_errs)

            # Gate 5: Negation Polarity
            neg_ok, neg_errs = self.check_negation_polarity(source_note, fields)
            all_infractions.extend(neg_errs)

            # Gate 6: Action Coherence
            if risk_ok:
                act_ok, act_errs = self.check_action_coherence(fields["Risk"], fields["Action"])
                all_infractions.extend(act_errs)

        # Confidence Threshold Check
        if confidence < self.min_confidence_threshold:
            all_infractions.append(f"LOW_CONFIDENCE: Confidence score {confidence:.3f} below minimum {self.min_confidence_threshold:.3f}")

        passed = len(all_infractions) == 0
        route = "CLINICAL_UI" if passed else "HUMAN_REVIEW"

        return SafetyFirewallResult(
            passed=passed,
            route=route,
            infractions=all_infractions,
            parsed_fields=fields,
            confidence=confidence
        )
