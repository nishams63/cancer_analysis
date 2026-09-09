"""
Structured Prompt Template and Output Parsing Module for Stage 5 SLM.
Enforces the canonical 3-field response structure (Risk, Key Finding, Action)
and tracks prompt template versions per Sections 9 & 10.
"""

import re
from typing import Dict, Any, Optional

PROMPT_TEMPLATE_NAME = "clinical_decision_support_v1"
PROMPT_TEMPLATE_VERSION = "1.0.0"

SYSTEM_INSTRUCTION = (
    "You are a clinical decision-support summarization assistant.\n"
    "Read the clinical note and produce a structured response.\n\n"
    "Return exactly:\n\n"
    "Risk: <Low | Moderate | High>\n\n"
    "Key Finding: <concise clinically relevant finding>\n\n"
    "Action: <appropriate action/monitoring recommendation>"
)


class ClinicalPromptTemplate:
    """Formats clinical notes and targets into standardized instruction-tuning pairs."""

    def __init__(self, version: str = PROMPT_TEMPLATE_VERSION):
        self.version = version
        self.name = PROMPT_TEMPLATE_NAME

    def format_input_prompt(self, clinical_note: str) -> str:
        """Formats the input prompt presenting the task and the clinical note."""
        clean_note = str(clinical_note).strip()
        return f"{SYSTEM_INSTRUCTION}\n\nClinical Note:\n{clean_note}\n\nResponse:\n"

    def format_target_output(self, target_risk: str, target_key_finding: str, target_action: str) -> str:
        """Formats the expected completion output containing the 3 primary clinical fields."""
        r_str = str(target_risk).strip()
        kf_str = str(target_key_finding).strip()
        a_str = str(target_action).strip()

        # Simplify risk to canonical tier if full description was provided
        r_tier = self._extract_risk_tier(r_str)

        return f"Risk: {r_tier}\nKey Finding: {kf_str}\nAction: {a_str}"

    @staticmethod
    def _extract_risk_tier(risk_text: str) -> str:
        """Extracts Low, Moderate, or High from risk text."""
        r_lower = risk_text.lower()
        if "moderate" in r_lower:
            return "Moderate"
        elif any(k in r_lower for k in ["high", "severe", "grade 3", "grade 4", "hold or reduce"]):
            return "High"
        else:
            return "Low"

    def parse_slm_output(self, output_text: str) -> Dict[str, Any]:
        """
        Parses raw SLM generation output into structured fields:
        Returns: {
            'risk': str or None,
            'key_finding': str or None,
            'action': str or None,
            'is_compliant': bool,
            'missing_fields': list
        }
        """
        text = str(output_text).strip()
        risk_match = re.search(r"Risk:\s*([^\n]+)", text, re.IGNORECASE)
        kf_match = re.search(r"Key Finding:\s*([^\n]+(?:\n(?!Action:)[^\n]+)*)", text, re.IGNORECASE)
        action_match = re.search(r"Action:\s*([^\n]+(?:\n[^\n]+)*)", text, re.IGNORECASE)

        risk_val = risk_match.group(1).strip() if risk_match else None
        kf_val = kf_match.group(1).strip() if kf_match else None
        act_val = action_match.group(1).strip() if action_match else None

        missing = []
        if not risk_val:
            missing.append("Risk")
        if not kf_val:
            missing.append("Key Finding")
        if not act_val:
            missing.append("Action")

        is_compliant = len(missing) == 0

        return {
            "risk": risk_val,
            "key_finding": kf_val,
            "action": act_val,
            "is_compliant": is_compliant,
            "missing_fields": missing,
            "raw_text": text
        }
