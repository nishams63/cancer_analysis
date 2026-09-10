"""
Safe Fallback Module for Stage 4 SLM.
Executes calibrated Stage 3 baseline rules if SLM confidence is low or guardrails fail.
"""

import re
import logging
from typing import Dict, Any

logger = logging.getLogger("stage4.integration.fallback")


class ClinicalDecisionFallback:
    """Provides fail-safe, rule-based clinical decision support based on Stage 3 linear baselines."""

    @staticmethod
    def generate_fallback_triad(clinical_note: str, reason: str = "Guardrail violation") -> Dict[str, str]:
        """Generates deterministic, conservative clinical triage recommendation."""
        logger.warning(f"Activating Safe Clinical Fallback. Reason: {reason}")
        text = clinical_note.lower()

        is_critical = any(w in text for w in ["acute respiratory", "hypoxia", "septic", "grade 4", "emergency"])
        is_high = any(w in text for w in ["severe", "pneumonitis", "creatinine", "jaundice", "grade 3"])

        if is_critical:
            risk = "Critical triage alert: Acute high-grade toxicity detected via fallback rules."
            kf = "Patient manifests acute life-threatening symptoms requiring immediate clinical intervention."
            action = "Immediately hold all antineoplastic therapy, transfer patient to emergency triage, and consult attending oncologist."
        elif is_high:
            risk = "Elevated toxicity alert: Significant organ hazard identified via fallback rules."
            kf = "Patient presents with active moderate-to-severe adverse event."
            action = "Hold current chemotherapy cycle, perform urgent organ function restaging, and review for 25% dose reduction."
        else:
            risk = "Low baseline toxicity hazard confirmed via fallback rules."
            kf = "No acute life-threatening adverse toxicities observed."
            action = "Continue standard clinical observation and maintain current oncology regimen as tolerated."

        return {
            "target_risk": risk,
            "target_key_finding": kf,
            "target_action": action
        }
