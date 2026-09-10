"""
Safety Gateway Module for Stage 4 Integration.
Directly integrates the validated 6-gate Stage 6 Clinical Safety Firewall.
Routes: PASS -> Clinical UI, FAIL -> Human Review Queue.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Link to Stage 6 evaluation-engineer source to reuse frozen firewall logic
EVAL_SRC = Path(__file__).resolve().parent.parent.parent / "evaluation-engineer" / "src"
if str(EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(EVAL_SRC))

try:
    from safety_firewall import ClinicalSafetyFirewall, SafetyFirewallResult
except ImportError:
    # Fallback import if directory path differs
    from ...evaluation_engineer.src.safety_firewall import ClinicalSafetyFirewall, SafetyFirewallResult

logger = logging.getLogger("safety_gateway")


class SafetyGateway:
    """Enforces clinical safety gates before returning any inference to downstream users."""

    def __init__(
        self,
        strict_mode: bool = True,
        confidence_threshold: float = 0.500
    ):
        self.firewall = ClinicalSafetyFirewall(
            strict_mode=strict_mode,
            min_confidence_threshold=confidence_threshold
        )
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        clinical_note: str,
        raw_output: str,
        confidence: float,
        reference_entities: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Runs the 6 sequential clinical safety gates:
        1. Schema Validation
        2. Risk Tier Validity
        3. Entity Grounding
        4. Hallucination Detection
        5. Negation Polarity Preservation
        6. Action Coherence
        Plus Calibrated Confidence Threshold Check.
        """
        fw_result = self.firewall.validate(
            source_note=clinical_note,
            generation_text=raw_output,
            confidence=confidence,
            reference_entities=reference_entities
        )

        is_pass = fw_result.passed
        safety_status = "PASS" if is_pass else "REVIEW"
        review_required = not is_pass

        verdict = {
            "safety_status": safety_status,
            "review_required": review_required,
            "passed": is_pass,
            "route": "CLINICAL_UI" if is_pass else "HUMAN_REVIEW",
            "infractions": fw_result.infractions,
            "parsed_fields": fw_result.parsed_fields,
            "confidence": round(confidence, 4),
            "confidence_threshold": self.confidence_threshold
        }

        if not is_pass:
            verdict["failure_reason"] = "; ".join(fw_result.infractions)
            verdict["failed_checks"] = fw_result.infractions
            verdict["review_status"] = "PENDING_CLINICIAN_REVIEW"

        return verdict
