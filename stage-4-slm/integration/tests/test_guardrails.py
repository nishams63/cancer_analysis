"""
Unit Tests for Stage 4 Clinical Safety Guardrails and Fallback.
Tests entity omission detection, hallucination blocking, contradiction trapping,
and safe degradation to Stage 3 baseline decision logic.
"""

import sys
from pathlib import Path
import pytest

INTEG_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(INTEG_SRC_DIR))

from guardrails import ClinicalSafetyGuardrails
from fallback import ClinicalDecisionFallback


def test_guardrail_clean_note_passes():
    """Verify compliant generation passes guardrails with zero violations."""
    guardrails = ClinicalSafetyGuardrails()
    note = "Patient with NSCLC receiving Erlotinib at 150 mg. No acute toxicities."
    triad = {
        "target_risk": "Low risk with Erlotinib 150 mg.",
        "target_key_finding": "EGFR confirmed; Erlotinib 150 mg administered.",
        "target_action": "Continue standard clinical monitoring."
    }
    passed, warnings, preserved = guardrails.audit(note, triad)
    assert passed is True
    assert len(warnings) == 0
    assert "Erlotinib" in preserved["drugs"]


def test_guardrail_blocks_hallucinated_drug():
    """Verify invented antineoplastic drug not present in note is blocked."""
    guardrails = ClinicalSafetyGuardrails()
    note = "Patient with breast cancer receiving Paclitaxel 80 mg/m2."
    triad = {
        "target_risk": "High risk with Cisplatin therapy.",
        "target_key_finding": "Patient administered Cisplatin.",
        "target_action": "Monitor renal panel."
    }
    passed, warnings, preserved = guardrails.audit(note, triad)
    assert passed is False
    assert any("Hallucinated" in w for w in warnings)


def test_guardrail_catches_contradiction():
    """Verify acute life-threatening toxicity contradicts maintenance action."""
    guardrails = ClinicalSafetyGuardrails()
    note = "Patient presenting with acute respiratory distress, hypoxia, and SpO2 < 90 on Docetaxel 100 mg."
    triad = {
        "target_risk": "Critical pulmonary toxicity on Docetaxel.",
        "target_key_finding": "Severe acute respiratory distress observed.",
        "target_action": "Continue standard chemotherapy and maintain current regimen."
    }
    passed, warnings, preserved = guardrails.audit(note, triad)
    assert passed is False
    assert any("Contradiction" in w for w in warnings)


def test_fallback_generates_safe_triad():
    """Verify fallback produces conservative hold actions on critical notes."""
    crit_note = "Patient has acute respiratory distress and severe hypoxia."
    triad = ClinicalDecisionFallback.generate_fallback_triad(crit_note, reason="Guardrail failed")

    assert "Critical" in triad["target_risk"]
    assert "hold all antineoplastic therapy" in triad["target_action"].lower()

    stable_note = "Patient tolerating oral therapy well with clear lungs and normal vitals."
    stable_triad = ClinicalDecisionFallback.generate_fallback_triad(stable_note, reason="Testing")
    assert "Continue standard" in stable_triad["target_action"]
