"""
Tests verifying prompt construction and clinical output schema parser.
"""

from src.tokenizer import load_tokenizer
from src.prompts import build_clinical_prompt, parse_clinical_output


def test_build_clinical_prompt():
    """Verifies that prompt builder generates official ChatML structure."""
    tok = load_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
    prompt = build_clinical_prompt(
        tok,
        clinical_note="Patient with NSCLC on osimertinib 80mg daily."
    )
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "osimertinib 80mg daily" in prompt
    assert "<|im_start|>assistant\n" in prompt


def test_parse_compliant_output():
    """Verifies that correctly structured generation is parsed into fields."""
    raw = (
        "Risk: Low\n"
        "Key Finding: Patient on Osimertinib 80mg tolerating therapy with no acute toxicities.\n"
        "Action: Continue current regimen and schedule cycle follow-up.<|im_end|>"
    )
    parsed = parse_clinical_output(raw)
    assert parsed["is_valid_format"] is True
    assert parsed["normalized_risk"] == "Low"
    assert "Osimertinib 80mg" in parsed["key_finding"]
    assert "Continue current regimen" in parsed["action"]
    assert len(parsed["missing_fields"]) == 0


def test_parse_malformed_output():
    """Verifies that malformed or missing fields are caught honestly."""
    # Missing Action field
    raw = "Risk: High\nKey Finding: Severe pneumonitis detected."
    parsed = parse_clinical_output(raw)
    assert parsed["is_valid_format"] is False
    assert "action" in parsed["missing_fields"]

    # Invalid risk tier
    raw_invalid_risk = (
        "Risk: CriticalEmergency\n"
        "Key Finding: Adverse reaction.\n"
        "Action: Stop drug immediately."
    )
    parsed2 = parse_clinical_output(raw_invalid_risk)
    assert parsed2["is_valid_format"] is False
    assert parsed2["normalized_risk"] == "Invalid"
