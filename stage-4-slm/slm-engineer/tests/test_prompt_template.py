"""
Tests for Clinical Prompt Template and Output Parsing Module.
"""

from prompt_template import ClinicalPromptTemplate


def test_input_prompt_formatting():
    tmpl = ClinicalPromptTemplate()
    prompt = tmpl.format_input_prompt("Patient receiving Osimertinib.")
    assert "Clinical Note:" in prompt
    assert "Patient receiving Osimertinib." in prompt
    assert "Risk: <Low | Moderate | High>" in prompt


def test_target_output_formatting():
    tmpl = ClinicalPromptTemplate()
    out = tmpl.format_target_output(
        target_risk="Low toxicity risk.",
        target_key_finding="EGFR mutation confirmed.",
        target_action="Continue standard monitoring."
    )
    assert out.startswith("Risk: Low")
    assert "Key Finding: EGFR mutation confirmed." in out
    assert "Action: Continue standard monitoring." in out


def test_parse_slm_output_compliant():
    tmpl = ClinicalPromptTemplate()
    sample = "Risk: Moderate\nKey Finding: Trastuzumab administered.\nAction: Monitor closely."
    parsed = tmpl.parse_slm_output(sample)
    assert parsed["is_compliant"] is True
    assert parsed["risk"] == "Moderate"
    assert parsed["key_finding"] == "Trastuzumab administered."
    assert parsed["action"] == "Monitor closely."


def test_parse_slm_output_missing_field():
    tmpl = ClinicalPromptTemplate()
    # Missing Action field
    sample = "Risk: Low\nKey Finding: Stable disease."
    parsed = tmpl.parse_slm_output(sample)
    assert parsed["is_compliant"] is False
    assert "Action" in parsed["missing_fields"]
