"""
Test Suite: Text Preprocessing & Privacy Sanitization.
Verifies that text normalization preserves clinical semantics, dosages, and negations,
and that PII scrubbers properly mask direct identifiers.
"""

import sys
from pathlib import Path
import pytest

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from text_preprocessing import normalize_clinical_text, compute_text_metrics
from de_identification import scrub_pii, verify_no_direct_identifiers

def test_unicode_and_whitespace_normalization():
    """Verify Unicode NFKC normalization and whitespace cleanup."""
    raw_text = "Patient presenting with  severe  fatigue.\n\n\n\nBP: 120/80\xa0mmHg. “Erlotinib” prescribed."
    cleaned = normalize_clinical_text(raw_text)
    assert "\xa0" not in cleaned
    assert "“" not in cleaned
    assert '"Erlotinib"' in cleaned
    assert "\n\n\n" not in cleaned

def test_negation_retention():
    """Verify that clinical negation expressions are preserved."""
    clinical_note = "Patient denies chest pain. No fever observed at home. Negative for EGFR T790M mutation."
    cleaned = normalize_clinical_text(clinical_note)
    metrics = compute_text_metrics(cleaned)
    assert "denies chest pain" in cleaned
    assert "no fever" in cleaned.lower()
    assert "negative for" in cleaned.lower()
    assert metrics["has_negation"] is True

def test_dosage_and_numerical_retention():
    """Verify that dosages, decimals, and clinical units are preserved intact."""
    text_with_doses = "Administer Cisplatin 75.5 mg/m2 IV. Serum creatinine is 1.85 mg/dL. SpO2 is 96%."
    cleaned = normalize_clinical_text(text_with_doses)
    metrics = compute_text_metrics(cleaned)
    assert "75.5 mg/m2" in cleaned
    assert "1.85 mg/dL" in cleaned
    assert "96%" in cleaned
    assert metrics["has_dosage"] is True

def test_html_tag_stripping():
    """Verify that accidental HTML tags are removed without affecting text."""
    dirty_text = "<p>Patient demonstrates <b>acute nephrotoxicity</b>.</p>"
    cleaned = normalize_clinical_text(dirty_text)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "acute nephrotoxicity" in cleaned

def test_pii_masking():
    """Verify that direct identifiers are sanitized into privacy tokens."""
    text_with_pii = (
        "CONFIDENTIAL RECORD\n"
        "Patient Name: Johnathan Doe | Phone: (555) 345-6789\n"
        "Email: j.doe@hospital-clinic.org | MRN: 9876543\n"
        "Diagnosed with Stage III NSCLC."
    )
    scrubbed, counts = scrub_pii(text_with_pii)
    assert "[NAME]" in scrubbed
    assert "[PHONE]" in scrubbed
    assert "[EMAIL]" in scrubbed
    assert "[MRN]" in scrubbed
    assert "Johnathan Doe" not in scrubbed
    assert "555" not in scrubbed
    assert counts["patient_name"] >= 1
    assert counts["phone"] >= 1
    assert counts["email"] >= 1
    assert counts["mrn"] >= 1

    violations = verify_no_direct_identifiers(scrubbed)
    assert len(violations) == 0, f"Unmasked violations remain: {violations}"
