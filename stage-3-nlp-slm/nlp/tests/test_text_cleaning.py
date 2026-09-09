"""
Test Suite: Clinical Text Cleaning.
Verifies conservative cleaning and strict preservation of dosages, numbers, negations, and units.
"""

import sys
from pathlib import Path
import pytest

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from text_cleaning import clean_clinical_text, is_valid_clinical_text


def test_preservation_of_numbers_and_dosages():
    """Verify numbers, decimals, and dosage units are preserved intact."""
    raw = "Patient received Cisplatin 75.5 mg/m2 IV. Serum creatinine: 1.85 mg/dL. SpO2: 96%."
    cleaned = clean_clinical_text(raw)
    assert "75.5 mg/m2" in cleaned
    assert "1.85 mg/dL" in cleaned
    assert "96%" in cleaned


def test_preservation_of_clinical_negation():
    """Verify negation cues and symptoms are not altered or removed."""
    raw = "Denies acute chest pain. No fever observed. Negative for EGFR T790M."
    cleaned = clean_clinical_text(raw)
    assert "Denies acute chest pain" in cleaned
    assert "No fever" in cleaned
    assert "Negative for EGFR T790M" in cleaned


def test_preservation_of_severity_grades_and_acronyms():
    """Verify severity grades and oncology acronyms remain intact."""
    raw = "Patient with Stage III NSCLC experienced Grade 3 neutropenia. ECOG PS: 1."
    cleaned = clean_clinical_text(raw)
    assert "Grade 3" in cleaned
    assert "NSCLC" in cleaned
    assert "ECOG PS: 1" in cleaned


def test_removal_of_html_and_control_chars():
    """Verify accidental markup and control characters are stripped."""
    raw = "<div>Patient report:\x00\x08<b>Stable</b>.\x0bSpO2\xa098%.</div>"
    cleaned = clean_clinical_text(raw)
    assert "<div>" not in cleaned
    assert "<b>" not in cleaned
    assert "\x00" not in cleaned
    assert "\xa0" not in cleaned
    assert "Stable" in cleaned
    assert "SpO2 98%" in cleaned


def test_is_valid_clinical_text():
    """Verify minimum and maximum word thresholds."""
    short = "Too short note"
    valid = "Patient is a 62-year-old female presenting for cycle 3 chemotherapy with no acute toxicities."
    assert is_valid_clinical_text(short, min_words=15) is False
    assert is_valid_clinical_text(valid, min_words=10) is True
    assert is_valid_clinical_text("", min_words=15) is False
