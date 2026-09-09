"""
Test Suite: Clinical Text Normalization.
Verifies Unicode NFKC decomposition, clinical unit standardization, and acronym casing.
"""

import sys
from pathlib import Path
import pytest

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from text_normalization import (
    normalize_clinical_text,
    standardize_clinical_units,
    standardize_abbreviations
)


def test_unicode_normalization():
    """Verify NFKC normalization transforms special characters and ligatures."""
    text = "Stage\xa0III\u200b cancer: \u201cDocetaxel\u201d administered."
    normalized = normalize_clinical_text(text)
    assert "\xa0" not in normalized
    assert "\u200b" not in normalized
    assert '"Docetaxel"' in normalized


def test_clinical_unit_standardization():
    """Verify irregular unit spacing is normalized to standard format."""
    text = "Dose: 75.5mg/m^2. Creatinine: 1.8mg/dl. Blood pressure: 130/85mmHg."
    standardized = standardize_clinical_units(text)
    assert "75.5 mg/m2" in standardized
    assert "1.8 mg/dL" in standardized
    assert "130/85 mmHg" in standardized


def test_abbreviation_casing():
    """Verify gene driver symbols and oncology acronyms are capitalized."""
    text = "Patient diagnosed with nsclc, egfr positive, kras wild-type, ecog ps 1."
    standardized = standardize_abbreviations(text)
    assert "NSCLC" in standardized
    assert "EGFR" in standardized
    assert "KRAS" in standardized
    assert "ECOG" in standardized
