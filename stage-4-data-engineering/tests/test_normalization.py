"""
Unit Tests for Section 4a Normalization Rules.
Tests case differences, whitespace/unicode, dosage spacing/units/ranges,
and versioned drug synonym lookups.
"""

import pytest
from pathlib import Path
import sys

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
CONFIG_DIR = TESTS_DIR.parent / "config"
sys.path.insert(0, str(SRC_DIR))

from entity_validator import EntityNormalizer


@pytest.fixture
def normalizer():
    synonyms_path = CONFIG_DIR / "drug_synonyms.yaml"
    return EntityNormalizer(drug_synonyms_path=str(synonyms_path))


def test_case_and_unicode_normalization(normalizer):
    """Verify NFKC normalization and case folding."""
    raw = "  EGFR \u00a0 Mutation  "  # Non-breaking space
    cleaned = normalizer.clean_unicode_and_case(raw)
    assert cleaned == "egfr mutation"


def test_dosage_spacing_normalization(normalizer):
    """Verify equivalence between '5 mg' and '5mg'."""
    assert normalizer.match_dosage("5 mg", "Patient took 5mg daily.")
    assert normalizer.match_dosage("5mg", "Patient took 5 mg daily.")
    assert normalizer.match_dosage("265.4 mg", "Docetaxel 265.4mg administered.")


def test_dosage_range_normalization(normalizer):
    """Verify that a single value within a specified range is treated as a match."""
    # 5-10 mg matched by 7 mg
    assert normalizer.match_dosage("5-10 mg", "Target prescribed at 7 mg.")
    assert normalizer.match_dosage("5–10 mg", "Target prescribed at 5 mg.")  # en-dash
    assert normalizer.match_dosage("5-10 mg", "Target prescribed at 10 mg.")
    # Outside range should fail
    assert not normalizer.match_dosage("5-10 mg", "Target prescribed at 15 mg.")


def test_safe_unit_conversion(normalizer):
    """Verify safe unambiguous unit conversion: 5000 mcg <-> 5 mg."""
    assert normalizer.match_dosage("5000 mcg", "Patient administered 5 mg.")
    assert normalizer.match_dosage("5 mg", "Patient administered 5000 mcg.")
    assert normalizer.match_dosage("1000 mg", "Patient received 1 g.")


def test_drug_synonyms_in_table(normalizer):
    """Verify known generic vs brand synonyms from checked-in table match."""
    # Warfarin <-> Coumadin
    assert normalizer.match_drug("Warfarin", "Patient prescribed Coumadin 5 mg.")
    assert normalizer.match_drug("Coumadin", "Prescribed Warfarin therapy.")
    # Erlotinib <-> Tarceva
    assert normalizer.match_drug("Erlotinib", "Patient responded well to Tarceva.")
    # Docetaxel <-> Taxotere
    assert normalizer.match_drug("Docetaxel", "Administered Taxotere cycle 3.")


def test_drug_synonyms_not_in_table(normalizer):
    """Verify unlisted drug synonyms are strictly treated as genuine mismatches."""
    # Unlisted alias should NOT match freely
    assert not normalizer.match_drug("Aspirin", "Patient took Bufferin daily.")
    assert not normalizer.match_drug("Cisplatin", "UnknownBrandX was administered.")
