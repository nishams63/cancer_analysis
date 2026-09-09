"""
Test Suite: Clinical Negation & Context Attribution.
Verifies rule-based classification of clinical concepts into AFFIRMED, NEGATED, HISTORICAL, and RESOLVED.
"""

import sys
from pathlib import Path
import pytest

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from negation_detection import resolve_concept_polarity, detect_negations_in_sentence


def test_affirmative_concept():
    """Verify that symptoms without negation cues are tagged AFFIRMED."""
    sent = "Patient reports mild nausea and fatigue."
    start = sent.find("nausea")
    end = start + len("nausea")
    assert resolve_concept_polarity(sent, start, end) == "AFFIRMED"


def test_pre_negation_no_and_denies():
    """Verify 'no' and 'denies' properly negate downstream concepts."""
    sent1 = "Patient presents with no fever or chills."
    s1 = sent1.find("fever")
    assert resolve_concept_polarity(sent1, s1, s1 + len("fever")) == "NEGATED"

    sent2 = "Patient denies dyspnea on exertion."
    s2 = sent2.find("dyspnea")
    assert resolve_concept_polarity(sent2, s2, s2 + len("dyspnea")) == "NEGATED"


def test_historical_context():
    """Verify 'history of' tags concepts as HISTORICAL rather than active acute symptoms."""
    sent = "Past medical history of neutropenia during cycle 1."
    s = sent.find("neutropenia")
    assert resolve_concept_polarity(sent, s, s + len("neutropenia")) == "HISTORICAL"


def test_resolved_post_negation():
    """Verify 'resolved' tags concepts as RESOLVED."""
    sent = "Skin rash has completely resolved."
    s = sent.find("rash")
    assert resolve_concept_polarity(sent, s, s + len("rash")) == "RESOLVED"


def test_pseudo_negation_no_change():
    """Verify pseudo-negation 'no change' does NOT negate the symptom."""
    sent = "There is no change in fatigue levels."
    s = sent.find("fatigue")
    assert resolve_concept_polarity(sent, s, s + len("fatigue")) == "AFFIRMED"


def test_scope_termination_boundary():
    """Verify conjunctions like 'but' terminate negation scope."""
    sent = "No nausea, but patient reports severe fatigue."
    s_nausea = sent.find("nausea")
    s_fatigue = sent.find("fatigue")

    assert resolve_concept_polarity(sent, s_nausea, s_nausea + len("nausea")) == "NEGATED"
    assert resolve_concept_polarity(sent, s_fatigue, s_fatigue + len("fatigue")) == "AFFIRMED"
