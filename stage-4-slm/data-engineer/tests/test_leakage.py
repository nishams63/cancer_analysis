"""
Unit Tests for Section 10 Leakage Audit.
Tests that cross-split patient overlap fails, near-duplicates are flagged,
and clean splits pass with patient_leakage = 0.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from leakage_audit import LeakageAuditor, LeakageAuditError


def test_clean_split_leakage_audit():
    """Verify clean dataset passes leakage audit with zero patient leakage."""
    df_clean = pd.DataFrame([
        {"patient_id": "PT-001", "clinical_note": "Patient PT-001 note on docetaxel.", "split": "TRAIN"},
        {"patient_id": "PT-002", "clinical_note": "Patient PT-002 note on cisplatin.", "split": "VALIDATION"},
        {"patient_id": "PT-003", "clinical_note": "Patient PT-003 note on erlotinib.", "split": "TEST"}
    ])
    auditor = LeakageAuditor()
    report = auditor.run_full_audit(df_clean)
    assert report["overall_audit_status"] == "PASSED"
    assert report["patient_leakage"] == 0


def test_injected_patient_leakage_fails():
    """Verify injected cross-split patient overlap raises LeakageAuditError."""
    # PT-001 appears in both TRAIN and TEST
    df_leaking = pd.DataFrame([
        {"patient_id": "PT-001", "clinical_note": "Patient note on docetaxel.", "split": "TRAIN"},
        {"patient_id": "PT-002", "clinical_note": "Patient note on cisplatin.", "split": "VALIDATION"},
        {"patient_id": "PT-001", "clinical_note": "Follow-up note for PT-001.", "split": "TEST"}
    ])
    auditor = LeakageAuditor()
    with pytest.raises(LeakageAuditError) as exc_info:
        auditor.audit_patient_isolation(df_leaking)
    assert "CRITICAL PATIENT LEAKAGE DETECTED" in str(exc_info.value)


def test_cross_split_exact_duplicate_fails():
    """Verify exact identical text across split boundaries is detected."""
    df_dup = pd.DataFrame([
        {"patient_id": "PT-001", "clinical_note": "Identical narrative text verbatim.", "split": "TRAIN"},
        {"patient_id": "PT-002", "clinical_note": "Identical narrative text verbatim.", "split": "VALIDATION"}
    ])
    auditor = LeakageAuditor()
    res = auditor.audit_exact_duplicates(df_dup)
    assert res["status"] == "FAILED"
    assert res["cross_split_exact_duplicates"] == 1


def test_near_duplicate_detection():
    """Verify high-similarity near-duplicates cross-split are caught."""
    df_near = pd.DataFrame([
        {"patient_id": "PT-001", "clinical_note": "Patient presents with severe acute dyspnea and cough following cycle 2.", "split": "TRAIN"},
        {"patient_id": "PT-002", "clinical_note": "Patient presents with severe acute dyspnea and cough following cycle 3.", "split": "VALIDATION"},
        {"patient_id": "PT-003", "clinical_note": "Completely different text without any shared medical terms.", "split": "TEST"}
    ])
    auditor = LeakageAuditor(similarity_threshold=0.80)
    res = auditor.audit_near_duplicates(df_near)
    assert res["near_duplicate_cross_matches"] >= 1
