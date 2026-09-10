"""
Tests for Split Verification and Leakage Audit Module.
"""

import pandas as pd
import pytest
from split_analysis import SplitAndLeakageAuditor


@pytest.fixture
def clean_split_df():
    return pd.DataFrame({
        "patient_id": ["PT-001", "PT-002", "PT-003", "PT-004"],
        "note_id": ["DOC-001", "DOC-002", "DOC-003", "DOC-004"],
        "clinical_note": [
            "Clinical progress note for patient 1 on cisplatin.",
            "Progress note for patient 2 on carboplatin.",
            "Progress note for patient 3 on docetaxel.",
            "Progress note for patient 4 on pembrolizumab."
        ],
        "target_risk": ["Low risk."] * 4,
        "target_key_finding": ["Finding."] * 4,
        "target_action": ["Action."] * 4,
        "split": ["TRAIN", "TRAIN", "VALIDATION", "TEST"]
    })


def test_clean_patient_isolation(clean_split_df):
    auditor = SplitAndLeakageAuditor()
    res = auditor.verify_patient_isolation(clean_split_df)
    assert res["is_patient_isolated"] is True
    assert res["total_patient_leakage"] == 0


def test_injected_patient_leakage(clean_split_df):
    # Inject patient PT-001 into VALIDATION
    leaky_df = clean_split_df.copy()
    leaky_df.loc[2, "patient_id"] = "PT-001"

    auditor = SplitAndLeakageAuditor()
    res = auditor.verify_patient_isolation(leaky_df)
    assert res["is_patient_isolated"] is False
    assert res["total_patient_leakage"] == 1


def test_metadata_id_leakage(clean_split_df):
    auditor = SplitAndLeakageAuditor()
    # Clean check
    res_clean = auditor.audit_metadata_and_target_leakage(clean_split_df)
    assert res_clean["metadata_id_leakage_count"] == 0

    # Inject note_id leakage into target_action
    leaky_df = clean_split_df.copy()
    leaky_df.loc[0, "target_action"] = "Patient requires observation per DOC-001."
    res_leaky = auditor.audit_metadata_and_target_leakage(leaky_df)
    assert res_leaky["metadata_id_leakage_count"] == 1
