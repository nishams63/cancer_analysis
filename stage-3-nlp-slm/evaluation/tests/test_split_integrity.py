"""
Unit tests verifying strict patient, encounter, and document split isolation.
"""

from data_loader import load_train_data, load_validation_data, load_locked_test_data
from leakage_checks import verify_split_isolation


def test_zero_leakage_across_all_splits():
    df_train = load_train_data()
    df_val = load_validation_data()
    df_test = load_locked_test_data()

    res = verify_split_isolation(df_train, df_val, df_test)

    assert res["is_completely_isolated"] is True
    assert res["patient_overlap"]["train_val"] == 0
    assert res["patient_overlap"]["train_test"] == 0
    assert res["patient_overlap"]["val_test"] == 0
    assert res["encounter_overlap"]["train_val"] == 0
    assert res["encounter_overlap"]["train_test"] == 0
    assert res["encounter_overlap"]["val_test"] == 0
    assert res["document_overlap"]["train_val"] == 0
    assert res["document_overlap"]["train_test"] == 0
    assert res["document_overlap"]["val_test"] == 0


def test_split_patient_counts():
    df_train = load_train_data()
    df_val = load_validation_data()
    df_test = load_locked_test_data()

    assert df_train["patient_id"].nunique() == 700
    assert df_val["patient_id"].nunique() == 150
    assert df_test["patient_id"].nunique() == 150
