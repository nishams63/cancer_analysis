"""
Unit tests for leakage checks, cryptographic hashes, temporal order, and target leakage terms.
"""

from data_loader import load_validation_data, load_locked_test_data
from leakage_checks import (
    verify_upstream_file_hashes,
    verify_temporal_consistency,
    verify_target_leakage_terms
)


def test_cryptographic_file_hashes():
    res = verify_upstream_file_hashes()
    assert res["all_files_invariant"] is True
    for fname, details in res["details"].items():
        assert details["matched"] is True, f"Hash mismatch for {fname}"


def test_temporal_consistency_validation_and_test():
    df_val = load_validation_data()
    df_test = load_locked_test_data()

    res_val = verify_temporal_consistency(df_val)
    res_test = verify_temporal_consistency(df_test)

    assert res_val["temporal_ordering_valid"] is True
    assert res_val["future_dated_inversions"] == 0
    assert res_test["temporal_ordering_valid"] is True
    assert res_test["future_dated_inversions"] == 0


def test_target_leakage_scan():
    df_val = load_validation_data()
    df_test = load_locked_test_data()

    res_val = verify_target_leakage_terms(df_val)
    res_test = verify_target_leakage_terms(df_test)

    assert res_val["target_leakage_free"] is True
    assert res_test["target_leakage_free"] is True
