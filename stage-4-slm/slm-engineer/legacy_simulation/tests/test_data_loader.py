"""
Tests for Stage 5 SLM Data Loader and EDA Readiness Gate.
"""

import json
import pandas as pd
import pytest
from data_loader import SLMDataLoader, SLMReadinessBlockError


@pytest.fixture
def dummy_dataset_and_eda(tmp_path):
    # Dummy dataset
    data_p = tmp_path / "dataset.parquet"
    df = pd.DataFrame({
        "patient_id": ["PT-001", "PT-002", "PT-003"],
        "note_id": ["DOC-001", "DOC-002", "DOC-003"],
        "clinical_note": ["Note 1", "Note 2", "Note 3"],
        "instruction": ["Instruction"] * 3,
        "target_risk": ["Low risk."] * 3,
        "target_key_finding": ["Finding."] * 3,
        "target_action": ["Action."] * 3,
        "split": ["TRAIN", "VALIDATION", "TEST"]
    })
    df.to_parquet(data_p, index=False)

    # Dummy EDA results
    eda_p = tmp_path / "eda_results.json"
    return data_p, eda_p, df


def test_eda_gate_blocks_on_not_ready(dummy_dataset_and_eda):
    data_p, eda_p, _ = dummy_dataset_and_eda
    with open(eda_p, "w", encoding="utf-8") as f:
        json.dump({
            "final_readiness": {
                "final_status": "NOT READY",
                "decision_reasons": ["[CRITICAL] negation_flips: Clinical negation flip rate is 14.88%."]
            }
        }, f)

    loader = SLMDataLoader(str(data_p), str(eda_p))
    gate = loader.check_eda_readiness()
    assert gate["is_blocked"] is True

    with pytest.raises(SLMReadinessBlockError, match="SLM training blocked by EDA gate"):
        loader.load_dataset(enforce_gate=True)


def test_eda_gate_passes_on_ready_with_warnings(dummy_dataset_and_eda):
    data_p, eda_p, _ = dummy_dataset_and_eda
    with open(eda_p, "w", encoding="utf-8") as f:
        json.dump({
            "final_readiness": {
                "final_status": "READY WITH WARNINGS",
                "decision_reasons": ["[WARNING] risk_class_imbalance"]
            }
        }, f)

    loader = SLMDataLoader(str(data_p), str(eda_p))
    gate = loader.check_eda_readiness()
    assert gate["is_blocked"] is False

    df, splits = loader.load_dataset(enforce_gate=True)
    assert len(df) == 3
    assert len(splits["train"]) == 1
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1


def test_patient_leakage_rejection(dummy_dataset_and_eda):
    data_p, eda_p, df = dummy_dataset_and_eda
    with open(eda_p, "w", encoding="utf-8") as f:
        json.dump({"final_readiness": {"final_status": "READY", "decision_reasons": []}}, f)

    # Inject leakage: PT-001 in both TRAIN and VALIDATION
    leaky_df = df.copy()
    leaky_df.loc[1, "patient_id"] = "PT-001"
    leaky_df.to_parquet(data_p, index=False)

    loader = SLMDataLoader(str(data_p), str(eda_p))
    with pytest.raises(ValueError, match="Patient leakage detected"):
        loader.load_dataset(enforce_gate=True)
