"""
Unit tests for Sanity & Data Leakage Audit Module.
"""

import tempfile
from pathlib import Path
import pandas as pd
import pytest
from leakage_audit import SanityLeakageAuditor


def test_leakage_audit_clean_data():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        dataset_file = tmp_path / "mock_dataset.parquet"

        # Mock dataset with strict patient isolation
        data = [
            {"patient_id": "P001", "split": "train", "prompt": "Train note P001", "target": "Risk: Low", "target_risk_category": "Low"},
            {"patient_id": "P002", "split": "train", "prompt": "Train note P002", "target": "Risk: High", "target_risk_category": "High"},
            {"patient_id": "P003", "split": "val", "prompt": "Val note P003", "target": "Risk: Low", "target_risk_category": "Low"},
            {"patient_id": "P004", "split": "test", "prompt": "Test note P004", "target": "Risk: Moderate", "target_risk_category": "Moderate"}
        ]
        df = pd.DataFrame(data)
        df.to_parquet(dataset_file, index=False)

        auditor = SanityLeakageAuditor(dataset_path=str(dataset_file), reports_dir=tmp_dir)

        results = auditor.run_all_audits(
            standard_test_metrics={"risk_macro_f1": 1.0},
            ood_synthetic_metrics={"risk_macro_f1": 0.90},
            ood_real_metrics={"risk_macro_f1": 0.88},
            adversarial_metrics={"overall_risk_accuracy": 0.91}
        )

        assert results["test_1_patient_overlap"]["passed"] is True
        assert results["test_2_prompt_target_leakage"]["passed"] is True
        assert results["test_3_cross_split_duplicates"]["passed"] is True
        assert results["test_4_score_ceiling_diagnosis"]["realistic_performance_drop_observed"] is True
        assert "PASSED" in results["final_audit_verdict"]
        assert (tmp_path / "sanity_leakage_audit_report.md").exists()
