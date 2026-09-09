"""
Unit tests for Blinded Clinician Review Workflow and Kappa Agreement.
"""

import tempfile
from pathlib import Path
import pandas as pd
import pytest
from clinician_review import ClinicianReviewInfrastructure


def test_clinician_review_packet_generation_and_agreement():
    with tempfile.TemporaryDirectory() as tmp_dir:
        infra = ClinicianReviewInfrastructure(output_dir=tmp_dir)

        df = pd.DataFrame([
            {"review_id": "REV_001", "prompt": "Note 1", "target": "Risk: Low\nKey Finding: Low finding.\nAction: Action 1", "expected_risk": "Low"},
            {"review_id": "REV_002", "prompt": "Note 2", "target": "Risk: High\nKey Finding: High finding.\nAction: Action 2", "expected_risk": "High"}
        ])
        preds = [
            "Risk: Low\nKey Finding: Low finding.\nAction: Action 1",
            "Risk: High\nKey Finding: High finding.\nAction: Action 2"
        ]

        # Test packet export
        packet_path = infra.generate_blinded_review_packet(df, preds)
        assert Path(packet_path).exists()

        # Test agreement calculation
        rev_a = [
            {"case_id": "REV_001", "risk_category": "Low", "clinical_correctness": 5, "action_appropriateness": 5},
            {"case_id": "REV_002", "risk_category": "High", "clinical_correctness": 5, "action_appropriateness": 5}
        ]
        rev_b = [
            {"case_id": "REV_001", "risk_category": "Low", "clinical_correctness": 4, "action_appropriateness": 4},
            {"case_id": "REV_002", "risk_category": "High", "clinical_correctness": 5, "action_appropriateness": 5}
        ]
        m_ratings = [{"risk_category": "Low"}, {"risk_category": "High"}]

        agreement = infra.compute_inter_rater_agreement(rev_a, rev_b, m_ratings)
        assert agreement["inter_rater_cohens_kappa"] == 1.0
        assert agreement["inter_rater_raw_agreement_rate"] == 1.0
        assert agreement["mean_clinical_correctness_score_1_to_5"] == 4.75
        assert agreement["model_alignment"]["model_vs_human_consensus_kappa"] == 1.0
        assert "PILOT WORKFLOW DEMONSTRATION" in agreement["validation_claim"]
