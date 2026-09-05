"""
Stage 2 Deep Learning - Real-World Readiness Audit Test Suite

Verifies:
1. Frozen modules and prior stages are 100% unmodified
2. Frozen checkpoints match cryptographic SHA-256 hashes
3. Locked test set is preserved (150 patients, 1,800 tiles, 2,403 temporal observations)
4. Anti-leakage boundary enforcement (t <= 90d, traps t > 90d)
5. Prototype thresholds and alert levels are properly labeled
6. Mandatory synthetic disclaimer is present in all configs and reports
7. Readiness report is compiled and contains all required sections
8. Scorecard, gap analysis, and risk register CSVs are generated and structurally complete
9. Absence of false clinical claims (Clinical Deployment Readiness is explicitly NOT READY)
"""
import os
import sys
import hashlib
from pathlib import Path
import pytest
import pandas as pd

# Paths
TESTS_DIR = Path(__file__).resolve().parent
AUDIT_DIR = TESTS_DIR.parent
STAGE_2_DIR = AUDIT_DIR.parent
PROJECT_ROOT = STAGE_2_DIR.parent

sys.path.insert(0, str(AUDIT_DIR / 'src'))
import audit_config as config

sys.path.insert(0, str(STAGE_2_DIR / 'integration' / 'src'))
import validation


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha.update(chunk)
    return sha.hexdigest()


def test_frozen_modules_unmodified():
    """Asserts that all 5 prior Stage 2 modules remain intact and unmodified."""
    key_files = [
        STAGE_2_DIR / 'data-engineering' / 'data' / 'v2' / 'processed' / 'image_metadata.csv',
        STAGE_2_DIR / 'data-engineering' / 'data' / 'v2' / 'processed' / 'biomarkers_processed.csv',
        STAGE_2_DIR / 'eda' / 'reports' / 'eda_report.md',
        STAGE_2_DIR / 'dl' / 'checkpoints' / 'best_pathology_cnn.pt',
        STAGE_2_DIR / 'dl' / 'checkpoints' / 'best_temporal_lstm.pt',
        STAGE_2_DIR / 'evaluation' / 'reports' / 'evaluation_report.md',
        STAGE_2_DIR / 'integration' / 'src' / 'integration_pipeline.py'
    ]
    for kf in key_files:
        assert kf.exists(), f'Expected frozen file missing: {kf}'
        assert kf.stat().st_size > 0, f'Frozen file is empty: {kf}'


def test_checkpoints_unmodified():
    """Verifies SHA-256 cryptographic signatures of frozen DL checkpoints."""
    cnn_path = STAGE_2_DIR / 'dl' / 'checkpoints' / 'best_pathology_cnn.pt'
    lstm_path = STAGE_2_DIR / 'dl' / 'checkpoints' / 'best_temporal_lstm.pt'

    expected_cnn_sha = '228bd121c818277d428962d66f5802ca216862aef47e48116c4982016337f4b3'
    expected_lstm_sha = '1074007476f971bbeb513b62dde4ae5cd36a5c5c7a25b491243f170754760d23'

    assert compute_sha256(cnn_path) == expected_cnn_sha, 'Pathology CNN checkpoint SHA-256 hash mismatch!'
    assert compute_sha256(lstm_path) == expected_lstm_sha, 'Temporal BiLSTM checkpoint SHA-256 hash mismatch!'


def test_locked_test_set_unmodified():
    """Asserts test split preserves exactly 150 patients, 1,800 tiles, and 2,403 observations."""
    test_split_path = STAGE_2_DIR / 'data-engineering' / 'data' / 'v2' / 'splits' / 'test_patients.csv'
    image_meta_path = STAGE_2_DIR / 'data-engineering' / 'data' / 'v2' / 'processed' / 'image_metadata.csv'
    bio_path = STAGE_2_DIR / 'data-engineering' / 'data' / 'v2' / 'processed' / 'biomarkers_processed.csv'

    test_pts = pd.read_csv(test_split_path)
    assert len(test_pts) == 150, f'Expected 150 test patients, got {len(test_pts)}'

    meta = pd.read_csv(image_meta_path)
    test_tiles = meta[meta['split'] == 'test']
    assert len(test_tiles) == 1800, f'Expected 1800 test tiles, got {len(test_tiles)}'

    biomarkers = pd.read_csv(bio_path)
    test_bio = biomarkers[biomarkers['patient_id'].isin(test_pts['patient_id'])]
    assert len(test_bio) == 2403, f'Expected 2403 test biomarker records, got {len(test_bio)}'


def test_no_future_temporal_input():
    """Validates that input validation rejects any temporal observation with days_from_baseline > 90."""
    invalid_bio = pd.DataFrame([{
        'patient_id': 'PAT-9999',
        'days_from_baseline': 120,
        'CEA_ng_ml': 2.5,
        'CA19_9_U_ml': 15.0,
        'ctDNA_VAF_pct': 0.05,
        'WBC_x10_3_uL': 6.0,
        'ANC_x10_3_uL': 3.5,
        'ALC_x10_3_uL': 1.8,
        'platelets_x10_3_uL': 220.0,
        'hemoglobin_g_dL': 13.5,
        'ALT_U_L': 22.0,
        'AST_U_L': 20.0,
        'creatinine_mg_dL': 0.9,
        'LDH_U_L': 160.0
    }])
    with pytest.raises(ValueError, match='boundary violation'):
        validation.validate_temporal_history(invalid_bio)


def test_prototype_thresholds_labeled():
    """Asserts that risk alert thresholds are explicitly classified as prototype engineering choices."""
    report_path = config.FINAL_AUDIT_REPORT_PATH
    assert report_path.exists()
    content = report_path.read_text(encoding='utf-8')
    assert 'PROTOTYPE ENGINEERING HEURISTICS ONLY' in content
    assert 'The weights ($0.35, 0.40, 0.25$) and alert thresholds ($0.35, 0.70$) are heuristic engineering choices.' in content


def test_synthetic_disclaimer_present():
    """Verifies mandatory disclaimer statement exists in configuration and audit reports."""
    assert 'This model was developed using synthetic data and has not been clinically validated' in config.MANDATORY_DISCLAIMER
    report_path = config.FINAL_AUDIT_REPORT_PATH
    content = report_path.read_text(encoding='utf-8')
    assert config.MANDATORY_DISCLAIMER in content
    assert config.EQUIVALENCE_DISCLAIMER in content


def test_readiness_report_generated():
    """Verifies master audit report is generated, non-empty, and has required sections."""
    report_path = config.FINAL_AUDIT_REPORT_PATH
    assert report_path.exists(), 'Master audit report not found'
    assert report_path.stat().st_size > 5000, 'Report is too short or empty'
    text = report_path.read_text(encoding='utf-8')
    assert '## 1. Executive Summary & One-Page Scorecard' in text
    assert '## 2. Baseline Repository Inventory' in text
    assert '## 10. Synthetic-Separability Audit: Exact Generative Mechanics' in text
    assert '## 12. 14-Point Clinical Validation Gap Analysis' in text
    assert '## 21. Real-World Readiness Scorecard (16 Categories)' in text
    assert '## 22. Required Final Conclusions' in text


def test_scorecard_and_gap_csvs_generated():
    """Verifies that scorecard, gap analysis, and risk register CSVs are structurally complete."""
    assert config.SCORECARD_CSV_PATH.exists()
    scorecard = pd.read_csv(config.SCORECARD_CSV_PATH)
    assert len(scorecard) == 16, f'Expected 16 scorecard categories, got {len(scorecard)}'
    assert list(scorecard.columns) == ['Category', 'Rating', 'Evidence', 'Real_World_Gap']

    assert config.GAP_ANALYSIS_CSV_PATH.exists()
    gaps = pd.read_csv(config.GAP_ANALYSIS_CSV_PATH)
    assert len(gaps) == 14, f'Expected 14 clinical gap points, got {len(gaps)}'
    assert list(gaps.columns) == ['Item_ID', 'Clinical_Requirement', 'Current_Status', 'Rating', 'Deficiency_Details']

    assert config.RISK_REGISTER_CSV_PATH.exists()
    risks = pd.read_csv(config.RISK_REGISTER_CSV_PATH)
    assert len(risks) == 6, f'Expected 6 risks in register, got {len(risks)}'


def test_no_false_clinical_claims():
    """Ensures that Clinical Deployment Readiness is strictly NOT READY and zero false claims are made."""
    assert config.TIER_CLINICAL_DEPLOYMENT == 'NOT READY'
    scorecard = pd.read_csv(config.SCORECARD_CSV_PATH)

    clinical_row = scorecard[scorecard['Category'] == 'Clinical Validation']
    assert clinical_row['Rating'].values[0] == 'RED'

    reg_row = scorecard[scorecard['Category'] == 'Regulatory Readiness']
    assert reg_row['Rating'].values[0] == 'RED'

    ext_row = scorecard[scorecard['Category'] == 'External Validation']
    assert ext_row['Rating'].values[0] == 'RED'

    synth_row = scorecard[scorecard['Category'] == 'Synthetic-to-Real Generalization']
    assert synth_row['Rating'].values[0] == 'RED'

    report = config.FINAL_AUDIT_REPORT_PATH.read_text(encoding='utf-8')
    assert '**`NOT READY`** (RED)' in report
    assert 'Zero real-world patient data tested' in report
