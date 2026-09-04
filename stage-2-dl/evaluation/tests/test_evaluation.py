"""
Stage 2 Deep Learning - Evaluation Test Suite
Verifies data integrity, zero-leakage, checkpoint consistency, metric ranges, and disclaimers.
"""
import os
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import torch

TEST_DIR = Path(__file__).resolve().parent
EVAL_DIR = TEST_DIR.parent
SRC_DIR = EVAL_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

import config


def test_locked_test_cohort_size():
    """Verifies the locked test cohort dimensions."""
    test_pts = pd.read_csv(config.TEST_SPLIT_PATH)
    assert len(test_pts) == 150, f"Expected 150 test patients, got {len(test_pts)}"
    assert test_pts['patient_id'].nunique() == 150

    img_df = pd.read_csv(config.IMAGE_METADATA_PATH)
    test_img = img_df[img_df['split'] == 'test']
    assert len(test_img) == 1800, f"Expected 1,800 test image tiles, got {len(test_img)}"
    assert test_img['class_label'].value_counts().to_dict() == {'benign': 600, 'malignant': 600, 'inflammation': 600}

    bio_df = pd.read_csv(config.BIOMARKERS_PATH)
    test_bio = bio_df[bio_df['split'] == 'test']
    assert len(test_bio) == 2403, f"Expected 2,403 test biomarker rows, got {len(test_bio)}"


def test_zero_patient_leakage():
    """Asserts zero overlap between train, validation, and test patient cohorts."""
    train_pts = set(pd.read_csv(config.TRAIN_SPLIT_PATH)['patient_id'])
    val_pts = set(pd.read_csv(config.VAL_SPLIT_PATH)['patient_id'])
    test_pts = set(pd.read_csv(config.TEST_SPLIT_PATH)['patient_id'])

    assert len(train_pts & test_pts) == 0, "Train and Test share patient IDs!"
    assert len(val_pts & test_pts) == 0, "Validation and Test share patient IDs!"
    assert len(train_pts & val_pts) == 0, "Train and Validation share patient IDs!"
    assert len(train_pts | val_pts | test_pts) == 1000, "Cohort does not sum to 1,000 patients!"


def test_temporal_historical_window_strictness():
    """Verifies that historical inputs contain zero observations past Day 90."""
    bio_df = pd.read_csv(config.BIOMARKERS_PATH)
    test_bio = bio_df[bio_df['split'] == 'test']

    # Boundary integrity
    leaks = test_bio[(test_bio['is_input_window'] == 1) & (test_bio['days_from_baseline'] > config.FORECAST_SPLIT_DAY)]
    assert len(leaks) == 0, f"Detected {len(leaks)} historical inputs exceeding Day 90!"

    leaks_inv = test_bio[(test_bio['is_input_window'] == 0) & (test_bio['days_from_baseline'] <= config.FORECAST_SPLIT_DAY)]
    assert len(leaks_inv) == 0, f"Detected {len(leaks_inv)} future inputs tagged on or before Day 90!"


def test_ctdna_velocity_backward_looking():
    """Verifies ctDNA velocity calculation is strictly backward-looking."""
    bio_df = pd.read_csv(config.BIOMARKERS_PATH)
    test_bio = bio_df[bio_df['split'] == 'test']

    for pat_id, grp in list(test_bio.groupby('patient_id'))[:10]:
        grp = grp.sort_values('days_from_baseline')
        vafs = grp['ctDNA_vaf_percent'].values
        days = grp['days_from_baseline'].values
        vels = grp['ctDNA_velocity_30d'].values
        assert vels[0] == 0.0, f"Baseline velocity at t=0 should be 0.0, got {vels[0]}"
        for i in range(1, len(grp)):
            dt = days[i] - days[i-1]
            if not np.isnan(vafs[i]) and not np.isnan(vafs[i-1]) and dt > 0:
                expected_vel = round(((vafs[i] - vafs[i-1]) / dt) * 30.0, 4)
                assert abs(expected_vel - vels[i]) < 1e-3


def test_frozen_checkpoints_exist_and_loadable():
    """Ensures checkpoints are present and readable."""
    assert config.FROZEN_IMAGE_CHECKPOINT.exists(), "Pathology CNN checkpoint missing!"
    assert config.FROZEN_TEMPORAL_CHECKPOINT.exists(), "Temporal BiLSTM checkpoint missing!"

    img_ckpt = torch.load(config.FROZEN_IMAGE_CHECKPOINT, map_location='cpu')
    assert 'state_dict' in img_ckpt
    assert img_ckpt.get('model_architecture') == 'resnet18'

    temp_ckpt = torch.load(config.FROZEN_TEMPORAL_CHECKPOINT, map_location='cpu')
    assert 'state_dict' in temp_ckpt
    assert temp_ckpt.get('model_architecture') == 'lstm'
    assert len(temp_ckpt.get('temporal_features', [])) == 13


def test_pathology_metrics_csv_schema():
    """Validates pathology test metrics report schema and values."""
    csv_path = config.REPORTS_DIR / 'pathology_test_metrics.csv'
    assert csv_path.exists(), "pathology_test_metrics.csv not generated!"

    df = pd.read_csv(csv_path)
    metrics = dict(zip(df['metric'], df['value']))

    assert 'test_accuracy' in metrics
    assert 'test_macro_f1' in metrics
    assert 0.0 <= metrics['test_accuracy'] <= 1.0
    assert 0.0 <= metrics['test_macro_f1'] <= 1.0
    assert metrics['total_test_samples'] == 1800


def test_temporal_metrics_csv_schema():
    """Validates temporal test metrics report schema and values."""
    csv_path = config.REPORTS_DIR / 'temporal_test_metrics.csv'
    assert csv_path.exists(), "temporal_test_metrics.csv not generated!"

    df = pd.read_csv(csv_path)
    metrics = dict(zip(df['metric'], df['value']))

    assert 'mae' in metrics
    assert 'r2_score' in metrics
    assert 'accuracy' in metrics
    assert 'f1_score' in metrics
    assert metrics['mae'] >= 0.0
    assert metrics['accuracy'] <= 1.0


def test_leakage_audit_report_exists():
    """Ensures leakage audit markdown was produced and contains PASS status."""
    audit_path = config.REPORTS_DIR / 'leakage_audit.md'
    assert audit_path.exists(), "leakage_audit.md missing!"
    with open(audit_path, 'r', encoding='utf-8') as f:
        text = f.read()
    assert "**PASS**" in text
    assert config.MANDATORY_DISCLAIMER in text


def test_mandatory_disclaimers_present():
    """Verifies that regulatory synthetic notices are present."""
    assert config.MANDATORY_DISCLAIMER != ""
    assert "Synthetic data != Real patient evidence != Clinical validation" in config.EQUIVALENCE_DISCLAIMER


def test_source_data_immutability():
    """Verifies that Data Engineering v2 files are non-empty and accessible."""
    assert config.IMAGE_METADATA_PATH.stat().st_size > 0
    assert config.BIOMARKERS_PATH.stat().st_size > 0
    assert config.TEST_SPLIT_PATH.stat().st_size > 0
