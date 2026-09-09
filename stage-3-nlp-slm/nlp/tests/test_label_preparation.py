"""
Test Suite: Target Label Preparation & Class Weighting.
Verifies label encoding, inverse transformations, and class weight calculations.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from label_preparation import compute_class_weights, TargetLabelManager


def test_compute_class_weights():
    """Verify class weights are inversely proportional to class frequency."""
    # 80 class 0, 20 class 1
    y = np.array([0] * 80 + [1] * 20)
    weights = compute_class_weights(y)
    assert weights[0] < weights[1]
    # N / (2 * 80) = 100 / 160 = 0.625
    # N / (2 * 20) = 100 / 40 = 2.5
    assert np.isclose(weights[0], 0.625)
    assert np.isclose(weights[1], 2.5)


def test_target_label_manager_fit_transform():
    """Verify TargetLabelManager correctly maps and inverts urgency and hazard classes."""
    train_df = pd.DataFrame({
        "urgency_level": ["LOW", "MEDIUM", "HIGH", "CRITICAL", "LOW"],
        "hazard_type": ["NONE", "HEPATIC", "PULMONARY", "NONE", "RENAL"]
    })
    mgr = TargetLabelManager()
    mgr.fit(train_df)

    y_urg = mgr.transform_urgency(train_df["urgency_level"])
    assert len(y_urg) == 5
    inv_urg = mgr.inverse_transform_urgency(y_urg)
    assert list(inv_urg) == list(train_df["urgency_level"])

    y_haz = mgr.transform_hazard(train_df["hazard_type"])
    assert len(y_haz) == 5
    inv_haz = mgr.inverse_transform_hazard(y_haz)
    assert list(inv_haz) == list(train_df["hazard_type"])
