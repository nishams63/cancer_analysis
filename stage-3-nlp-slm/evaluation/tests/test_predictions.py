"""
Unit tests for batch prediction runner in evaluation module.
"""

import numpy as np
from data_loader import load_validation_data
from model_loader import load_frozen_artifacts
from prediction_runner import run_predictions


def test_prediction_runner_output():
    df_val = load_validation_data().head(15)
    artifacts = load_frozen_artifacts()

    pred_df, urg_probs, haz_probs, X_mat, struct_df = run_predictions(df_val, artifacts)

    assert len(pred_df) == 15
    assert urg_probs.shape == (15, 4)
    assert haz_probs.shape == (15, 8)
    assert X_mat.shape[1] == 1012

    # Check probabilities
    assert np.allclose(urg_probs.sum(axis=1), 1.0, atol=1e-4)
    assert np.allclose(haz_probs.sum(axis=1), 1.0, atol=1e-4)

    # Check confidence columns
    assert (pred_df["urgency_confidence"] >= 0.0).all() and (pred_df["urgency_confidence"] <= 1.0).all()
    assert (pred_df["hazard_confidence"] >= 0.0).all() and (pred_df["hazard_confidence"] <= 1.0).all()

    # Check predictions validity
    assert set(pred_df["predicted_urgency"]).issubset(set(artifacts.urgency_encoder.classes_))
    assert set(pred_df["predicted_hazard"]).issubset(set(artifacts.hazard_encoder.classes_))
