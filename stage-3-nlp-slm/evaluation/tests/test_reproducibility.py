"""
Unit tests for deterministic reproducibility across multiple evaluation runs.
"""

import numpy as np
from data_loader import load_validation_data
from model_loader import load_frozen_artifacts
from prediction_runner import run_predictions


def test_dual_pass_exact_reproducibility():
    df_val = load_validation_data().head(30)
    artifacts = load_frozen_artifacts()

    # Pass 1
    pred1, urg_probs1, haz_probs1, X1, _ = run_predictions(df_val, artifacts)

    # Pass 2
    pred2, urg_probs2, haz_probs2, X2, _ = run_predictions(df_val, artifacts)

    assert (pred1["predicted_urgency"] == pred2["predicted_urgency"]).all()
    assert (pred1["predicted_hazard"] == pred2["predicted_hazard"]).all()
    assert np.allclose(urg_probs1, urg_probs2, atol=1e-8)
    assert np.allclose(haz_probs1, haz_probs2, atol=1e-8)
    assert (X1 != X2).nnz == 0  # Sparse matrix exact equivalence
