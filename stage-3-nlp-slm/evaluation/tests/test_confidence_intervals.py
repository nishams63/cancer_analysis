"""
Unit tests for bootstrap confidence intervals calculation and determinism.
"""

import numpy as np
import pandas as pd
from confidence_intervals import compute_patient_bootstrap_ci


def test_bootstrap_ci_bounds_and_determinism():
    # Construct synthetic patient dataset
    data = []
    classes = ["LOW", "HIGH", "CRITICAL"]
    for i in range(20):
        pid = f"PT-{i:03d}"
        for _ in range(3):
            lbl = np.random.choice(classes)
            data.append({"patient_id": pid, "label": lbl})

    df = pd.DataFrame(data)
    y_true = df["label"].values
    y_pred = df["label"].values  # perfect agreement

    ci_res1 = compute_patient_bootstrap_ci(df, y_true, y_pred, n_iterations=50, random_seed=42, critical_class="CRITICAL")
    ci_res2 = compute_patient_bootstrap_ci(df, y_true, y_pred, n_iterations=50, random_seed=42, critical_class="CRITICAL")

    acc_ci1 = ci_res1["metrics"]["accuracy"]
    acc_ci2 = ci_res2["metrics"]["accuracy"]

    # Check bounds
    assert acc_ci1["ci_lower"] <= acc_ci1["mean"] <= acc_ci1["ci_upper"]
    assert acc_ci1["mean"] == 1.0

    # Check deterministic seed equality
    assert acc_ci1["ci_lower"] == acc_ci2["ci_lower"]
    assert acc_ci1["ci_upper"] == acc_ci2["ci_upper"]
    assert acc_ci1["std_err"] == acc_ci2["std_err"]
