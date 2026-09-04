"""
Longitudinal Biomarker Sequence Preparation for Stage 2 Deep Learning (Version 2).

Features:
  - Chronological Monotonicity: Verifies t(i) > t(i-1) with zero duplicates per patient.
  - Forecasting Windows: Strict demarcation of Historical Input (<= 90d) and Future Prediction (> 90d).
  - Explicit Future Targets: 30-day forward ctDNA target and progression trend labels.
  - Controlled Missingness: 3–8% tracked with binary indicator masks.
  - Zero Patient Leakage: Split assignments strictly inherit from patient-level partition.

DISCLAIMER:
Synthetic data created for deep-learning research and pipeline prototyping.
These data are not real patient records and are not clinically validated.
Synthetic data != Real patient evidence != Clinical validation.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

try:
    from .config import (
        BIOMARKERS_PROCESSED_PATH,
        CLINICAL_DISCLAIMER,
        FORECAST_SPLIT_DAY,
        RAW_BIOMARKERS_DIR,
    )
    from .create_splits import get_patient_split_map
except ImportError:
    from config import (
        BIOMARKERS_PROCESSED_PATH,
        CLINICAL_DISCLAIMER,
        FORECAST_SPLIT_DAY,
        RAW_BIOMARKERS_DIR,
    )
    from create_splits import get_patient_split_map


def prepare_longitudinal_biomarkers_v2() -> pd.DataFrame:
    """
    Cleans, sorts chronologically, verifies temporal validity, calculates velocity features,
    and formats the master v2 longitudinal biomarker dataset.
    """
    raw_path = RAW_BIOMARKERS_DIR / "raw_biomarkers.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw biomarker file not found at {raw_path}. Run data_generation first.")

    raw_df = pd.read_csv(raw_path)
    patient_split_map = get_patient_split_map()

    print(f"Preparing longitudinal biomarker sequences for {raw_df['patient_id'].nunique()} patients...")

    # Strict sorting: patient_id, then days_from_baseline
    df_sorted = raw_df.sort_values(by=["patient_id", "days_from_baseline"]).reset_index(drop=True)

    processed_rows = []

    for patient_id, group in df_sorted.groupby("patient_id"):
        split = patient_split_map.get(patient_id, "unknown")
        if split == "unknown":
            raise ValueError(f"Patient {patient_id} missing from split map!")

        prev_day = None
        prev_ctdna = None

        group_records = group.to_dict("records")
        for row in group_records:
            current_day = int(row["days_from_baseline"])

            # Strict chronological check: t(i) > t(i-1)
            if prev_day is not None and current_day <= prev_day:
                raise ValueError(
                    f"Temporal inversion or duplicate day for patient {patient_id}: "
                    f"day {current_day} follows day {prev_day}"
                )

            delta_days = 0 if prev_day is None else (current_day - prev_day)
            curr_ctdna = row["ctDNA_vaf_percent"]

            # Compute ctDNA velocity (rate of change per 30 days)
            if prev_ctdna is not None and curr_ctdna is not None and delta_days > 0 and not pd.isna(curr_ctdna) and not pd.isna(prev_ctdna):
                ctdna_velocity_30d = round(((curr_ctdna - prev_ctdna) / delta_days) * 30.0, 4)
            else:
                ctdna_velocity_30d = 0.0

            # Forecasting window validation
            is_input = 1 if current_day <= FORECAST_SPLIT_DAY else 0
            w_type = "historical_input" if is_input == 1 else "future_prediction"

            row_processed = {
                "patient_id": patient_id,
                "split": split,
                "timepoint_index": row["timepoint_index"],
                "timestamp": row["timestamp"],
                "days_from_baseline": current_day,
                "delta_days": delta_days,
                "ctDNA_vaf_percent": row["ctDNA_vaf_percent"],
                "cea_ng_ml": row["cea_ng_ml"],
                "ca125_u_ml": row["ca125_u_ml"],
                "ldh_u_l": row["ldh_u_l"],
                "crp_mg_l": row["crp_mg_l"],
                "ctDNA_velocity_30d": ctdna_velocity_30d,
                "ctDNA_missing": int(row["ctDNA_missing"]),
                "cea_missing": int(row["cea_missing"]),
                "ca125_missing": int(row["ca125_missing"]),
                "ldh_missing": int(row["ldh_missing"]),
                "crp_missing": int(row["crp_missing"]),
                "trajectory_pattern": row["trajectory_pattern"],
                "window_type": w_type,
                "is_input_window": is_input,
                "future_ctDNA_30d_target": row.get("future_ctDNA_30d_target"),
                "future_progression_trend": int(row.get("future_progression_trend", 0)),
                "data_source": "synthetic",
                "disclaimer": CLINICAL_DISCLAIMER,
            }
            processed_rows.append(row_processed)

            prev_day = current_day
            if curr_ctdna is not None and not pd.isna(curr_ctdna):
                prev_ctdna = curr_ctdna

    processed_df = pd.DataFrame(processed_rows)
    BIOMARKERS_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(BIOMARKERS_PROCESSED_PATH, index=False)
    print(f"Biomarker processing complete. Saved {len(processed_df)} observations to {BIOMARKERS_PROCESSED_PATH}")

    # Summary by split and window
    print("\nBiomarker observation summary by split:")
    summary = processed_df.groupby(["split", "window_type"]).agg(
        patient_count=("patient_id", "nunique"),
        total_obs=("timepoint_index", "count"),
        ctdna_missing_pct=("ctDNA_missing", lambda m: f"{m.mean()*100:.1f}%"),
    )
    print(summary)

    return processed_df


if __name__ == "__main__":
    prepare_longitudinal_biomarkers_v2()
