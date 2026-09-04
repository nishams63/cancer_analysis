"""
Patient-level split generator with zero cross-split leakage (Version 2).

DISCLAIMER:
Synthetic data created for deep-learning research and pipeline prototyping.
These data are not real patient records and are not clinically validated.
Synthetic data != Real patient evidence != Clinical validation.
"""

import random
from typing import Dict, List, Tuple
import pandas as pd

try:
    from .config import (
        CLINICAL_DISCLAIMER,
        NUM_PATIENTS,
        NUM_TEST,
        NUM_TRAIN,
        NUM_VAL,
        RANDOM_SEED,
        SPLITS_DIR,
        TEST_PATIENTS_PATH,
        TRAIN_PATIENTS_PATH,
        TRAJECTORY_ARCHETYPES,
        VAL_PATIENTS_PATH,
    )
except ImportError:
    from config import (
        CLINICAL_DISCLAIMER,
        NUM_PATIENTS,
        NUM_TEST,
        NUM_TRAIN,
        NUM_VAL,
        RANDOM_SEED,
        SPLITS_DIR,
        TEST_PATIENTS_PATH,
        TRAIN_PATIENTS_PATH,
        TRAJECTORY_ARCHETYPES,
        VAL_PATIENTS_PATH,
    )


def generate_patient_cohort(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate deterministic cohort metadata for 1,000 synthetic patients."""
    random.seed(seed)
    stages = ["Stage IIIA", "Stage IIIB", "Stage IV"]
    cohort_data = []

    for i in range(1, NUM_PATIENTS + 1):
        patient_id = f"PAT-{i:04d}"
        trajectory = TRAJECTORY_ARCHETYPES[(i - 1) % len(TRAJECTORY_ARCHETYPES)]
        stage = stages[(i - 1) % len(stages)]
        age = 45 + ((i * 11) % 35)

        cohort_data.append({
            "patient_id": patient_id,
            "cancer_type": "NSCLC",
            "clinical_stage": stage,
            "age": age,
            "biomarker_trajectory": trajectory,
            "data_source": "synthetic",
            "disclaimer": CLINICAL_DISCLAIMER,
        })

    return pd.DataFrame(cohort_data)


def create_patient_splits(
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split 1,000 patients into Train (700), Validation (150), and Test (150).
    Stratified by biomarker_trajectory to ensure balanced longitudinal dynamics.
    Guarantees:
      train_patients ∩ validation_patients = empty
      train_patients ∩ test_patients = empty
      validation_patients ∩ test_patients = empty
    """
    cohort_df = generate_patient_cohort(seed=seed)

    # Stratified shuffle across trajectory archetypes
    train_dfs, val_dfs, test_dfs = [], [], []

    for traj_name, group in cohort_df.groupby("biomarker_trajectory"):
        shuffled_group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        # Each trajectory has 200 patients: 140 train, 30 val, 30 test
        n_group = len(shuffled_group)
        n_train = int(n_group * (NUM_TRAIN / NUM_PATIENTS))  # 140
        n_val = int(n_group * (NUM_VAL / NUM_PATIENTS))      # 30
        
        train_dfs.append(shuffled_group.iloc[:n_train])
        val_dfs.append(shuffled_group.iloc[n_train:n_train + n_val])
        test_dfs.append(shuffled_group.iloc[n_train + n_val:])

    train_df = pd.concat(train_dfs).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_dfs).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_dfs).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    # Strict Zero-Leakage Assertions
    train_set = set(train_df["patient_id"])
    val_set = set(val_df["patient_id"])
    test_set = set(test_df["patient_id"])

    assert len(train_set) == NUM_TRAIN, f"Expected {NUM_TRAIN} train patients, got {len(train_set)}"
    assert len(val_set) == NUM_VAL, f"Expected {NUM_VAL} val patients, got {len(val_set)}"
    assert len(test_set) == NUM_TEST, f"Expected {NUM_TEST} test patients, got {len(test_set)}"

    assert len(train_set & val_set) == 0, f"LEAKAGE: Train & Val overlap: {train_set & val_set}"
    assert len(train_set & test_set) == 0, f"LEAKAGE: Train & Test overlap: {train_set & test_set}"
    assert len(val_set & test_set) == 0, f"LEAKAGE: Val & Test overlap: {val_set & test_set}"

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATIENTS_PATH, index=False)
    val_df.to_csv(VAL_PATIENTS_PATH, index=False)
    test_df.to_csv(TEST_PATIENTS_PATH, index=False)

    print(f"Patient splits (v2) created successfully at {SPLITS_DIR}")
    print(f"  - Train: {len(train_df)} patients (saved to {TRAIN_PATIENTS_PATH.name})")
    print(f"  - Validation: {len(val_df)} patients (saved to {VAL_PATIENTS_PATH.name})")
    print(f"  - Test: {len(test_df)} patients (saved to {TEST_PATIENTS_PATH.name})")
    print("  - Zero-leakage verification: 100% disjoint patient sets.")

    return train_df, val_df, test_df


def get_patient_split_map(seed: int = RANDOM_SEED) -> Dict[str, str]:
    """Returns mapping: patient_id -> 'train' | 'validation' | 'test'."""
    if not (TRAIN_PATIENTS_PATH.exists() and VAL_PATIENTS_PATH.exists() and TEST_PATIENTS_PATH.exists()):
        train_df, val_df, test_df = create_patient_splits(seed=seed)
    else:
        train_df = pd.read_csv(TRAIN_PATIENTS_PATH)
        val_df = pd.read_csv(VAL_PATIENTS_PATH)
        test_df = pd.read_csv(TEST_PATIENTS_PATH)

    mapping = {}
    for pid in train_df["patient_id"]:
        mapping[pid] = "train"
    for pid in val_df["patient_id"]:
        mapping[pid] = "validation"
    for pid in test_df["patient_id"]:
        mapping[pid] = "test"
    return mapping


if __name__ == "__main__":
    create_patient_splits()
