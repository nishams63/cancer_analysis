"""
Stage 2 EDA - Cohort & Patient Split Analysis Module
"""
import pandas as pd
from typing import Dict, Any, Tuple
try:
    from . import config
except (ImportError, ValueError):
    import config

def load_split_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads train, validation, and test patient split manifests."""
    train_df = pd.read_csv(config.TRAIN_SPLIT_PATH)
    val_df = pd.read_csv(config.VAL_SPLIT_PATH)
    test_df = pd.read_csv(config.TEST_SPLIT_PATH)
    return train_df, val_df, test_df

def verify_disjoint_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """Verifies disjoint set intersection between splits to ensure 0% patient leakage."""
    train_pats = set(train_df['patient_id'])
    val_pats = set(val_df['patient_id'])
    test_pats = set(test_df['patient_id'])

    tr_val_leak = train_pats.intersection(val_pats)
    tr_test_leak = train_pats.intersection(test_pats)
    val_test_leak = val_pats.intersection(test_pats)

    total_patients = len(train_pats | val_pats | test_pats)
    is_disjoint = (len(tr_val_leak) == 0 and len(tr_test_leak) == 0 and len(val_test_leak) == 0)

    return {
        'total_patients': total_patients,
        'train_patient_count': len(train_pats),
        'val_patient_count': len(val_pats),
        'test_patient_count': len(test_pats),
        'train_val_overlap': len(tr_val_leak),
        'train_test_overlap': len(tr_test_leak),
        'val_test_overlap': len(val_test_leak),
        'is_disjoint': is_disjoint,
        'status': 'PASS' if is_disjoint else 'FAIL'
    }

def analyze_split_distributions(
    image_meta_df: pd.DataFrame,
    biomarkers_df: pd.DataFrame,
    train_pats: pd.DataFrame,
    val_pats: pd.DataFrame,
    test_pats: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes split-level statistics combining patient, tile, and temporal distributions.
    Produces split_statistics.csv records.
    """
    splits = ['train', 'validation', 'test']
    records = []

    # Map patients to split
    pat_split = {}
    for p in train_pats['patient_id']:
        pat_split[p] = 'train'
    for p in val_pats['patient_id']:
        pat_split[p] = 'validation'
    for p in test_pats['patient_id']:
        pat_split[p] = 'test'

    # Trajectory per patient
    pat_traj = biomarkers_df.groupby('patient_id')['trajectory_pattern'].first().to_dict()

    for s in splits:
        pats_in_split = [p for p, sp in pat_split.items() if sp == s]
        p_count = len(pats_in_split)

        img_sub = image_meta_df[image_meta_df['split'] == s]
        t_count = len(img_sub)

        bio_sub = biomarkers_df[biomarkers_df['split'] == s]
        obs_count = len(bio_sub)

        # Class counts
        cls_counts = img_sub['class_label'].value_counts()
        b_count = int(cls_counts.get('benign', 0))
        m_count = int(cls_counts.get('malignant', 0))
        i_count = int(cls_counts.get('inflammation', 0))

        b_pct = round((b_count / t_count * 100), 2) if t_count > 0 else 0.0
        m_pct = round((m_count / t_count * 100), 2) if t_count > 0 else 0.0
        i_pct = round((i_count / t_count * 100), 2) if t_count > 0 else 0.0

        # Trajectory counts for patients in this split
        split_trajs = [pat_traj.get(p) for p in pats_in_split]
        traj_s = pd.Series(split_trajs).value_counts()

        rec = {
            'split': s,
            'patient_count': p_count,
            'tile_count': t_count,
            'temporal_observation_count': obs_count,
            'benign_count': b_count,
            'benign_pct': b_pct,
            'malignant_count': m_count,
            'malignant_pct': m_pct,
            'inflammation_count': i_count,
            'inflammation_pct': i_pct,
            'traj_stable_count': int(traj_s.get('stable', 0)),
            'traj_stable_pct': round(int(traj_s.get('stable', 0)) / p_count * 100, 2),
            'traj_gradual_inc_count': int(traj_s.get('gradual_increase', 0)),
            'traj_gradual_inc_pct': round(int(traj_s.get('gradual_increase', 0)) / p_count * 100, 2),
            'traj_gradual_dec_count': int(traj_s.get('gradual_decrease', 0)),
            'traj_gradual_dec_pct': round(int(traj_s.get('gradual_decrease', 0)) / p_count * 100, 2),
            'traj_fluctuating_count': int(traj_s.get('fluctuating', 0)),
            'traj_fluctuating_pct': round(int(traj_s.get('fluctuating', 0)) / p_count * 100, 2),
            'traj_rapid_inc_count': int(traj_s.get('rapid_increase', 0)),
            'traj_rapid_inc_pct': round(int(traj_s.get('rapid_increase', 0)) / p_count * 100, 2),
        }
        records.append(rec)

    split_stats_df = pd.DataFrame(records)
    return split_stats_df
