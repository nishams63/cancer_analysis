"""Auditable temporal data and a separate, outcome-independent synthetic MIL challenge."""
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from . import config


def stable_seed(identifier, seed=42):
    return int.from_bytes(hashlib.sha256(f'{seed}:{identifier}'.encode()).digest()[:4], 'little')


def audit_manifests():
    manifests = {s: set(pd.read_csv(config.SPLITS_DIR / f'{s}_patients.csv').patient_id)
                 for s in config.SPLITS}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if manifests[a] & manifests[b]:
            raise ValueError(f'Patient overlap: {a}/{b}')
    for path in [config.IMAGE_METADATA_PATH, config.BIOMARKERS_PATH]:
        df = pd.read_csv(path)
        if df.groupby('patient_id').split.nunique().max() != 1:
            raise ValueError(f'Conflicting patient splits: {path}')
        for split, patients in manifests.items():
            if set(df.loc[df.split.eq(split), 'patient_id']) != patients:
                raise ValueError(f'Manifest disagrees with {path}: {split}')
    return {s: len(v) for s, v in manifests.items()}


def composition_bags(metadata, seed=42, difficulty=0.5):
    """12 within-patient sampled tiles; no biomarker values or future labels are read.

    Existing balanced bags have a constant any-malignant label. This separate
    synthetic task samples a dominant histology and then a mixed biopsy bag.
    Repeated source tiles are explicitly recorded and are not independent tissue.
    """
    if not 0 <= difficulty <= 1:
        raise ValueError('difficulty must be in [0,1]')
    bags = []
    for patient, grp in metadata.groupby('patient_id', sort=True):
        rng = np.random.default_rng(stable_seed(patient, seed))
        dominant = int(rng.integers(3))
        probabilities = np.full(3, (0.1 + 0.15 * difficulty))
        probabilities[dominant] = 1 - probabilities.sum() + probabilities[dominant]
        classes = rng.choice(3, size=12, p=probabilities)
        rows = []
        for k, label in enumerate(classes):
            available = grp[grp.class_label.eq(config.IDX_TO_CLASS[int(label)])]
            row = available.iloc[int(rng.integers(len(available)))].to_dict()
            row['source_tile_id'] = row['tile_id']
            row['bag_position'] = k
            rows.append(row)
        # Target is observed dominant composition, not the latent sampled class.
        counts = np.bincount(classes, minlength=3)
        label = int(np.argmax(counts))  # predefined benign-first tie policy
        bags.append(dict(patient_id=patient, rows=rows, patient_label=label,
                         tile_labels=classes.tolist()))
    return bags


LABS = config.TEMPORAL_NUMERICAL_FEATURES[:5]


def prepare_history(history, norm_params, max_seq_len=10, challenge=None, seed=42):
    """Features depend only on supplied <=90-day observations; targets never enter here."""
    g = history.sort_values('days_from_baseline').copy()
    if g.empty or not g.days_from_baseline.between(0, 90).all():
        raise ValueError('History must contain only day 0–90 observations')
    if g.days_from_baseline.duplicated().any():
        raise ValueError('Duplicate visit dates')
    rng = np.random.default_rng(seed)
    if challenge == 'missing_visits' and len(g) > 2:
        keep = np.ones(len(g), dtype=bool)
        keep[1:-1] = rng.random(len(g)-2) >= 0.33
        g = g.iloc[keep].copy()
    elif challenge == 'short_histories':
        g = g.tail(2).copy()
    elif challenge == 'irregular_intervals':
        days = g.days_from_baseline.to_numpy(float)
        if len(days) > 2:
            for i in range(1, len(days)-1):
                days[i] = np.clip(days[i] + rng.uniform(-5, 5), days[i-1]+0.1, days[i+1]-0.1)
        g['days_from_baseline'] = days
    elif challenge == 'measurement_noise':
        for col in LABS:
            g[col] = (g[col] * (1 + rng.normal(0, 0.15, len(g)))).clip(lower=0)
    elif challenge in ('outliers', 'temporary_spikes'):
        i = int(rng.integers(len(g)))
        col = 'crp_mg_l' if challenge == 'outliers' else 'ctDNA_vaf_percent'
        g.loc[g.index[i], col] *= 3
    elif challenge == 'missing_biomarkers':
        g[LABS[int(rng.integers(5))]] = np.nan
    elif challenge is not None:
        raise ValueError(f'Unknown temporal challenge: {challenge}')
    for col, mask in zip(LABS, config.TEMPORAL_MASK_FEATURES):
        if col not in g:
            g[col] = np.nan
        g[mask] = g[col].isna().astype(float)
        # ffill is causal. No backward fill or future-derived imputation.
        g[col] = g[col].ffill().fillna(norm_params[col][0])
    g['delta_days'] = g.days_from_baseline.diff().fillna(0)
    g['ctDNA_velocity_30d'] = (g.ctDNA_vaf_percent.diff() / g.delta_days.replace(0, np.nan) * 30).fillna(0)
    # Absolute/relative days retain their physical units for the time embedding.
    matrix = []
    for col in config.TEMPORAL_NUMERICAL_FEATURES:
        mean, std = norm_params[col]
        values = g[col].to_numpy(float)
        matrix.append(values if col in ('days_from_baseline', 'delta_days') else (values-mean)/std)
    matrix.extend(g[col].to_numpy(float) for col in config.TEMPORAL_MASK_FEATURES)
    x = np.stack(matrix, axis=1)[-max_seq_len:].astype('float32')
    padded = np.zeros((max_seq_len, config.NUM_TEMPORAL_FEATURES), dtype='float32')
    padded[:len(x)] = x
    if not np.isfinite(padded).all():
        raise ValueError('Nonfinite input features')
    return torch.from_numpy(padded), len(x), g


class AuditedTemporalDataset(Dataset):
    """Day 90 landmark, ctDNA measurement nearest day 120 within +/-12 days.

    Missing targets are excluded, with patient IDs/reasons retained in exclusions.
    Classification retains the generator's future-window progression definition;
    it is a synthetic biomarker trend, not a clinical recurrence diagnosis.
    """
    def __init__(self, split='train', norm_params=None, max_seq_len=10, challenge=None, seed=42):
        if split not in config.SPLITS:
            raise ValueError(split)
        full = pd.read_csv(config.BIOMARKERS_PATH)
        hist = full[full.days_from_baseline.between(0, 90)].copy()
        train = hist[hist.split.eq('train')]
        self.norm_params = norm_params or {
            col: (float(train[col].mean()), max(float(train[col].std(ddof=0)), 1e-6))
            for col in config.TEMPORAL_NUMERICAL_FEATURES}
        self.df = hist[hist.split.eq(split)].sort_values(['patient_id','days_from_baseline'])
        self.split, self.challenge, self.max_seq_len = split, challenge, max_seq_len
        self.sequences, self.exclusions = [], []
        future = full[full.days_from_baseline.gt(90)]
        future_map = {p: g for p, g in future.groupby('patient_id')}
        for patient, g in self.df.groupby('patient_id', sort=True):
            candidates = future_map.get(patient, full.iloc[:0]).copy()
            candidates = candidates[candidates.ctDNA_vaf_percent.notna() & candidates.days_from_baseline.between(108,132)]
            if candidates.empty:
                self.exclusions.append({'patient_id': patient, 'reason': 'No observed ctDNA within day 120 +/-12'})
                continue
            candidates['distance'] = (candidates.days_from_baseline - 120).abs()
            target = candidates.sort_values(['distance','days_from_baseline']).iloc[0]
            x, length, effective = prepare_history(g, self.norm_params, max_seq_len, challenge, stable_seed(patient,seed))
            self.sequences.append(dict(patient_id=patient, features=x, length=length,
                target_ctdna=torch.tensor(float(target.ctDNA_vaf_percent), dtype=torch.float32),
                target_progression=torch.tensor(float(g.iloc[-1][config.TARGET_CLASSIFICATION]), dtype=torch.float32),
                target_day=float(target.days_from_baseline), forecast_anchor_day=90.,
                last_input_day=float(effective.days_from_baseline.max()), split=split))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]
