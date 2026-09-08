"""
Stage 2 Deep Learning - Enhanced Datasets & Dataset V2 Challenge Layer
Includes:
  1. Enhanced PathologyTileDataset with realistic training augmentations
  2. PatientPathologyBagDataset for 12-tile Attention-MIL modeling
  3. Continuous TemporalSequenceDataset for irregular longitudinal time series
  4. DatasetV2PerturbationSuite for controlled clinical stress-testing
"""
import os
import io
import math
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from . import config


# -------------------------------------------------------------------------
# 1. Pathology Datasets
# -------------------------------------------------------------------------

class GaussianNoiseTransform:
    """Additive Gaussian noise on normalized image tensors."""
    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return tensor + noise


class JpegCompressionTransform:
    """Simulates lossy whole-slide compression artifacts."""
    def __init__(self, quality: int = 45):
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


def get_pathology_transforms(
    split: str = 'train',
    use_imagenet_norm: bool = True,
    apply_challenge: Optional[str] = None
) -> transforms.Compose:
    """
    Returns separate train and val/test transforms.
    Ensures validation and test sets receive strictly deterministic evaluation transforms.
    """
    norm_mean = config.IMAGENET_MEAN if use_imagenet_norm else config.DATASET_MEAN
    norm_std = config.IMAGENET_STD if use_imagenet_norm else config.DATASET_STD

    base_resize = transforms.Resize(config.PATHOLOGY_CONFIG.image_size)
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

    if apply_challenge:
        # Programmatic perturbation for Dataset V2 stress testing
        ops = [base_resize]
        if apply_challenge == 'brightness':
            ops.append(transforms.ColorJitter(brightness=0.35))
        elif apply_challenge == 'contrast':
            ops.append(transforms.ColorJitter(contrast=0.45))
        elif apply_challenge == 'stain_color':
            ops.append(transforms.ColorJitter(hue=0.08, saturation=0.25))
        elif apply_challenge == 'blur':
            ops.append(transforms.GaussianBlur(kernel_size=5, sigma=(1.5, 2.0)))
        elif apply_challenge == 'compression':
            ops.append(JpegCompressionTransform(quality=35))
        
        ops.extend([transforms.ToTensor(), normalize])
        if apply_challenge == 'noise':
            ops.append(GaussianNoiseTransform(std=0.08))
        return transforms.Compose(ops)

    if split == 'train':
        # Realistic training-only augmentations preserving pathology meaning
        return transforms.Compose([
            base_resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.04),
            transforms.ToTensor(),
            normalize,
            GaussianNoiseTransform(std=0.02)
        ])
    else:
        # Deterministic evaluation transform for validation and locked test
        return transforms.Compose([
            base_resize,
            transforms.ToTensor(),
            normalize
        ])


def resolve_image_path(raw_path: str, rel_path: str = '') -> str:
    """Safely resolves image file paths across different machines."""
    if os.path.exists(raw_path):
        return raw_path
    
    fname = os.path.basename(raw_path)
    for cls_name in ['benign', 'malignant', 'inflammation']:
        cand = config.PATHOLOGY_TILES_DIR / cls_name / fname
        if cand.exists():
            return str(cand)

    if rel_path:
        cleaned_rel = rel_path.replace('data-engineering/data-engineering/', 'data-engineering/')
        cand2 = config.STAGE_2_DIR / cleaned_rel
        if cand2.exists():
            return str(cand2)

    matches = list(config.PATHOLOGY_TILES_DIR.rglob(fname))
    if matches:
        return str(matches[0])

    return raw_path


class PathologyTileDataset(Dataset):
    """
    Standard tile-level classification dataset.
    Strictly partitions by patient_id using split manifests.
    """
    def __init__(
        self,
        split: str = 'train',
        use_imagenet_norm: bool = True,
        challenge: Optional[str] = None
    ):
        assert split in config.SPLITS, f"Invalid split: {split}. Must be one of {config.SPLITS}"
        self.split = split
        self.challenge = challenge
        self.transform = get_pathology_transforms(split=split, use_imagenet_norm=use_imagenet_norm, apply_challenge=challenge)
        
        full_df = pd.read_csv(config.IMAGE_METADATA_PATH)
        self.df = full_df[full_df['split'] == split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        img_path = resolve_image_path(row['image_path'], row.get('relative_path', ''))
        
        image = Image.open(img_path).convert('RGB')
        tensor_img = self.transform(image)
        
        class_str = row['class_label']
        label = config.CLASS_TO_IDX[class_str]
        
        meta = {
            'tile_id': row['tile_id'],
            'patient_id': row['patient_id'],
            'slide_id': row['slide_id'],
            'class_label': class_str,
            'split': self.split
        }
        return tensor_img, label, meta


class PatientPathologyBagDataset(Dataset):
    """
    Patient-Level Bag Dataset for Multiple Instance Learning (MIL).
    Groups all 12 tiles for each patient into a single bag:
      bag shape: (12, 3, 224, 224)
      patient_label: majority / primary morphological class (or tile class distribution)
    """
    def __init__(
        self,
        split: str = 'train',
        use_imagenet_norm: bool = True,
        challenge: Optional[str] = None,
        max_tiles_override: Optional[int] = None
    ):
        assert split in config.SPLITS, f"Invalid split: {split}. Must be one of {config.SPLITS}"
        self.split = split
        self.transform = get_pathology_transforms(split=split, use_imagenet_norm=use_imagenet_norm, apply_challenge=challenge)
        self.max_tiles_override = max_tiles_override
        
        meta_df = pd.read_csv(config.IMAGE_METADATA_PATH)
        split_df = meta_df[meta_df['split'] == split].copy()
        
        # Group by patient
        self.patient_bags = []
        for pat_id, grp in split_df.groupby('patient_id'):
            # Determine patient-level primary label: if any tile is malignant -> high suspicion
            tile_labels = [config.CLASS_TO_IDX[c] for c in grp['class_label']]
            # Patient has malignant histology if >= 1 malignant tile, else inflammation if present, else benign
            if 1 in tile_labels:
                pat_label = 1  # malignant
            elif 2 in tile_labels:
                pat_label = 2  # inflammation
            else:
                pat_label = 0  # benign
                
            self.patient_bags.append({
                'patient_id': pat_id,
                'rows': grp.to_dict('records'),
                'patient_label': pat_label,
                'tile_labels': tile_labels
            })

    def __len__(self) -> int:
        return len(self.patient_bags)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, str]:
        bag_info = self.patient_bags[idx]
        rows = bag_info['rows']
        
        if self.max_tiles_override is not None and self.max_tiles_override < len(rows):
            # Challenge condition: subset of biopsy tiles available
            rows = rows[:self.max_tiles_override]
            
        tensors = []
        tile_labels = []
        for r in rows:
            p = resolve_image_path(r['image_path'], r.get('relative_path', ''))
            im = Image.open(p).convert('RGB')
            tensors.append(self.transform(im))
            tile_labels.append(config.CLASS_TO_IDX[r['class_label']])
            
        bag_tensor = torch.stack(tensors)  # (K, 3, 224, 224)
        tile_label_tensor = torch.tensor(tile_labels, dtype=torch.long)
        
        return bag_tensor, bag_info['patient_label'], tile_label_tensor, bag_info['patient_id']


# -------------------------------------------------------------------------
# 2. Longitudinal Biomarker Dataset with Irregular Intervals
# -------------------------------------------------------------------------

class ContinuousTemporalSequenceDataset(Dataset):
    """
    Longitudinal Biomarker Sequence Dataset handling irregular time intervals.
    Strictly filters to historical induction window (days_from_baseline <= 90).
    Explicitly provides:
      - Continuous feature tensor: (max_seq_len, 13)
      - delta_days and days_from_baseline features
      - True sequence length (for causal mask and padding suppression)
      - Dual targets: 30d future ctDNA VAF and progression status
    """
    def __init__(
        self,
        split: str = 'train',
        norm_params: Optional[Dict[str, Tuple[float, float]]] = None,
        max_seq_len: int = 10,
        challenge: Optional[str] = None
    ):
        assert split in config.SPLITS, f"Invalid split: {split}. Must be one of {config.SPLITS}"
        self.split = split
        self.max_seq_len = max_seq_len
        self.challenge = challenge
        
        full_df = pd.read_csv(config.BIOMARKERS_PATH)
        
        # Strict anti-leakage filter: only historical observations (<= 90 days)
        hist_df = full_df[(full_df['is_input_window'] == 1) & (full_df['days_from_baseline'] <= config.FORECAST_SPLIT_DAY)].copy()
        self.df = hist_df[hist_df['split'] == split].copy()
        self.df = self.df.sort_values(['patient_id', 'timepoint_index', 'days_from_baseline'])
        
        # Learn normalization parameters strictly from training patients
        if norm_params is None:
            if split == 'train':
                self.norm_params = self._compute_train_norm_params(self.df)
            else:
                train_sub = hist_df[hist_df['split'] == 'train']
                self.norm_params = self._compute_train_norm_params(train_sub)
        else:
            self.norm_params = norm_params
            
        self.sequences = self._prepare_sequences()

    def _compute_train_norm_params(self, train_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        params = {}
        for feat in config.TEMPORAL_NUMERICAL_FEATURES:
            vals = train_df[feat].dropna().values
            mean_v = float(np.mean(vals)) if len(vals) > 0 else 0.0
            std_v = float(np.std(vals)) if len(vals) > 0 else 1.0
            if std_v < 1e-6:
                std_v = 1.0
            params[feat] = (mean_v, std_v)
        return params

    def _prepare_sequences(self) -> List[Dict[str, Any]]:
        seqs = []
        for pat_id, grp in self.df.groupby('patient_id'):
            g = grp.copy()
            
            # Apply Dataset V2 perturbations if challenge is active
            if self.challenge == 'missing_visits' and len(g) > 2:
                # Drop an interim visit (never drop baseline visit)
                drop_idx = np.random.choice(range(1, len(g) - 1))
                g = g.drop(g.index[drop_idx])
            elif self.challenge == 'irregular_intervals':
                # Jitter delta_days
                g['delta_days'] = g['delta_days'] + np.random.uniform(-8, 15, size=len(g))
                g['delta_days'] = g['delta_days'].clip(lower=1.0)
            elif self.challenge == 'measurement_noise':
                # Add 15% Gaussian noise to lab values
                for num_col in ['ctDNA_vaf_percent', 'cea_ng_ml', 'ca125_u_ml', 'ldh_u_l', 'crp_mg_l']:
                    noise = np.random.normal(0, 0.15, size=len(g))
                    g[num_col] = (g[num_col] * (1.0 + noise)).clip(lower=0.0)
            elif self.challenge == 'missing_biomarkers':
                # Simulate dropout of a lab panel
                drop_col = np.random.choice(['cea_ng_ml', 'ca125_u_ml', 'ldh_u_l'])
                g[drop_col] = 0.0
                mask_col = drop_col.split('_')[0] + '_missing'
                if mask_col in g.columns:
                    g[mask_col] = 1.0
            elif self.challenge == 'outliers':
                # Random 3x spike
                if np.random.rand() < 0.2:
                    g['crp_mg_l'] = g['crp_mg_l'] * 3.0
                    
            # Imputation within trajectory (forward fill then zero fill)
            for col in config.TEMPORAL_NUMERICAL_FEATURES:
                g[col] = g[col].ffill().fillna(0.0)
                
            # Normalize continuous features
            norm_matrix = np.zeros((len(g), len(config.TEMPORAL_NUMERICAL_FEATURES)), dtype=np.float32)
            for i, col in enumerate(config.TEMPORAL_NUMERICAL_FEATURES):
                mean_v, std_v = self.norm_params[col]
                norm_matrix[:, i] = (g[col].values - mean_v) / std_v
                
            mask_matrix = g[config.TEMPORAL_MASK_FEATURES].values.astype(np.float32)
            feat_matrix = np.concatenate([norm_matrix, mask_matrix], axis=1)
            
            actual_len = min(len(feat_matrix), self.max_seq_len)
            padded_feat = np.zeros((self.max_seq_len, config.NUM_TEMPORAL_FEATURES), dtype=np.float32)
            padded_feat[:actual_len, :] = feat_matrix[:actual_len, :]
            
            # Extract target from final historical observation
            last_row = g.iloc[-1]
            target_ctdna = float(last_row[config.TARGET_REGRESSION]) if pd.notna(last_row[config.TARGET_REGRESSION]) else 0.0
            target_prog = int(last_row[config.TARGET_CLASSIFICATION]) if pd.notna(last_row[config.TARGET_CLASSIFICATION]) else 0
            
            seqs.append({
                'patient_id': pat_id,
                'features': torch.tensor(padded_feat, dtype=torch.float32),
                'length': actual_len,
                'target_ctdna': torch.tensor(target_ctdna, dtype=torch.float32),
                'target_progression': torch.tensor(target_prog, dtype=torch.float32),
                'split': self.split
            })
        return seqs

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.sequences[idx]
