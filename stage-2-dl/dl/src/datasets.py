"""
Stage 2 Deep Learning (DL) - Dataset Implementations
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Dict, Any, Tuple, List, Optional

try:
    from . import config
except (ImportError, ValueError):
    import config

class PathologyTileDataset(Dataset):
    """
    PyTorch Dataset for 224x224x3 pathology image tiles.
    Strictly partitions by patient_id using Data Engineering split manifests.
    Augmentation is restricted exclusively to the 'train' split.
    """
    def __init__(self, split: str = 'train', use_imagenet_norm: bool = False, image_meta_path: Optional[str] = None):
        assert split in config.SPLITS, f"Invalid split: {split}. Must be one of {config.SPLITS}"
        self.split = split
        self.use_imagenet_norm = use_imagenet_norm
        
        meta_path = image_meta_path or config.IMAGE_METADATA_PATH
        full_df = pd.read_csv(meta_path)
        
        # Filter strictly to the specified patient partition
        self.df = full_df[full_df['split'] == split].reset_index(drop=True)
        
        # Select normalization constants
        if use_imagenet_norm:
            norm_mean = config.IMAGENET_MEAN
            norm_std = config.IMAGENET_STD
        else:
            norm_mean = config.DATASET_MEAN
            norm_std = config.DATASET_STD

        # Base transform for all splits
        base_ops = [
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ]

        # Training-only augmentations
        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                *base_ops
            ])
        else:
            self.transform = transforms.Compose(base_ops)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        # Fallback path resolution if absolute drive path differs
        if not os.path.exists(img_path):
            rel = row.get('relative_path', '')
            alt_path = config.STAGE_2_DIR / rel.replace('data-engineering/data-engineering/', 'data-engineering/')
            if alt_path.exists():
                img_path = str(alt_path)
                
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


class TemporalSequenceDataset(Dataset):
    """
    PyTorch Dataset for longitudinal biomarker sequences.
    Strictly filters to historical induction window (days_from_baseline <= 90).
    Normalizes continuous features using parameters learned exclusively from training patients.
    Extracts landmark prediction targets: future_ctDNA_30d_target and future_progression_trend.
    """
    def __init__(
        self,
        split: str = 'train',
        bio_path: Optional[str] = None,
        norm_params: Optional[Dict[str, Tuple[float, float]]] = None,
        max_seq_len: int = 10
    ):
        assert split in config.SPLITS, f"Invalid split: {split}. Must be one of {config.SPLITS}"
        self.split = split
        self.max_seq_len = max_seq_len
        
        csv_path = bio_path or config.BIOMARKERS_PATH
        full_df = pd.read_csv(csv_path)
        
        # Strict anti-leakage filter: only historical input window observations (<= 90 days)
        hist_df = full_df[(full_df['is_input_window'] == 1) & (full_df['days_from_baseline'] <= config.FORECAST_SPLIT_DAY)].copy()
        
        # Filter to patient split
        self.df = hist_df[hist_df['split'] == split].copy()
        
        # Sort chronologically
        self.df = self.df.sort_values(['patient_id', 'timepoint_index', 'days_from_baseline'])
        
        # Compute normalization parameters strictly on TRAINING data if not supplied
        if norm_params is None:
            if split == 'train':
                self.norm_params = self._compute_train_norm_params(self.df)
            else:
                train_sub = hist_df[hist_df['split'] == 'train']
                self.norm_params = self._compute_train_norm_params(train_sub)
        else:
            self.norm_params = norm_params
            
        # Group by patient into pre-processed sequence tensors
        self.patient_sequences = self._prepare_patient_sequences()

    def _compute_train_norm_params(self, train_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Learns mean and std strictly on training set patients."""
        params = {}
        for feat in config.TEMPORAL_NUMERICAL_FEATURES:
            vals = train_df[feat].dropna().values
            mean_v = float(np.mean(vals)) if len(vals) > 0 else 0.0
            std_v = float(np.std(vals)) if len(vals) > 0 else 1.0
            if std_v < 1e-6:
                std_v = 1.0
            params[feat] = (mean_v, std_v)
        return params

    def _prepare_patient_sequences(self) -> List[Dict[str, Any]]:
        """Constructs padded sequence arrays and landmark forecasting targets."""
        sequences = []
        patient_groups = self.df.groupby('patient_id')
        
        for pat_id, group in patient_groups:
            # Impute missing values within patient trajectory (forward fill, then 0 fill)
            grp = group.copy()
            for col in config.TEMPORAL_NUMERICAL_FEATURES:
                grp[col] = grp[col].ffill().fillna(0.0)
                
            # Normalize numerical features
            norm_matrix = np.zeros((len(grp), len(config.TEMPORAL_NUMERICAL_FEATURES)), dtype=np.float32)
            for i, col in enumerate(config.TEMPORAL_NUMERICAL_FEATURES):
                mean_v, std_v = self.norm_params[col]
                norm_matrix[:, i] = (grp[col].values - mean_v) / std_v
                
            # Keep mask features binary (0 or 1)
            mask_matrix = grp[config.TEMPORAL_MASK_FEATURES].values.astype(np.float32)
            
            # Combined features (shape: L, D)
            feat_matrix = np.concatenate([norm_matrix, mask_matrix], axis=1)
            actual_len = min(len(feat_matrix), self.max_seq_len)
            
            # Padded tensor (max_seq_len, D)
            padded_feat = np.zeros((self.max_seq_len, config.NUM_TEMPORAL_FEATURES), dtype=np.float32)
            padded_feat[:actual_len, :] = feat_matrix[:actual_len, :]
            
            # Extract landmark target at final historical visit (t <= 90)
            last_row = grp.iloc[-1]
            target_ctdna = float(last_row[config.TARGET_REGRESSION]) if pd.notna(last_row[config.TARGET_REGRESSION]) else 0.0
            target_prog = int(last_row[config.TARGET_CLASSIFICATION]) if pd.notna(last_row[config.TARGET_CLASSIFICATION]) else 0
            
            sequences.append({
                'patient_id': pat_id,
                'features': torch.tensor(padded_feat, dtype=torch.float32),
                'length': actual_len,
                'target_ctdna': torch.tensor(target_ctdna, dtype=torch.float32),
                'target_progression': torch.tensor(target_prog, dtype=torch.float32),
                'split': self.split
            })
            
        return sequences

    def __len__(self) -> int:
        return len(self.patient_sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.patient_sequences[idx]
