"""
Stage 2 Deep Learning (DL) - Standalone Inference Utilities
"""
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Any, Union, List, Optional
from pathlib import Path

try:
    from . import config, utils, image_model, temporal_model
except (ImportError, ValueError):
    import config, utils, image_model, temporal_model

class PathologyImagePredictor:
    """
    Self-contained inference engine for pathology tile classification.
    Accepts image file path or PIL Image; returns class probabilities and prediction.
    """
    def __init__(self, checkpoint_path: Optional[str] = None):
        chk_path = checkpoint_path or str(config.BEST_IMAGE_CHECKPOINT)
        if not os.path.exists(chk_path):
            raise FileNotFoundError(f"Checkpoint not found at: {chk_path}. Train model first.")

        self.checkpoint = utils.load_checkpoint(chk_path, map_location='cpu')
        arch = self.checkpoint.get('model_architecture', 'resnet18')
        
        self.model = image_model.build_image_model(
            arch=arch,
            pretrained=False,
            num_classes=config.NUM_CLASSES
        )
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.eval()

        norm_mean = self.checkpoint.get('norm_mean', config.IMAGENET_MEAN)
        norm_std = self.checkpoint.get('norm_std', config.IMAGENET_STD)

        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])

    def predict(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Runs single-tile inference.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor_x = self.transform(img).unsqueeze(0)  # (1, 3, 224, 224)

        with torch.no_grad():
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
            pred_idx = int(np.argmax(probs))

        pred_class = config.IDX_TO_CLASS[pred_idx]
        return {
            'prediction': pred_class,
            'class_idx': pred_idx,
            'confidence': float(probs[pred_idx]),
            'probabilities': {
                'benign': float(probs[0]),
                'malignant': float(probs[1]),
                'inflammation': float(probs[2])
            },
            'disclaimer': config.MANDATORY_DISCLAIMER
        }


class TemporalBiomarkerPredictor:
    """
    Self-contained inference engine for patient biomarker trajectory forecasting.
    Accepts patient DataFrame of visits within induction window (t <= 90);
    Predicts forward 30-day ctDNA VAF and future progression risk.
    """
    def __init__(self, checkpoint_path: Optional[str] = None):
        chk_path = checkpoint_path or str(config.BEST_TEMPORAL_CHECKPOINT)
        if not os.path.exists(chk_path):
            raise FileNotFoundError(f"Checkpoint not found at: {chk_path}. Train model first.")

        self.checkpoint = utils.load_checkpoint(chk_path, map_location='cpu')
        arch = self.checkpoint.get('model_architecture', 'lstm')
        hidden_dim = self.checkpoint.get('hidden_dim', 64)

        self.model = temporal_model.build_temporal_model(
            arch=arch,
            input_dim=config.NUM_TEMPORAL_FEATURES,
            hidden_dim=hidden_dim
        )
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.eval()

        self.norm_params = self.checkpoint.get('norm_params', {})
        self.features_list = self.checkpoint.get('temporal_features', config.ALL_TEMPORAL_FEATURES)

    def predict(self, patient_history_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs patient sequence forecasting.
        """
        df = patient_history_df.copy()
        # Enforce anti-leakage boundary check
        if (df['days_from_baseline'] > config.FORECAST_SPLIT_DAY).any():
            raise ValueError(
                f"Input sequence contains observations beyond Day {config.FORECAST_SPLIT_DAY}. "
                "Forecasting model must strictly receive historical data (<= 90 days)."
            )

        df = df.sort_values('days_from_baseline')

        # Forward fill and 0-fill
        for col in config.TEMPORAL_NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(0.0)
            else:
                df[col] = 0.0

        # Normalize
        norm_matrix = np.zeros((len(df), len(config.TEMPORAL_NUMERICAL_FEATURES)), dtype=np.float32)
        for i, col in enumerate(config.TEMPORAL_NUMERICAL_FEATURES):
            if col in self.norm_params:
                mean_v, std_v = self.norm_params[col]
            else:
                mean_v, std_v = 0.0, 1.0
            norm_matrix[:, i] = (df[col].values - mean_v) / (std_v + 1e-6)

        # Mask features
        mask_matrix = np.zeros((len(df), len(config.TEMPORAL_MASK_FEATURES)), dtype=np.float32)
        for i, col in enumerate(config.TEMPORAL_MASK_FEATURES):
            if col in df.columns:
                mask_matrix[:, i] = df[col].values.astype(np.float32)

        feat_matrix = np.concatenate([norm_matrix, mask_matrix], axis=1)
        seq_len = len(feat_matrix)

        tensor_x = torch.tensor(feat_matrix, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 13)
        lengths = torch.tensor([seq_len], dtype=torch.long)

        with torch.no_grad():
            pred_ctdna, pred_prog_logits = self.model(tensor_x, lengths)
            ctdna_val = float(pred_ctdna.item())
            prog_prob = float(torch.sigmoid(pred_prog_logits).item())
            prog_class = int(prog_prob >= 0.5)

        return {
            'predicted_ctDNA_30d_vaf': round(max(0.0, ctdna_val), 4),
            'predicted_progression_risk': round(prog_prob, 4),
            'predicted_progression': prog_class,
            'input_sequence_length': seq_len,
            'max_days_from_baseline': int(df['days_from_baseline'].max()),
            'disclaimer': config.MANDATORY_DISCLAIMER
        }
