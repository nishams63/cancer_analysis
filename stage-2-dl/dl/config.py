"""
Stage 2 Deep Learning - Central Experiment & Hyperparameter Configuration
Structured with dataclasses for reproducibility, validation, and multi-seed experimentation.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

# Root Directory Anchors
BASE_DIR = Path(__file__).resolve().parent
STAGE_2_DIR = BASE_DIR.parent
PROJECT_ROOT = STAGE_2_DIR.parent

# Data Engineering Paths
DATA_V2_DIR = STAGE_2_DIR / 'data-engineering' / 'data' / 'v2'
PROCESSED_DATA_DIR = DATA_V2_DIR / 'processed'
SPLITS_DIR = DATA_V2_DIR / 'splits'

IMAGE_METADATA_PATH = PROCESSED_DATA_DIR / 'image_metadata.csv'
BIOMARKERS_PATH = PROCESSED_DATA_DIR / 'biomarkers_processed.csv'
PATHOLOGY_TILES_DIR = PROCESSED_DATA_DIR / 'pathology_tiles'

TRAIN_SPLIT_PATH = SPLITS_DIR / 'train_patients.csv'
VAL_SPLIT_PATH = SPLITS_DIR / 'validation_patients.csv'
TEST_SPLIT_PATH = SPLITS_DIR / 'test_patients.csv'

# Output Artifacts Directories
CHECKPOINTS_DIR = BASE_DIR / 'checkpoints'
RESULTS_DIR = STAGE_2_DIR / 'results'
DOCS_DIR = STAGE_2_DIR / 'docs'
FIGURES_DIR = BASE_DIR / 'figures'

# Ensure required directories exist
for d in [CHECKPOINTS_DIR, RESULTS_DIR, DOCS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Frozen Baseline Checkpoints
BASELINE_IMAGE_CHECKPOINT = CHECKPOINTS_DIR / 'best_pathology_cnn.pt'
BASELINE_TEMPORAL_CHECKPOINT = CHECKPOINTS_DIR / 'best_temporal_lstm.pt'

# Upgraded Checkpoint Paths
CHECKPOINT_RESNET18 = CHECKPOINTS_DIR / 'best_resnet18.pt'
CHECKPOINT_RESNET50 = CHECKPOINTS_DIR / 'best_resnet50.pt'
CHECKPOINT_EFFICIENTNET = CHECKPOINTS_DIR / 'best_efficientnet_b0.pt'
CHECKPOINT_VIT = CHECKPOINTS_DIR / 'best_vit.pt'
CHECKPOINT_ATTENTION_MIL = CHECKPOINTS_DIR / 'best_attention_mil.pt'
CHECKPOINT_TEMPORAL_TRANSFORMER = CHECKPOINTS_DIR / 'best_temporal_transformer.pt'
CHECKPOINT_MULTIMODAL_FUSION = CHECKPOINTS_DIR / 'best_multimodal_fusion.pt'

# Global Evaluation Seeds
EXPERIMENT_SEEDS: List[int] = [42, 123, 2026]
DEFAULT_SEED: int = 42

# Cohort & Class Definitions
SPLITS = ['train', 'validation', 'test']
CLASSES = ['benign', 'malignant', 'inflammation']
NUM_CLASSES = 3
CLASS_TO_IDX = {'benign': 0, 'malignant': 1, 'inflammation': 2}
IDX_TO_CLASS = {0: 'benign', 1: 'malignant', 2: 'inflammation'}

# Normalization Constants
DATASET_MEAN = [0.8422, 0.7340, 0.8337]
DATASET_STD = [0.1928, 0.2293, 0.1498]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Temporal Feature Constants
FORECAST_SPLIT_DAY = 90
TEMPORAL_NUMERICAL_FEATURES = [
    'ctDNA_vaf_percent',
    'cea_ng_ml',
    'ca125_u_ml',
    'ldh_u_l',
    'crp_mg_l',
    'ctDNA_velocity_30d',
    'delta_days',
    'days_from_baseline',
]
TEMPORAL_MASK_FEATURES = [
    'ctDNA_missing',
    'cea_missing',
    'ca125_missing',
    'ldh_missing',
    'crp_missing',
]
ALL_TEMPORAL_FEATURES = TEMPORAL_NUMERICAL_FEATURES + TEMPORAL_MASK_FEATURES
NUM_TEMPORAL_FEATURES = len(ALL_TEMPORAL_FEATURES)  # 13

TARGET_REGRESSION = 'future_ctDNA_30d_target'
TARGET_CLASSIFICATION = 'future_progression_trend'

# Scientific Disclaimers
MANDATORY_DISCLAIMER = (
    "Research prototype evaluated on synthetic data. "
    "Not clinically validated; not for medical decision-making or patient care."
)


@dataclass
class PathologyConfig:
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_classes: int = 3
    num_tiles_per_patient: int = 12
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 4
    dropout: float = 0.3
    feature_dim: int = 128
    models: List[str] = field(default_factory=lambda: ['resnet18', 'resnet50', 'efficientnet_b0', 'vit'])


@dataclass
class AttentionMILConfig:
    num_tiles: int = 12
    in_features: int = 512  # Backbone feature embedding dimension
    hidden_dim: int = 128
    attention_dim: int = 64
    dropout: float = 0.25
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    batch_size: int = 16


@dataclass
class TemporalTransformerConfig:
    input_dim: int = NUM_TEMPORAL_FEATURES
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.2
    max_seq_len: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    lambda_cls: float = 1.0  # Configurable multi-task weight
    epochs: int = 8
    batch_size: int = 32


@dataclass
class MultimodalFusionConfig:
    pathology_dim: int = 3  # Or latent embedding dim
    temporal_dim: int = 64
    fused_dim: int = 64
    hidden_dim: int = 64
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 6
    batch_size: int = 32
    lambda_cls: float = 1.0
    # Fixed linear baseline weights
    baseline_w_mal: float = 0.35
    baseline_w_prog: float = 0.40
    baseline_w_vaf: float = 0.25


@dataclass
class ChallengeConfig:
    """Parameters for realistic Dataset V2 perturbations"""
    # Pathology perturbations
    brightness_delta: float = 0.25
    contrast_factor: float = 0.60
    blur_kernel_size: int = 5
    noise_std: float = 0.08
    jpeg_quality: int = 40
    # Temporal perturbations
    measurement_noise_scale: float = 0.15
    visit_drop_rate: float = 0.33
    feature_drop_rate: float = 0.25
    outlier_prob: float = 0.05
    outlier_scale: float = 3.0


# Instantiate default configurations
PATHOLOGY_CONFIG = PathologyConfig()
ATTENTION_MIL_CONFIG = AttentionMILConfig()
TEMPORAL_TRANSFORMER_CONFIG = TemporalTransformerConfig()
MULTIMODAL_FUSION_CONFIG = MultimodalFusionConfig()
CHALLENGE_CONFIG = ChallengeConfig()
