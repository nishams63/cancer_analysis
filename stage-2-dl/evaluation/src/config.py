"""
Stage 2 Deep Learning - Independent Locked Test Evaluation Configuration
"""
from pathlib import Path
import torch

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
EVAL_DIR = SRC_DIR.parent
STAGE_2_DIR = EVAL_DIR.parent
PROJECT_ROOT = STAGE_2_DIR.parent

# Data Engineering Source Paths (Read-Only)
DATA_ENG_V2 = STAGE_2_DIR / 'data-engineering' / 'data' / 'v2'
IMAGE_METADATA_PATH = DATA_ENG_V2 / 'processed' / 'image_metadata.csv'
BIOMARKERS_PATH = DATA_ENG_V2 / 'processed' / 'biomarkers_processed.csv'
PATHOLOGY_TILES_DIR = DATA_ENG_V2 / 'processed' / 'pathology_tiles'

TRAIN_SPLIT_PATH = DATA_ENG_V2 / 'splits' / 'train_patients.csv'
VAL_SPLIT_PATH = DATA_ENG_V2 / 'splits' / 'validation_patients.csv'
TEST_SPLIT_PATH = DATA_ENG_V2 / 'splits' / 'test_patients.csv'

# Frozen DL Checkpoints (Read-Only)
DL_DIR = STAGE_2_DIR / 'dl'
CHECKPOINTS_DIR = DL_DIR / 'checkpoints'
FROZEN_IMAGE_CHECKPOINT = CHECKPOINTS_DIR / 'best_pathology_cnn.pt'
FROZEN_TEMPORAL_CHECKPOINT = CHECKPOINTS_DIR / 'best_temporal_lstm.pt'

# Evaluation Output Directories
FIGURES_DIR = EVAL_DIR / 'figures'
REPORTS_DIR = EVAL_DIR / 'reports'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Image Modality Configuration
IMAGE_SIZE = (224, 224)
CLASSES = ['benign', 'malignant', 'inflammation']
NUM_CLASSES = 3
CLASS_TO_IDX = {'benign': 0, 'malignant': 1, 'inflammation': 2}
IDX_TO_CLASS = {0: 'benign', 1: 'malignant', 2: 'inflammation'}

# Standard ImageNet normalization (used by ResNet-18)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Temporal Modality Configuration
FORECAST_SPLIT_DAY = 90
TARGET_REGRESSION = 'future_ctDNA_30d_target'
TARGET_CLASSIFICATION = 'future_progression_trend'

def get_checkpoint_temporal_features(ckpt_path: Path = FROZEN_TEMPORAL_CHECKPOINT):
    """Dynamically extracts features from checkpoint to avoid hardcoding."""
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        return ckpt.get('temporal_features', [])
    return [
        'ctDNA_vaf_percent', 'cea_ng_ml', 'ca125_u_ml', 'ldh_u_l', 'crp_mg_l',
        'ctDNA_velocity_30d', 'delta_days', 'days_from_baseline',
        'ctDNA_missing', 'cea_missing', 'ca125_missing', 'ldh_missing', 'crp_missing'
    ]

# Mandatory Disclaimers
MANDATORY_DISCLAIMER = (
    "This model was developed using synthetic data and has not been clinically validated. "
    "Performance on this dataset does not establish clinical effectiveness, safety, or real-world patient benefit."
)
EQUIVALENCE_DISCLAIMER = "Synthetic data != Real patient evidence != Clinical validation"
RANDOM_SEED = 42
