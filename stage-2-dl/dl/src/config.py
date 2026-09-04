"""
Stage 2 Deep Learning (DL) - Configuration Module
"""
from pathlib import Path

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
DL_DIR = SRC_DIR.parent
PROJECT_ROOT = DL_DIR.parent.parent
STAGE_2_DIR = PROJECT_ROOT / 'stage-2-dl'
DATA_ENG_DATA_V2 = STAGE_2_DIR / 'data-engineering' / 'data' / 'v2'

# Source Data Paths (Read-Only)
IMAGE_METADATA_PATH = DATA_ENG_DATA_V2 / 'processed' / 'image_metadata.csv'
BIOMARKERS_PATH = DATA_ENG_DATA_V2 / 'processed' / 'biomarkers_processed.csv'
PATHOLOGY_TILES_DIR = DATA_ENG_DATA_V2 / 'processed' / 'pathology_tiles'

TRAIN_SPLIT_PATH = DATA_ENG_DATA_V2 / 'splits' / 'train_patients.csv'
VAL_SPLIT_PATH = DATA_ENG_DATA_V2 / 'splits' / 'validation_patients.csv'
TEST_SPLIT_PATH = DATA_ENG_DATA_V2 / 'splits' / 'test_patients.csv'

# Output Paths
CHECKPOINTS_DIR = DL_DIR / 'checkpoints'
FIGURES_DIR = DL_DIR / 'figures'
REPORTS_DIR = DL_DIR / 'reports'

BEST_IMAGE_CHECKPOINT = CHECKPOINTS_DIR / 'best_pathology_cnn.pt'
BEST_TEMPORAL_CHECKPOINT = CHECKPOINTS_DIR / 'best_temporal_lstm.pt'
DL_REPORT_PATH = REPORTS_DIR / 'dl_report.md'

# Reproducibility
RANDOM_SEED = 42

# Cohort Partitions & Classes
SPLITS = ['train', 'validation', 'test']
CLASSES = ['benign', 'malignant', 'inflammation']

# Image Classification Configuration
NUM_CLASSES = 3
IMAGE_SIZE = (224, 224)
CLASS_TO_IDX = {'benign': 0, 'malignant': 1, 'inflammation': 2}
IDX_TO_CLASS = {0: 'benign', 1: 'malignant', 2: 'inflammation'}

# Normalization constants
# Measured empirical dataset statistics from EDA
DATASET_MEAN = [0.8422, 0.7340, 0.8337]
DATASET_STD = [0.1928, 0.2293, 0.1498]

# Standard ImageNet statistics (for pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Temporal Modeling Configuration
FORECAST_SPLIT_DAY = 90

# Feature definitions
# Verified: ctDNA_velocity_30d is strictly computed from (t - (t-1)) / delta_t (backward-looking), zero at t=0
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
NUM_TEMPORAL_FEATURES = len(ALL_TEMPORAL_FEATURES)  # 13 features

# Target definitions from Data Dictionary
TARGET_REGRESSION = 'future_ctDNA_30d_target'
TARGET_CLASSIFICATION = 'future_progression_trend'

# Clinical / Synthetic Disclaimers
MANDATORY_DISCLAIMER = (
    "This model was developed using synthetic data and has not been clinically validated. "
    "Performance on this dataset does not establish clinical effectiveness, safety, or real-world patient benefit."
)
EQUIVALENCE_DISCLAIMER = "Synthetic data != Real patient evidence != Clinical validation"
