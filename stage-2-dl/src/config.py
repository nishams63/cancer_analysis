"""
Configuration module for Stage 2 Deep Learning Data Engineering (Version 2).

DISCLAIMER:
Synthetic data created for deep-learning research and pipeline prototyping.
These data are not real patient records and are not clinically validated.
Synthetic data != Real patient evidence != Clinical validation.
"""

from pathlib import Path

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Version 2 Data Directory Structure
DATA_V2_DIR = DATA_DIR / "v2"
RAW_DATA_DIR = DATA_V2_DIR / "raw"
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
RAW_BIOMARKERS_DIR = RAW_DATA_DIR / "biomarkers"

PROCESSED_DATA_DIR = DATA_V2_DIR / "processed"
PROCESSED_TILES_DIR = PROCESSED_DATA_DIR / "pathology_tiles"
IMAGE_METADATA_PATH = PROCESSED_DATA_DIR / "image_metadata.csv"
BIOMARKERS_PROCESSED_PATH = PROCESSED_DATA_DIR / "biomarkers_processed.csv"

SPLITS_DIR = DATA_V2_DIR / "splits"
TRAIN_PATIENTS_PATH = SPLITS_DIR / "train_patients.csv"
VAL_PATIENTS_PATH = SPLITS_DIR / "validation_patients.csv"
TEST_PATIENTS_PATH = SPLITS_DIR / "test_patients.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "data_engineering_report.md"
DATA_DICTIONARY_PATH = REPORTS_DIR / "data_dictionary.md"
DATASET_STATISTICS_PATH = REPORTS_DIR / "dataset_statistics.csv"
AUGMENTATION_QC_PATH = REPORTS_DIR / "augmentation_samples_comparison.png"

# Reproducibility
RANDOM_SEED = 42

# Cohort & Dataset Specifications (v2)
NUM_PATIENTS = 1000
NUM_TRAIN = 700
NUM_VAL = 150
NUM_TEST = 150

# Pathology Tile Specs (v2)
TILE_SIZE = (224, 224)  # Height, Width
CHANNELS = 3            # RGB
TILES_PER_PATIENT = 12  # Total 12,000 tiles (1,000 pts * 12 tiles)
TOTAL_TILES = 12000
SLIDES_PER_PATIENT = 2  # 6 tiles per slide
PATHOLOGY_CLASSES = ["benign", "malignant", "inflammation"]

# Longitudinal Biomarker Specs (v2)
MIN_TIMEPOINTS = 14
MAX_TIMEPOINTS = 18
FORECAST_SPLIT_DAY = 90  # Input window: <= 90 days, Prediction window: > 90 days
TARGET_FORECAST_DAYS = 30  # 30-day forward prediction target
INTERVAL_DAYS_BASE = 13   # ~Bi-weekly schedule across ~190-230 days
MISSINGNESS_MIN_RATE = 0.03  # 3% minimum missingness
MISSINGNESS_MAX_RATE = 0.08  # 8% maximum missingness

# Trajectory Archetypes
TRAJECTORY_ARCHETYPES = [
    "stable",
    "gradual_increase",
    "gradual_decrease",
    "fluctuating",
    "rapid_increase",
]

# Mandatory Disclaimer
CLINICAL_DISCLAIMER = (
    "Synthetic data created for deep-learning research and pipeline prototyping. "
    "These data are not real patient records and are not clinically validated. "
    "Synthetic data != Real patient evidence != Clinical validation."
)
