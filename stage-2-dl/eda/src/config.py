"""
Stage 2 Exploratory Data Analysis (EDA) - Configuration Module
"""
from pathlib import Path

# Paths
SRC_DIR = Path(__file__).resolve().parent
EDA_DIR = SRC_DIR.parent
PROJECT_ROOT = EDA_DIR.parent.parent
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
REPORTS_DIR = EDA_DIR / 'reports'
FIGURES_DIR = EDA_DIR / 'figures'

IMAGE_STATS_CSV = REPORTS_DIR / 'image_statistics.csv'
TEMPORAL_STATS_CSV = REPORTS_DIR / 'temporal_statistics.csv'
SPLIT_STATS_CSV = REPORTS_DIR / 'split_statistics.csv'
FINDINGS_MD = REPORTS_DIR / 'findings.md'
EDA_REPORT_MD = REPORTS_DIR / 'eda_report.md'

# Reproducibility
RANDOM_SEED = 42

# Domain Constants
CLASSES = ['benign', 'malignant', 'inflammation']
SPLITS = ['train', 'validation', 'test']
BIOMARKERS = ['ctDNA_vaf_percent', 'cea_ng_ml', 'ca125_u_ml', 'ldh_u_l', 'crp_mg_l']
BIOMARKER_LABELS = {
    'ctDNA_vaf_percent': 'ctDNA VAF (%)',
    'cea_ng_ml': 'CEA (ng/mL)',
    'ca125_u_ml': 'CA-125 (U/mL)',
    'ldh_u_l': 'LDH (U/L)',
    'crp_mg_l': 'CRP (mg/L)',
}
MISSING_MASKS = {
    'ctDNA_vaf_percent': 'ctDNA_missing',
    'cea_ng_ml': 'cea_missing',
    'ca125_u_ml': 'ca125_missing',
    'ldh_u_l': 'ldh_missing',
    'crp_mg_l': 'crp_missing',
}
TRAJECTORY_PATTERNS = [
    'stable',
    'gradual_increase',
    'gradual_decrease',
    'fluctuating',
    'rapid_increase',
]

# Clinical / Synthetic Disclaimers
MANDATORY_DISCLAIMER = (
    'Synthetic histopathology-like images and biomarker trajectories '
    'generated for deep-learning pipeline prototyping; not clinically validated.'
)
EQUIVALENCE_DISCLAIMER = 'Synthetic data != Real patient evidence != Clinical validation'

# Plot Styling Palette
CLASS_COLORS = {
    'benign': '#2b83ba',       # Soft Blue
    'malignant': '#d7191c',    # Strong Red
    'inflammation': '#fdae61', # Amber / Gold
}

SPLIT_COLORS = {
    'train': '#2ca02c',        # Green
    'validation': '#1f77b4',   # Blue
    'test': '#ff7f0e',         # Orange
}

TRAJECTORY_COLORS = {
    'stable': '#2b83ba',
    'gradual_increase': '#fdae61',
    'gradual_decrease': '#abdda4',
    'fluctuating': '#9970ab',
    'rapid_increase': '#d7191c',
}
