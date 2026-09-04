"""
Master End-to-End Pipeline Runner for Stage 2 Deep Learning (Version 2).

Executes the complete v2 data engineering workflow:
  1. Patient cohort generation (1,000 patients) & zero-leakage split creation (700/150/150)
  2. Procedural synthesis of 12,000 anti-shortcut H&E pathology tiles & ~16,000 longitudinal biomarker records
  3. Image standardization, QA filtering, and class organization into data/v2/processed/pathology_tiles/
  4. Biomarker sequence preparation, chronological verification, and velocity feature engineering
  5. Training-only augmentation visual preview export
  6. Comprehensive automated validation suite, statistics export, and audit report generation

Usage:
  python stage-2-dl/src/pipeline.py

DISCLAIMER:
Synthetic data created for deep-learning research and pipeline prototyping.
These data are not real patient records and are not clinically validated.
Synthetic data != Real patient evidence != Clinical validation.
"""

import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

try:
    from .augmentation import export_sample_augmentations
    from .biomarker_preparation import prepare_longitudinal_biomarkers_v2
    from .config import CLINICAL_DISCLAIMER
    from .create_splits import create_patient_splits
    from .data_generation import run_data_generation_v2
    from .image_preparation import prepare_pathology_tiles_v2
    from .validation import DataValidatorV2
except ImportError:
    from augmentation import export_sample_augmentations
    from biomarker_preparation import prepare_longitudinal_biomarkers_v2
    from config import CLINICAL_DISCLAIMER
    from create_splits import create_patient_splits
    from data_generation import run_data_generation_v2
    from image_preparation import prepare_pathology_tiles_v2
    from validation import DataValidatorV2


def run_pipeline():
    start_time = time.time()
    print("=" * 75)
    print("STAGE 2 DEEP LEARNING DATA ENGINEERING PIPELINE (VERSION 2)")
    print("Project: Personalized Precision Medicine for Oncology Treatment")
    print("Target: 1,000 Patients | 12,000 Pathology Tiles | ~16,000 Temporal Records")
    print("=" * 75)
    print(f"NOTICE: {CLINICAL_DISCLAIMER}\n")

    print("[Step 1/6] Creating 1,000 patient splits with 0% patient leakage (700/150/150)...")
    create_patient_splits()

    print("\n[Step 2/6] Generating 12,000 raw visual tiles & ~16,000 temporal observations...")
    run_data_generation_v2()

    print("\n[Step 3/6] Standardizing and organizing 12,000 pathology tiles...")
    prepare_pathology_tiles_v2()

    print("\n[Step 4/6] Processing longitudinal temporal biomarker sequences with forecasting windows...")
    prepare_longitudinal_biomarkers_v2()

    print("\n[Step 5/6] Exporting training-only augmentation visual preview...")
    export_sample_augmentations()

    print("\n[Step 6/6] Executing comprehensive 19-point validation suite & exporting statistics...")
    validator = DataValidatorV2()
    passed = validator.run_all_checks()

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    if passed:
        print(f"STAGE 2 (v2) DATASET PIPELINE COMPLETED SUCCESSFULLY in {elapsed:.2f}s ({elapsed/60:.1f} min)!")
        print("Dataset is certified and ready for DL Engineer handoff.")
    else:
        print(f"STAGE 2 (v2) DATASET PIPELINE FAILED AUDIT in {elapsed:.2f}s!")
        sys.exit(1)
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
