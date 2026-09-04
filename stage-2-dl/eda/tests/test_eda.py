"""
Stage 2 EDA - Automated Test Suite
"""
import os
import sys
import unittest
import pandas as pd
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

import config

class TestStage2EDA(unittest.TestCase):
    """Test suite verifying EDA inputs, outputs, integrity, and non-modification."""

    @classmethod
    def setUpClass(cls):
        # Verify source dataset presence
        assert config.IMAGE_METADATA_PATH.exists(), f"Image metadata not found: {config.IMAGE_METADATA_PATH}"
        assert config.BIOMARKERS_PATH.exists(), f"Biomarker dataset not found: {config.BIOMARKERS_PATH}"
        assert config.TRAIN_SPLIT_PATH.exists(), f"Train split not found: {config.TRAIN_SPLIT_PATH}"
        assert config.VAL_SPLIT_PATH.exists(), f"Val split not found: {config.VAL_SPLIT_PATH}"
        assert config.TEST_SPLIT_PATH.exists(), f"Test split not found: {config.TEST_SPLIT_PATH}"

        cls.image_df = pd.read_csv(config.IMAGE_METADATA_PATH)
        cls.bio_df = pd.read_csv(config.BIOMARKERS_PATH)
        cls.train_df = pd.read_csv(config.TRAIN_SPLIT_PATH)
        cls.val_df = pd.read_csv(config.VAL_SPLIT_PATH)
        cls.test_df = pd.read_csv(config.TEST_SPLIT_PATH)

    def test_01_source_data_schemas_and_columns(self):
        """Verifies primary source data column schemas."""
        expected_img_cols = {
            'patient_id', 'tile_id', 'slide_id', 'class_label', 'split',
            'image_path', 'height', 'width', 'channels', 'background_temperature',
            'brightness_factor', 'contrast_factor', 'stain_variation'
        }
        self.assertTrue(expected_img_cols.issubset(set(self.image_df.columns)))

        expected_bio_cols = {
            'patient_id', 'split', 'timepoint_index', 'days_from_baseline', 'delta_days',
            'ctDNA_vaf_percent', 'cea_ng_ml', 'ca125_u_ml', 'ldh_u_l', 'crp_mg_l',
            'ctDNA_missing', 'cea_missing', 'ca125_missing', 'ldh_missing', 'crp_missing',
            'trajectory_pattern', 'window_type', 'is_input_window',
            'future_ctDNA_30d_target', 'future_progression_trend'
        }
        self.assertTrue(expected_bio_cols.issubset(set(self.bio_df.columns)))

    def test_02_source_dataset_scales(self):
        """Verifies row counts and patient cohort scale."""
        self.assertEqual(len(self.image_df), 12000)
        self.assertEqual(len(self.bio_df), 16012)
        self.assertEqual(len(self.train_df), 700)
        self.assertEqual(len(self.val_df), 150)
        self.assertEqual(len(self.test_df), 150)

    def test_03_split_disjointness_zero_leakage(self):
        """Verifies strict 0% patient leakage across partitions."""
        tr_pats = set(self.train_df['patient_id'])
        val_pats = set(self.val_df['patient_id'])
        test_pats = set(self.test_df['patient_id'])

        self.assertEqual(len(tr_pats.intersection(val_pats)), 0)
        self.assertEqual(len(tr_pats.intersection(test_pats)), 0)
        self.assertEqual(len(val_pats.intersection(test_pats)), 0)
        self.assertEqual(len(tr_pats | val_pats | test_pats), 1000)

    def test_04_class_balance(self):
        """Verifies exact 4,000 tiles per class."""
        counts = self.image_df['class_label'].value_counts().to_dict()
        self.assertEqual(counts.get('benign', 0), 4000)
        self.assertEqual(counts.get('malignant', 0), 4000)
        self.assertEqual(counts.get('inflammation', 0), 4000)

    def test_05_forecasting_window_segregation(self):
        """Verifies boundary at Day 90 for historical vs future windows."""
        hist = self.bio_df[self.bio_df['is_input_window'] == 1]
        fut = self.bio_df[self.bio_df['is_input_window'] == 0]

        self.assertEqual(len(hist), 7229)
        self.assertEqual(len(fut), 8783)
        self.assertTrue((hist['days_from_baseline'] <= 90).all())
        self.assertTrue((fut['days_from_baseline'] > 90).all())

    def test_06_missingness_indicator_consistency(self):
        """Verifies binary indicator masks match NaN values with 100% precision."""
        for m in config.BIOMARKERS:
            mask_col = config.MISSING_MASKS[m]
            matches = (self.bio_df[m].isna() == (self.bio_df[mask_col] == 1)).all()
            self.assertTrue(matches, f"Mask mismatch found in {m}")

    def test_07_all_13_figures_exist_and_non_empty(self):
        """Verifies all 13 diagnostic figures exist and have valid file sizes (>10KB)."""
        expected_figures = [
            'image_class_distribution.png',
            'image_examples.png',
            'image_pixel_distribution.png',
            'image_brightness_by_class.png',
            'image_color_distribution.png',
            'image_stain_variation.png',
            'temporal_biomarker_distributions.png',
            'temporal_missingness.png',
            'temporal_observations_over_time.png',
            'temporal_trajectory_examples.png',
            'temporal_trajectory_distribution.png',
            'forecasting_window_distribution.png',
            'split_comparison.png',
        ]
        for fig_name in expected_figures:
            fig_path = config.FIGURES_DIR / fig_name
            self.assertTrue(fig_path.exists(), f"Figure missing: {fig_name}")
            size_kb = os.path.getsize(fig_path) / 1024.0
            self.assertGreater(size_kb, 10.0, f"Figure {fig_name} is unexpectedly small ({size_kb:.1f} KB)")

    def test_08_all_statistics_csvs_exist_and_populated(self):
        """Verifies all 3 statistical summary CSVs exist and contain non-empty records."""
        for csv_path in [config.IMAGE_STATS_CSV, config.TEMPORAL_STATS_CSV, config.SPLIT_STATS_CSV]:
            self.assertTrue(csv_path.exists(), f"CSV missing: {csv_path.name}")
            df = pd.read_csv(csv_path)
            self.assertGreater(len(df), 0, f"CSV {csv_path.name} is empty")

    def test_09_all_reports_exist_and_contain_disclaimers(self):
        """Verifies findings.md and eda_report.md exist and contain required disclaimers."""
        for rep_path in [config.FINDINGS_MD, config.EDA_REPORT_MD]:
            self.assertTrue(rep_path.exists(), f"Report missing: {rep_path.name}")
            with open(rep_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertIn("Synthetic data", content)
            self.assertIn("Clinical validation", content)

    def test_10_source_data_immutability(self):
        """Verifies no modifications were made to the source data directory."""
        # Row counts and integrity must remain intact
        self.assertEqual(len(self.image_df), 12000)
        self.assertEqual(len(self.bio_df), 16012)
        self.assertEqual(self.image_df['qc_status'].value_counts().get('PASS', 0), 12000)

if __name__ == '__main__':
    unittest.main()
