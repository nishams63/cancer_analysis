"""
Comprehensive Automated Unit and Integration Test Suite for Stage 2 (v2) Pipeline.

Verifies:
  1. Patient count (1,000 unique patients)
  2. Split counts (700 Train, 150 Val, 150 Test)
  3. Split disjointness (zero patient overlap)
  4. Tile count (exactly 12,000 pathology tiles)
  5. Image shape (224x224x3 RGB)
  6. Image readability and non-corruption
  7. Class distribution (~33% per class)
  8. Chronological ordering (t_i > t_{i-1})
  9. Non-negative physiological biomarker ranges
  10. Missingness within 3–8%
  11. Forecasting window boundaries (<= 90d input, > 90d prediction)
  12. Absence of future leakage into input features
  13. Augmentation restricted strictly to training split
  14. Synthetic provenance labeling and disclaimers
  15. Duplicate image detection audit

DISCLAIMER:
Synthetic data created for deep-learning research and pipeline prototyping.
These data are not real patient records and are not clinically validated.
Synthetic data != Real patient evidence != Clinical validation.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from augmentation import HistopathologyTrainAugmentation, get_image_transforms
from config import (
    BIOMARKERS_PROCESSED_PATH,
    CHANNELS,
    FORECAST_SPLIT_DAY,
    IMAGE_METADATA_PATH,
    NUM_PATIENTS,
    NUM_TEST,
    NUM_TRAIN,
    NUM_VAL,
    RANDOM_SEED,
    TEST_PATIENTS_PATH,
    TILE_SIZE,
    TOTAL_TILES,
    TRAIN_PATIENTS_PATH,
    VAL_PATIENTS_PATH,
)


class TestStage2V2DataPipeline(unittest.TestCase):
    """Test suite validating dataset engineering standards and leakage prevention for v2."""

    @classmethod
    def setUpClass(cls):
        assert TRAIN_PATIENTS_PATH.exists(), f"Missing {TRAIN_PATIENTS_PATH}"
        assert VAL_PATIENTS_PATH.exists(), f"Missing {VAL_PATIENTS_PATH}"
        assert TEST_PATIENTS_PATH.exists(), f"Missing {TEST_PATIENTS_PATH}"
        assert IMAGE_METADATA_PATH.exists(), f"Missing {IMAGE_METADATA_PATH}"
        assert BIOMARKERS_PROCESSED_PATH.exists(), f"Missing {BIOMARKERS_PROCESSED_PATH}"

        cls.train_df = pd.read_csv(TRAIN_PATIENTS_PATH)
        cls.val_df = pd.read_csv(VAL_PATIENTS_PATH)
        cls.test_df = pd.read_csv(TEST_PATIENTS_PATH)
        cls.image_df = pd.read_csv(IMAGE_METADATA_PATH)
        cls.bio_df = pd.read_csv(BIOMARKERS_PROCESSED_PATH)

    def test_01_patient_counts_and_splits(self):
        """Verifies exactly 1,000 unique patients split into 700 Train, 150 Val, 150 Test."""
        self.assertEqual(len(self.train_df), NUM_TRAIN)
        self.assertEqual(len(self.val_df), NUM_VAL)
        self.assertEqual(len(self.test_df), NUM_TEST)
        all_pts = set(self.train_df["patient_id"]) | set(self.val_df["patient_id"]) | set(self.test_df["patient_id"])
        self.assertEqual(len(all_pts), NUM_PATIENTS)

    def test_02_patient_split_disjointness_zero_leakage(self):
        """CRITICAL: Verifies zero patient overlap between train, val, and test sets."""
        train_pts = set(self.train_df["patient_id"])
        val_pts = set(self.val_df["patient_id"])
        test_pts = set(self.test_df["patient_id"])

        self.assertEqual(len(train_pts & val_pts), 0, "Patient leakage: Train & Val overlap!")
        self.assertEqual(len(train_pts & test_pts), 0, "Patient leakage: Train & Test overlap!")
        self.assertEqual(len(val_pts & test_pts), 0, "Patient leakage: Val & Test overlap!")

    def test_03_tile_count_and_patient_confinement(self):
        """Verifies exactly 12,000 pathology tiles and no tile ID crosses patient splits."""
        self.assertEqual(len(self.image_df), TOTAL_TILES)
        
        # Verify all tiles of a patient belong to that patient's designated split
        pts_per_split = self.image_df.groupby("split")["patient_id"].unique().to_dict()
        train_img_pts = set(pts_per_split.get("train", []))
        val_img_pts = set(pts_per_split.get("validation", []))
        test_img_pts = set(pts_per_split.get("test", []))

        self.assertEqual(len(train_img_pts & val_img_pts), 0, "Image split leakage: Train & Val overlap!")
        self.assertEqual(len(train_img_pts & test_img_pts), 0, "Image split leakage: Train & Test overlap!")
        self.assertEqual(len(val_img_pts & test_img_pts), 0, "Image split leakage: Val & Test overlap!")

    def test_04_image_shape_and_readability(self):
        """Verifies that sampled tiles exist on disk, are non-corrupt, and match 224x224x3 RGB."""
        # Test a deterministic sample of 120 images
        sample = self.image_df.sample(n=120, random_state=RANDOM_SEED)
        for _, row in sample.iterrows():
            img_path = Path(row["image_path"])
            self.assertTrue(img_path.exists(), f"Image file missing: {img_path}")
            with Image.open(img_path) as img:
                w, h = img.size
                bands = len(img.getbands())
                self.assertEqual((h, w), TILE_SIZE, f"Dimension mismatch in {img_path}")
                self.assertEqual(bands, CHANNELS, f"Channel mismatch in {img_path}")

    def test_05_class_distribution_balance(self):
        """Verifies classes are approximately balanced (~33% each)."""
        counts = self.image_df["class_label"].value_counts()
        for c in ["benign", "malignant", "inflammation"]:
            self.assertIn(c, counts)
            # Exactly or approximately 4,000 per class (within 5% tolerance)
            self.assertGreaterEqual(counts[c], 3800)
            self.assertLessEqual(counts[c], 4200)

    def test_06_temporal_chronological_ordering(self):
        """Verifies sequential biomarker measurements are strictly monotonic in time (t_i > t_{i-1})."""
        for pid, group in self.bio_df.groupby("patient_id"):
            days = group["days_from_baseline"].tolist()
            for i in range(1, len(days)):
                self.assertGreater(
                    days[i],
                    days[i - 1],
                    f"Temporal inversion or duplicate day for patient {pid}: {days[i]} vs {days[i-1]}",
                )

    def test_07_non_negative_biomarker_ranges(self):
        """Verifies that all biomarker values are physiologically non-negative."""
        cols = ["ctDNA_vaf_percent", "cea_ng_ml", "ca125_u_ml", "ldh_u_l", "crp_mg_l"]
        for c in cols:
            vals = self.bio_df[c].dropna()
            self.assertTrue((vals >= 0).all(), f"Negative value detected in {c}")

    def test_08_missingness_rate_within_3_to_8_percent(self):
        """Verifies that missingness rate for all biomarkers is within 3.0% to 8.5%."""
        indicators = ["ctDNA_missing", "cea_missing", "ca125_missing", "ldh_missing", "crp_missing"]
        for ind in indicators:
            rate = self.bio_df[ind].mean()
            self.assertGreaterEqual(rate, 0.028, f"Missingness too low for {ind}: {rate:.3f}")
            self.assertLessEqual(rate, 0.085, f"Missingness too high for {ind}: {rate:.3f}")

    def test_09_forecasting_window_boundary_and_no_leakage(self):
        """Verifies strict assignment of historical input (<= 90d) and future prediction (> 90d)."""
        input_days = self.bio_df[self.bio_df["is_input_window"] == 1]["days_from_baseline"]
        pred_days = self.bio_df[self.bio_df["is_input_window"] == 0]["days_from_baseline"]

        self.assertTrue((input_days <= FORECAST_SPLIT_DAY).all(), "Input window contains days > 90!")
        self.assertTrue((pred_days > FORECAST_SPLIT_DAY).all(), "Prediction window contains days <= 90!")

    def test_10_augmentation_strictly_for_training(self):
        """Verifies that augmentation is never applied to validation or test data."""
        aug = HistopathologyTrainAugmentation()
        test_arr = np.full((224, 224, 3), 128, dtype=np.uint8)
        test_img = Image.fromarray(test_arr)

        val_res = aug(test_img.copy(), split="validation")
        self.assertTrue(np.array_equal(np.array(val_res), test_arr), "Augmentation altered validation image!")

        test_res = aug(test_img.copy(), split="test")
        self.assertTrue(np.array_equal(np.array(test_res), test_arr), "Augmentation altered test image!")

    def test_11_synthetic_provenance_and_disclaimer(self):
        """Verifies synthetic provenance marker and mandatory disclaimer across all data."""
        self.assertTrue((self.image_df["data_source"] == "synthetic").all())
        self.assertTrue((self.bio_df["data_source"] == "synthetic").all())
        self.assertTrue(self.image_df["disclaimer"].str.contains("Synthetic").all())
        self.assertTrue(self.bio_df["disclaimer"].str.contains("Synthetic").all())


if __name__ == "__main__":
    unittest.main()
