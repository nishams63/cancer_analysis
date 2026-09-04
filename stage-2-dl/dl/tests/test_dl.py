"""
Stage 2 Deep Learning (DL) - Automated Unit Test Suite
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

import config
import utils
import datasets
import image_model
import temporal_model

class TestStage2DL(unittest.TestCase):
    """Test suite verifying DL datasets, architectures, leakage safeguards, and inference formats."""

    @classmethod
    def setUpClass(cls):
        utils.set_seed(config.RANDOM_SEED)

    def test_01_dataset_loading_and_patient_split_integrity(self):
        """Verifies dataset splits load strictly disjoint patient sets."""
        tr_ds = datasets.PathologyTileDataset(split='train')
        val_ds = datasets.PathologyTileDataset(split='validation')
        test_ds = datasets.PathologyTileDataset(split='test')

        self.assertEqual(len(tr_ds), 8400)
        self.assertEqual(len(val_ds), 1800)
        self.assertEqual(len(test_ds), 1800)

        tr_pats = set(tr_ds.df['patient_id'])
        val_pats = set(val_ds.df['patient_id'])
        test_pats = set(test_ds.df['patient_id'])

        self.assertEqual(len(tr_pats.intersection(val_pats)), 0)
        self.assertEqual(len(tr_pats.intersection(test_pats)), 0)
        self.assertEqual(len(val_pats.intersection(test_pats)), 0)

    def test_02_image_tensor_shape_and_classes(self):
        """Verifies image tensor shape is [3, 224, 224] and labels are in {0, 1, 2}."""
        val_ds = datasets.PathologyTileDataset(split='validation')
        img, label, meta = val_ds[0]

        self.assertEqual(img.shape, (3, 224, 224))
        self.assertIn(label, [0, 1, 2])
        self.assertIn(meta['class_label'], config.CLASSES)
        self.assertEqual(config.CLASS_TO_IDX[meta['class_label']], label)

    def test_03_class_mapping_consistency(self):
        """Verifies bidirectional class mapping."""
        for name, idx in config.CLASS_TO_IDX.items():
            self.assertEqual(config.IDX_TO_CLASS[idx], name)

    def test_04_temporal_sequence_construction_and_shapes(self):
        """Verifies sequence construction and tensor shapes."""
        train_temp = datasets.TemporalSequenceDataset(split='train', max_seq_len=10)
        self.assertEqual(len(train_temp), 700)

        sample = train_temp[0]
        feats = sample['features']
        length = sample['length']
        target_ctdna = sample['target_ctdna']
        target_prog = sample['target_progression']

        self.assertEqual(feats.shape, (10, config.NUM_TEMPORAL_FEATURES))
        self.assertGreater(length, 0)
        self.assertLessEqual(length, 10)
        self.assertTrue(torch.is_tensor(target_ctdna))
        self.assertTrue(torch.is_tensor(target_prog))

    def test_05_historical_window_filtering_anti_leakage(self):
        """Verifies historical window is strictly <= 90 days with zero future timestamps."""
        for split in ['train', 'validation', 'test']:
            temp_ds = datasets.TemporalSequenceDataset(split=split)
            # Inspect underlying dataframe inside dataset
            max_day = temp_ds.df['days_from_baseline'].max()
            self.assertLessEqual(max_day, config.FORECAST_SPLIT_DAY)
            self.assertTrue((temp_ds.df['is_input_window'] == 1).all())

    def test_06_temporal_norm_fitted_on_train_only(self):
        """Verifies that validation dataset uses training-derived normalization parameters."""
        tr_ds = datasets.TemporalSequenceDataset(split='train')
        val_ds = datasets.TemporalSequenceDataset(split='validation', norm_params=tr_ds.norm_params)

        for feat in config.TEMPORAL_NUMERICAL_FEATURES:
            tr_mean, tr_std = tr_ds.norm_params[feat]
            val_mean, val_std = val_ds.norm_params[feat]
            self.assertEqual(tr_mean, val_mean)
            self.assertEqual(tr_std, val_std)

    def test_07_image_model_forward_pass(self):
        """Verifies ResNet-18 and custom CNN forward pass output shapes."""
        # ResNet18
        m_resnet = image_model.build_image_model('resnet18', pretrained=False)
        m_resnet.eval()
        dummy_x = torch.randn(2, 3, 224, 224)
        out_res = m_resnet(dummy_x)
        self.assertEqual(out_res.shape, (2, config.NUM_CLASSES))

        # Custom CNN
        m_cnn = image_model.build_image_model('custom_cnn', pretrained=False)
        m_cnn.eval()
        out_cnn = m_cnn(dummy_x)
        self.assertEqual(out_cnn.shape, (2, config.NUM_CLASSES))

    def test_08_temporal_bilstm_forward_pass(self):
        """Verifies multi-task BiLSTM output shapes and variable length indexing."""
        m_lstm = temporal_model.build_temporal_model('lstm', input_dim=config.NUM_TEMPORAL_FEATURES, hidden_dim=32)
        m_lstm.eval()

        dummy_seq = torch.randn(3, 8, config.NUM_TEMPORAL_FEATURES)
        dummy_lengths = torch.tensor([5, 7, 8], dtype=torch.long)

        pred_ctdna, pred_prog_logits = m_lstm(dummy_seq, dummy_lengths)
        self.assertEqual(pred_ctdna.shape, (3,))
        self.assertEqual(pred_prog_logits.shape, (3,))

    def test_09_checkpoint_save_and_load(self):
        """Verifies checkpoint serialization and deserialization."""
        test_chk_path = config.CHECKPOINTS_DIR / 'test_chk.pt'
        dummy_state = {
            'epoch': 5,
            'model_architecture': 'test_arch',
            'state_dict': {'layer.weight': torch.tensor([1.0, 2.0])},
            'metric': 0.95
        }
        utils.save_checkpoint(dummy_state, str(test_chk_path))
        self.assertTrue(test_chk_path.exists())

        loaded = utils.load_checkpoint(str(test_chk_path))
        self.assertEqual(loaded['epoch'], 5)
        self.assertEqual(loaded['metric'], 0.95)
        self.assertTrue(torch.equal(loaded['state_dict']['layer.weight'], torch.tensor([1.0, 2.0])))

        # Clean up
        if test_chk_path.exists():
            os.remove(test_chk_path)

    def test_10_missing_value_mask_handling(self):
        """Verifies missingness masks are properly included as non-zero input features."""
        train_temp = datasets.TemporalSequenceDataset(split='train')
        sample = train_temp[0]
        feats = sample['features']
        # The last 5 channels are the missing masks
        mask_channels = feats[:, -5:]
        self.assertEqual(mask_channels.shape[1], 5)
        # Verify values are binary 0 or 1
        unique_vals = set(np.unique(mask_channels.numpy()))
        self.assertTrue(unique_vals.issubset({0.0, 1.0}))

if __name__ == '__main__':
    unittest.main()
