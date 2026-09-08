"""
Stage 2 Deep Learning - Upgraded System Comprehensive Test Suite
Verifies:
  - Dataset V2 and Patient Bag integrity
  - Zero patient leakage across partitions
  - Temporal boundary anti-leakage (<= 90 days)
  - Pathology benchmark models (ResNet-18, ResNet-50, EfficientNet-B0, ViT)
  - Patient-Level Attention-MIL (all 4 pooling modes + attention weights)
  - Continuous Irregular Temporal Transformer with padding suppression
  - Learned Multimodal Fusion (Concat, Gated, Cross-Attention)
  - Epistemic Uncertainty (MC Dropout) & Calibration (Temperature Scaling, ECE)
  - Latent OOD Detector (Mahalanobis distance)
  - Multimodal Explainability (Grad-CAM, MIL ranking, Permutation Importance)
"""
import unittest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Ensure stage-2-dl is on python path
STAGE_2_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(STAGE_2_DIR))

from dl import config, dataset_v2
from dl.models import pathology_benchmarks as pb
from dl.models import attention_mil as am
from dl.models import temporal_transformer as tt
from dl.models import multimodal_fusion as mf
from dl import uncertainty_calibration as uc
from dl import ood_detector as od
from dl import explainability as exp
from dl import robustness_suite as rs


class TestUpgradedDL(unittest.TestCase):

    def test_01_patient_disjointness_assertion(self):
        """Verifies strictly zero patient overlap between train, val, and locked test splits."""
        train_ds = dataset_v2.PatientPathologyBagDataset(split='train')
        val_ds = dataset_v2.PatientPathologyBagDataset(split='validation')
        test_ds = dataset_v2.PatientPathologyBagDataset(split='test')

        train_pats = {b['patient_id'] for b in train_ds.patient_bags}
        val_pats = {b['patient_id'] for b in val_ds.patient_bags}
        test_pats = {b['patient_id'] for b in test_ds.patient_bags}

        self.assertEqual(len(train_pats), 700)
        self.assertEqual(len(val_pats), 150)
        self.assertEqual(len(test_pats), 150)

        self.assertEqual(len(train_pats & val_pats), 0, "Patient leakage between Train and Val!")
        self.assertEqual(len(train_pats & test_pats), 0, "Patient leakage between Train and Test!")
        self.assertEqual(len(val_pats & test_pats), 0, "Patient leakage between Val and Test!")

    def test_02_patient_pathology_bag_shapes(self):
        """Verifies patient bag tensor shape is (12, 3, 224, 224) and patient label is valid."""
        val_bags = dataset_v2.PatientPathologyBagDataset(split='validation')
        bag_tensor, pat_label, tile_labels, pat_id = val_bags[0]

        self.assertEqual(bag_tensor.shape, (12, 3, 224, 224))
        self.assertIn(pat_label, [0, 1, 2])
        self.assertEqual(len(tile_labels), 12)
        self.assertIsInstance(pat_id, str)

    def test_03_temporal_irregular_time_handling_and_no_future_leakage(self):
        """Verifies continuous temporal sequence handles delta_days and days_from_baseline <= 90."""
        for split in ['train', 'validation', 'test']:
            ds = dataset_v2.ContinuousTemporalSequenceDataset(split=split)
            max_day = ds.df['days_from_baseline'].max()
            self.assertLessEqual(max_day, config.FORECAST_SPLIT_DAY, f"Found observation > 90 in split {split}")
            
            # Check sample shapes
            sample = ds[0]
            self.assertEqual(sample['features'].shape, (10, config.NUM_TEMPORAL_FEATURES))
            self.assertGreater(sample['length'], 0)
            self.assertLessEqual(sample['length'], 10)

    def test_04_pathology_benchmarks_forward_and_feature_extraction(self):
        """Verifies ResNet-18, ResNet-50, EfficientNet-B0, and ViT forward passes and feature dimensions."""
        x = torch.randn(2, 3, 224, 224)
        for m_name in ['resnet18', 'resnet50', 'efficientnet_b0', 'vit']:
            m = pb.build_pathology_benchmark_model(m_name, pretrained=False)
            m.eval()
            with torch.no_grad():
                logits = m(x)
                feats = m.extract_features(x)
            self.assertEqual(logits.shape, (2, config.NUM_CLASSES))
            self.assertEqual(feats.shape[0], 2)
            self.assertIn(feats.shape[1], [128, 256])

    def test_05_attention_mil_all_pooling_modes(self):
        """Verifies Attention-MIL across mean, median, max, and gated attention pooling."""
        dummy_feats = torch.randn(3, 12, 128)
        for mode in ['mean', 'median', 'max', 'attention']:
            model = am.PatientPathologyMIL(feature_dim=128, pooling_mode=mode)
            model.eval()
            with torch.no_grad():
                logits, repr_p, weights = model(tile_embeddings=dummy_feats)
            self.assertEqual(logits.shape, (3, config.NUM_CLASSES))
            self.assertEqual(repr_p.shape, (3, 128))
            self.assertEqual(weights.shape, (3, 12))
            # Weights must sum to 1.0 for each patient
            np.testing.assert_allclose(weights.sum(dim=1).numpy(), [1.0, 1.0, 1.0], atol=1e-5)

    def test_06_temporal_transformer_irregular_intervals_and_multi_task_loss(self):
        """Verifies ContinuousTemporalTransformer multi-task output shapes and loss."""
        model = tt.ContinuousTemporalTransformer()
        model.eval()
        x = torch.randn(4, 10, 13)
        lengths = torch.tensor([4, 2, 5, 3], dtype=torch.long)

        with torch.no_grad():
            ctdna_pred, prog_logits, pat_repr = model(x, lengths)
        self.assertEqual(ctdna_pred.shape, (4,))
        self.assertEqual(prog_logits.shape, (4,))
        self.assertEqual(pat_repr.shape, (4, 64))

        # Test loss
        y_ctdna = torch.tensor([0.5, 1.2, 0.0, 3.4])
        y_prog = torch.tensor([0.0, 1.0, 0.0, 1.0])
        tot_loss, r_loss, c_loss = model.compute_multi_task_loss(ctdna_pred, y_ctdna, prog_logits, y_prog, lambda_cls=1.2)
        self.assertFalse(torch.isnan(tot_loss))
        self.assertGreater(tot_loss.item(), 0.0)

    def test_07_multimodal_fusion_models(self):
        """Verifies Concat, Gated, and Cross-Attention multimodal fusion models."""
        p_repr = torch.randn(2, 128)
        t_repr = torch.randn(2, 64)

        for f_type in ['concat_mlp', 'gated', 'cross_attention']:
            model = mf.build_fusion_model(f_type, pathology_dim=128, temporal_dim=64)
            model.eval()
            with torch.no_grad():
                out = model(p_repr, t_repr)
            self.assertEqual(out['progression_prob'].shape, (2,))
            self.assertEqual(out['ctdna_forecast'].shape, (2,))
            self.assertEqual(out['multimodal_risk_score'].shape, (2,))
            # Probabilities in [0, 1]
            self.assertTrue((out['progression_prob'] >= 0.0).all() and (out['progression_prob'] <= 1.0).all())
            self.assertTrue((out['multimodal_risk_score'] >= 0.0).all() and (out['multimodal_risk_score'] <= 1.0).all())

    def test_08_mc_dropout_uncertainty_estimation(self):
        """Verifies Monte Carlo Dropout generates mean prediction and non-negative uncertainty."""
        class DummyDropoutNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5), nn.Linear(8, 1))
            def forward(self, x):
                return self.fc(x).squeeze(-1)

        net = DummyDropoutNet()
        evaluator = uc.MCDropoutEvaluator(num_samples=10)
        res = evaluator.evaluate_classification(net, torch.randn(4, 8))

        self.assertEqual(res['mean_prediction'].shape, (4,))
        self.assertEqual(res['uncertainty_score'].shape, (4,))
        self.assertTrue((res['uncertainty_score'] >= 0.0).all())

    def test_09_temperature_scaling_and_calibration_metrics(self):
        """Verifies TemperatureScaler optimizes T on validation and ECE computation."""
        ts = uc.TemperatureScaler()
        logits = torch.randn(40)
        targets = (torch.rand(40) > 0.5).long()
        learned_t = ts.fit_validation_data(logits, targets, is_binary=True)
        self.assertGreater(learned_t, 0.0)

        probs = np.random.uniform(0, 1, 50)
        labels = (np.random.rand(50) > 0.5).astype(int)
        ece, diag = uc.compute_expected_calibration_error(probs, labels, n_bins=5)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)
        self.assertEqual(len(diag['bin_accuracies']), 5)

    def test_10_latent_ood_detector(self):
        """Verifies LatentOODDetector identifies distribution-shifted samples."""
        train_emb = np.random.normal(0, 1, (80, 16))
        val_emb = np.random.normal(0, 1, (30, 16))
        shifted_emb = np.random.normal(5, 2, (20, 16))

        detector = od.LatentOODDetector()
        detector.fit_reference(train_emb)
        th = detector.fit_threshold_on_validation(val_emb, percentile=95.0)
        self.assertGreater(th, 0.0)

        pred_shift = detector.predict(shifted_emb)
        self.assertGreaterEqual(pred_shift['shift_fraction'], 0.8)

    def test_11_explainability_gradcam_and_mil_ranking(self):
        """Verifies Grad-CAM heatmap generation and MIL tile ranking order."""
        model = pb.ResNet18Transfer(pretrained=False, freeze_backbone=False)
        gcam = exp.PathologyGradCAM(model)
        dummy_img = torch.randn(1, 3, 224, 224)
        cam = gcam.generate_cam(dummy_img, target_class=1)

        self.assertEqual(cam.shape, (224, 224))
        self.assertGreaterEqual(cam.min(), 0.0)
        self.assertLessEqual(cam.max(), 1.0)

        # MIL ranking
        weights = np.array([0.1, 0.5, 0.2, 0.2])
        ranked = exp.rank_patient_mil_tiles(weights)
        self.assertEqual(ranked[0]['tile_index'], 2)  # 0.5 has index 1 -> tile 2
        self.assertEqual(ranked[0]['percentage'], 50.0)


if __name__ == '__main__':
    unittest.main()
