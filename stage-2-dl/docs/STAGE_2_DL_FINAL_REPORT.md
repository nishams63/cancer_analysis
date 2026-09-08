# Stage 2 DL experimental report

1. ResNet50 vs ResNet18 tile F1: 1.0000 vs 1.0000. Observed difference +0.0000; one training seed, no claim of statistical superiority.

2. Attention vs mean patient-histology F1: 0.9792 vs 0.9866.

3. Transformer vs BiLSTM progression F1: 0.9440 vs 1.0000.

4. Validation-selected learned fusion: fusion_concat_mlp. Fixed/learned comparisons use actual common-cohort progression predictions below.

5. Overall validation-selected system: bilstm. Single-modality candidates were included; see validation rows in CSV.

6. Three seeds independently retrain the learned fusion head. Stability is conditional on fixed feature encoders; it is not full-pipeline multi-seed stability.

7. Gated fusion test ECE: 0.0341 raw, 0.0390 calibrated.

8. Perturbation results, including any improvements, are reported in the robustness table.

9. Missing modalities are explicitly tested; zero-embedding degradation is not evidence of a trained fallback. No-input requests must abstain.

10. Far-OOD synthetic texture ROC-AUC: 0.4872; clean flag rate 0.0338. Real OOD sensitivity remains unvalidated.

11. False positives: 5; false negatives: 1. See actual patient-level failure table.

12. Recommended research configuration by validation: bilstm. The expected ResNet50/attention/Transformer/gated system was not forced to win.

## Actual ablations

| ID | configuration | n | accuracy | precision | recall | f1 | roc_auc | pr_auc | ece | brier | mae | rmse | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | pathology_only | 148 | 0.5946 | 0.0000 | 0.0000 | 0.0000 | 0.4759 | 0.4146 | 0.0347 | 0.2415 | 1.1443 | 1.4526 | -0.0636 |
| B | transformer | 148 | 0.9527 | 0.9077 | 0.9833 | 0.9440 | 0.9922 | 0.9874 | 0.0322 | 0.0341 | 0.4081 | 0.5817 | 0.8294 |
| C | r50_attention_transformer_fixed | 148 | 0.7365 | 0.8889 | 0.4000 | 0.5517 | 0.9604 | 0.9189 | 0.2225 | 0.1313 | 0.4081 | 0.5817 | 0.8294 |
| D | fusion_concat_mlp | 148 | 0.9595 | 0.9219 | 0.9833 | 0.9516 | 0.9930 | 0.9891 | 0.0374 | 0.0313 | 0.4098 | 0.5790 | 0.8310 |
| E | r18_bilstm_fixed | 148 | 0.9527 | 1.0000 | 0.8833 | 0.9381 | 1.0000 | 1.0000 | 0.2566 | 0.0906 | 0.2278 | 0.3205 | 0.9482 |
| F | r50_transformer_fixed | 148 | 0.9122 | 0.9608 | 0.8167 | 0.8829 | 0.9835 | 0.9706 | 0.2298 | 0.1101 | 0.4081 | 0.5817 | 0.8294 |
| G | fusion_gated | 148 | 0.9595 | 0.9219 | 0.9833 | 0.9516 | 0.9934 | 0.9894 | 0.0341 | 0.0314 | 0.3902 | 0.5662 | 0.8384 |
| H | G with MC mean prediction; OOD annotations added subsequently | 148 | 0.9595 | 0.9219 | 0.9833 | 0.9516 | 0.9937 | 0.9902 | 0.0322 | 0.0314 | 0.3902 | 0.5662 | 0.8384 |

## Independent fusion seeds

| seed | f1 | roc_auc | mae | r2 |
| --- | --- | --- | --- | --- |
| 42.0000 | 0.9516 | 0.9930 | 0.4098 | 0.8310 |
| 123.0000 | 0.9440 | 0.9924 | 0.4011 | 0.8337 |
| 2026.0000 | 0.9440 | 0.9924 | 0.4095 | 0.8287 |

| metric | mean | sd |
| --- | --- | --- |
| f1 | 0.9465 | 0.0044 |
| roc_auc | 0.9926 | 0.0003 |
| mae | 0.4068 | 0.0049 |
| r2 | 0.8311 | 0.0025 |

## Limitations

Synthetic data only; research-only, no clinical validation. The bag-composition challenge resamples each patient’s own synthetic tiles and does not create independent tissue. Pathology was generated independently of temporal prognosis; no outcome-conditioned pathology shortcut was added. The exact preserved BiLSTM has padding-sensitive backward states. Frozen pretrained backbones were not fine-tuned end-to-end. Missing target patients are excluded using a predeclared observation window; exclusions are in protocol.json. Classification is a synthetic future biomarker trend, not diagnosed recurrence. Validation is reused for selection and calibration, which may increase selection uncertainty. No clinical OOD data are available. See protocol.json and prediction CSVs for reproducibility.
