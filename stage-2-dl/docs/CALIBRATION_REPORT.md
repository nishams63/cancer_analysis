# Calibration report

| model | temperature | raw_ece | calibrated_ece | raw_brier | calibrated_brier |
| --- | --- | --- | --- | --- | --- |
| fusion_gated | 0.9193 | 0.0341 | 0.0390 | 0.0314 | 0.0318 |

Temperature was fitted on validation only. Test reliability diagram: `../results/validated_v3/reliability_diagram.png`. MC uncertainty is prediction spread, not a clinical confidence interval. Calibration degradation is reported without clipping.
