# Model comparison

| model | domain | f1 | roc_auc | pr_auc | mae | rmse | r2 | ece | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet18 | tile histology | 1.0000 | 1.0000 | 1.0000 | undefined | undefined | undefined | undefined | undefined |
| resnet50 | tile histology | 1.0000 | 1.0000 | 1.0000 | undefined | undefined | undefined | undefined | undefined |
| efficientnet_b0 | tile histology | 1.0000 | 1.0000 | 1.0000 | undefined | undefined | undefined | undefined | undefined |
| vit | tile histology | 1.0000 | 1.0000 | 1.0000 | undefined | undefined | undefined | undefined | undefined |
| bilstm | progression and ctDNA | 1.0000 | 1.0000 | 1.0000 | 0.2278 | 0.3205 | 0.9482 | 0.0019 | 0.0000 |
| transformer | progression and ctDNA | 0.9440 | 0.9922 | 0.9874 | 0.4081 | 0.5817 | 0.8294 | 0.0322 | 0.0341 |
| mil_mean | patient histology | 0.9866 | 0.9996 | 0.9992 | undefined | undefined | undefined | undefined | undefined |
| mil_median | patient histology | 0.9730 | 0.9985 | 0.9971 | undefined | undefined | undefined | undefined | undefined |
| mil_max | patient histology | 0.7439 | 0.8830 | 0.7946 | undefined | undefined | undefined | undefined | undefined |
| mil_attention | patient histology | 0.9792 | 0.9995 | 0.9989 | undefined | undefined | undefined | undefined | undefined |
| r18_bilstm_fixed | progression and ctDNA | 0.9381 | 1.0000 | 1.0000 | 0.2278 | 0.3205 | 0.9482 | 0.2566 | 0.0906 |
| r50_bilstm_fixed | progression and ctDNA | 0.6813 | 1.0000 | 1.0000 | 0.2278 | 0.3205 | 0.9482 | 0.2651 | 0.1134 |
| r50_transformer_fixed | progression and ctDNA | 0.8829 | 0.9835 | 0.9706 | 0.4081 | 0.5817 | 0.8294 | 0.2298 | 0.1101 |
| r50_attention_transformer_fixed | progression and ctDNA | 0.5517 | 0.9604 | 0.9189 | 0.4081 | 0.5817 | 0.8294 | 0.2225 | 0.1313 |
| fusion_concat_mlp | progression and ctDNA | 0.9516 | 0.9930 | 0.9891 | 0.4098 | 0.5790 | 0.8310 | 0.0374 | 0.0313 |
| fusion_gated | progression and ctDNA | 0.9516 | 0.9934 | 0.9894 | 0.3902 | 0.5662 | 0.8384 | 0.0341 | 0.0314 |
| fusion_cross_attention | progression and ctDNA | 0.9516 | 0.9930 | 0.9889 | 0.3920 | 0.5614 | 0.8412 | 0.0401 | 0.0318 |
| pathology_only | progression and ctDNA | 0.0000 | 0.4759 | 0.4146 | 1.1443 | 1.4526 | -0.0636 | 0.0347 | 0.2415 |
| final_fusion_seed42 | progression and ctDNA | 0.9516 | 0.9930 | 0.9891 | 0.4098 | 0.5790 | 0.8310 | 0.0374 | 0.0313 |
| final_fusion_seed123 | progression and ctDNA | 0.9440 | 0.9924 | 0.9880 | 0.4011 | 0.5743 | 0.8337 | 0.0385 | 0.0327 |
| final_fusion_seed2026 | progression and ctDNA | 0.9440 | 0.9924 | 0.9880 | 0.4095 | 0.5830 | 0.8287 | 0.0321 | 0.0321 |

Histology and progression are different targets; their F1 values must not be compared directly. Timings, counts, complete metrics and weights are in model_comparison.csv.
