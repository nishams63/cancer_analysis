# Failure analysis

| type | count |
| --- | --- |
| correct | 142 |
| false_positive | 5 |
| false_negative | 1 |

Actual largest ctDNA errors:

| patient_id | failure | target_ctdna | ctdna_forecast | ctdna_absolute_error | uncertainty | ood |
| --- | --- | --- | --- | --- | --- | --- |
| PAT-0651 | false_positive | 0.2282 | 2.5601 | 2.3319 | 0.0193 | False |
| PAT-0606 | false_positive | 0.3804 | 2.2491 | 1.8687 | 0.0942 | False |
| PAT-0790 | correct | 5.7679 | 4.0189 | 1.7490 | 0.0062 | False |
| PAT-0746 | false_positive | 0.1733 | 1.7864 | 1.6131 | 0.1490 | False |
| PAT-0006 | false_positive | 0.4419 | 2.0363 | 1.5944 | 0.0897 | True |
| PAT-0970 | correct | 4.8718 | 3.2945 | 1.5773 | 0.0122 | False |
| PAT-0646 | false_positive | 0.4316 | 1.7643 | 1.3327 | 0.1740 | False |
| PAT-0757 | false_negative | 2.2438 | 0.9434 | 1.3004 | 0.0832 | False |
| PAT-0760 | correct | 4.4294 | 3.2574 | 1.1720 | 0.0116 | False |
| PAT-0190 | correct | 5.0912 | 3.9678 | 1.1234 | 0.0047 | False |

All patient predictions, false positives/negatives, uncertainty and OOD flags are in failure_analysis.csv. Temporal and missing-modality challenge predictions are saved separately. Attributions are experimental sensitivities, not biological explanations.
