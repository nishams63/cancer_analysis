# Stage 5 Integration — Cross-Stage Failure Summary

- **Batch ID**: `RERUN-001`
- **Total Failures**: 2

## Failure Distribution by Taxonomy Code
| Failure Code | Count | Severity | Description |
| :--- | :---: | :---: | :--- |
| `F01` | 0 | High | Missed secondary resistance driver in clinical narrative (Stage 3) |
| `F03` | 1 | Critical | ctDNA spike classified as Standard Risk by Tabular ML (Stage 1) |
| `F06` | 1 | Critical | Monotherapy recommended despite acquired kinase bypass (Stage 4) |

## Failure Isolation Invariant
Stages 1–4 execute independently through defensive adapters. When Stage 2 detects unavailable multimodal imaging, it emits `SKIPPED_INPUT_UNAVAILABLE`, allowing Stages 1, 3, and 4 to complete evaluation without pipeline crash.
