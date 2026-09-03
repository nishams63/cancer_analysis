# Stage 1 ML Evaluation Module

This directory contains the independent evaluation suite, reports, results, and artifacts for the **Personalized Precision Medicine for Oncology Treatment Optimization** Stage 1 Machine Learning system.

## Evaluated Candidate: Candidate V4 (Frozen)
- **Model**: Regularized LightGBM (depth=3, n_estimators=80, balanced weights, gentle thresholds `[1.0, 1.05, 1.05]`)
- **Locked Test Results**:
  - **Macro F1**: `0.5288`
  - **High-Risk Recall**: `0.6287`
  - **Accuracy**: `0.5766`
  - **Weighted F1**: `0.5766`

## Directory Layout
- `reports/`:
  - `final_evaluation_report.md`: Complete independent evaluation report
  - `error_analysis.md`: Detailed breakdown of classification error transitions and confidence
  - `generalization_analysis.md`: CV vs locked-test generalization gap analysis
- `results/`:
  - `final_metrics.json`: Full structured evaluation metrics
  - `confusion_matrix.csv`: Raw 3x3 confusion matrix table
  - `predictions.csv`: Encounter-level predictions, probabilities, and confidence scores
  - `subgroup_metrics.csv`: Performance metrics across clinical subgroups
- `figures/`:
  - `confusion_matrix.png`: High-resolution confusion matrix heatmap
  - `class_distribution.png`: Actual vs predicted distribution bar chart
- `tests/`:
  - `test_evaluation.py`: Pytest suite for evaluation verification

## Running the Evaluation Pipeline
```bash
py stage-1-ml/evaluation/evaluation.py
```
