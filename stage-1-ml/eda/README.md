# Stage 1 EDA Engineer Module

## Purpose
This module contains the Exploratory Data Analysis (EDA) pipeline for the oncology treatment optimization project. The purpose is to understand the dataset, discover patterns, analyze biomarkers, identify target leakage, and provide ML recommendations.

## Input Dataset
The module reads the cleaned dataset created by the Data Engineer:
`../data-engineering/data/processed/master_patient_dataset.csv`

## Folder Structure
```
stage-1-ml/eda/
├── notebooks/          # Jupyter notebooks for interactive EDA
├── src/                # Python scripts for automated EDA
├── visualizations/     # Output directory for generated charts
├── reports/            # Output directory for markdown reports
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation
Ensure you have a Python 3.10+ environment. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run Automated EDA
To run the full EDA pipeline and generate reports and visualizations:
```bash
python src/eda_analysis.py
```

## How to Open Notebook
To view or interact with the Jupyter notebook:
```bash
jupyter notebook notebooks/stage1_eda.ipynb
```

## Generated Outputs
Running the analysis will generate:
- PNG visualizations in the `visualizations/` directory.
- A comprehensive markdown report in `reports/eda_report.md`.

## Important Assumptions
- The dataset has already been cleaned by the Data Engineering pipeline.
- The target variable is `toxicity_risk` with classes `Low`, `Moderate`, `High`.

## Limitations
- **Medical Safety**: This is a research prototype. It does not diagnose patients, prescribe treatment, or recommend medical changes. Observed associations are dataset-level patterns and require clinical validation.
