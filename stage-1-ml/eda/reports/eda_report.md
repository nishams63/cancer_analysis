# Stage 1 EDA Report

## 1. Executive Summary
This report summarizes the Exploratory Data Analysis (EDA) performed on the oncology treatment optimization dataset. The dataset consists of 8,754 patient records and 35 features. The primary target variable is `toxicity_risk`. Key findings include strong associations between previous toxicity events, specific cancer stages, and primary mutations with toxicity risk. Potential leakage was identified in several features that must be addressed prior to model training.

## 2. Dataset Overview
- **Total Rows**: 8,754
- **Total Columns**: 35
- **Numerical Features**: 20
- **Categorical Features**: 14
- **Memory Usage**: ~9.05 MB

## 3. Data Quality Verification
The dataset is clean and preprocessed by the Data Engineer. No significant missing values or unexpected categories were identified that require immediate remediation during EDA.

## 4. Target Variable Analysis
The target variable `toxicity_risk` is categorized into `Low`, `Moderate`, and `High`. The dataset shows a relatively balanced distribution across these three classes, though slightly uneven. Refer to `toxicity_risk_distribution.png` for exact percentages. Class imbalance should be monitored but may not require aggressive oversampling.

## 5. Numerical Feature Analysis
Numerical features such as `previous_toxicity_grade`, `comorbidity_count`, and `hemoglobin` show strong associations with toxicity risk. Detailed distributions and boxplots by risk category are available in `numerical_distributions.png` and `numerical_boxplots.png`.

## 6. Categorical Feature Analysis
Categorical features like `previous_adverse_event` and `cancer_stage` demonstrate highly significant associations with toxicity risk. Stacked bar charts are available in `categorical_analysis.png`.

## 7. Biomarker Analysis
Key biomarkers including `mutation_primary` and `ctdna_level` were identified. These biomarkers show observable dataset-level patterns indicating varying risk profiles. See biomarker visualizations for category-specific risks. Note that these are observed associations and require clinical validation.

## 8. Preliminary Feature Association Leaderboard
The following features demonstrate the strongest statistical association with `toxicity_risk`:
1. `previous_toxicity_grade` (Numerical, ANOVA F: 1136.4, p ≈ 0)
2. `previous_adverse_event` (Categorical, Chi-Square: 1962.7, p ≈ 0)
3. `cancer_stage` (Categorical, Chi-Square: 158.8, p < 1e-31)
4. `mutation_primary` (Categorical, Chi-Square: 110.6, p < 1e-15)
5. `comorbidity_count` (Numerical, ANOVA F: 28.3, p < 1e-12)

## 9. Correlation Analysis
Correlation analysis of numerical features reveals potential multicollinearity among certain clinical measurements. Refer to `correlation_matrix.png` for highly correlated feature pairs. 

## 10. Outlier and Rare Category Analysis
While outliers exist in clinical measures like `hemoglobin` and `creatinine_level`, they represent valid physiological extremes and have not been removed. Rare categories (<1% frequency) exist in some treatment types and should be handled appropriately during encoding to avoid unstable estimates.

## 11. Target Leakage Analysis
The following features require investigation for potential target leakage:
- **POTENTIAL LEAKAGE**: `patient_id`, `encounter_id`, `observation_date` (Identifiers/Dates serving as near-perfect predictors due to dataset structure).
- **INVESTIGATE**: `previous_toxicity_grade`, `previous_adverse_event`, `treatment_response` (Suspiciously high association or names implying post-treatment outcomes).

## 12. Key Findings
- Historical toxicity and adverse events are the strongest predictors of current toxicity risk.
- Biomarkers such as `mutation_primary` and `ctdna_level` show statistically significant differences across risk groups.
- Direct identifiers and potential post-treatment outcomes exist in the dataset and pose a high risk of target leakage.

## 13. Recommendations for ML Engineer
1. **Drop Identifiers**: Remove `patient_id`, `encounter_id`, and `observation_date` to prevent trivial target leakage.
2. **Investigate Leakage**: Work with clinical domain experts to determine if `previous_toxicity_grade`, `previous_adverse_event`, and `treatment_response` are known prior to the prediction time. If they are post-treatment, they must be dropped.
3. **Handle Rare Categories**: Use Target Encoding or group rare categories into an 'Other' bucket for features with high cardinality or low frequency.
4. **Collinearity**: Consider dimensionality reduction (e.g., PCA) or regularization (L1/L2) to handle correlated numerical features.
5. **Class Imbalance**: Evaluate models using stratified sampling and metrics like F1-macro or ROC-AUC, rather than plain accuracy.

## 14. Limitations
This EDA is an exploratory research prototype. Findings represent dataset-level patterns and do not imply clinical causality. Do not use these findings for actual patient diagnosis or treatment recommendations without rigorous clinical validation.

## 15. Conclusion
The dataset provides a rich set of clinical and biomarker features capable of modeling toxicity risk. Addressing the identified target leakage and appropriate feature engineering will be critical for the next stage of ML model development.
