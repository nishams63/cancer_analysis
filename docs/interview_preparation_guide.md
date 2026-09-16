# Personalized Precision Medicine for Oncology Treatment Optimization
## Master Technical Interview Preparation Guide: Multi-Stage & Role-by-Role Deep Dive

---

## Executive Summary & System Architecture

This project implements an end-to-end, enterprise-grade, offline-capable Clinical Decision Support System (CDSS) for precision oncology. The system is engineered across four modular stages, each fulfilling a distinct clinical predictive and generative mandate:

1. **Stage 1 (Machine Learning):** Tabular toxicity risk prediction (Low / Moderate / High) from clinical encounters, vitals, labs, and genomics using regularized Gradient Boosted Decision Trees (LightGBM).
2. **Stage 2 (Deep Learning):** Multimodal tumor progression and pathology classification utilizing Computer Vision (ResNet-18 transfer learning on histopathology tiles) and Temporal Deep Learning (Bidirectional LSTM & Temporal Transformer multi-task forecasters on longitudinal biomarker trajectories).
3. **Stage 3 (Natural Language Processing):** Clinical narrative comprehension, negation-aware concept extraction (NegEx algorithm), and structured classification (Urgency & Hazard) from oncology notes.
4. **Stage 4 (Small Language Model):** Fine-tuned generative Small Language Model (Qwen2.5-1.5B via PEFT LoRA) producing Clinical Decision Support Triads (Risk Assessment, Key Findings, Actionable Recommendations) running 100% offline on standard hospital CPU infrastructure with a 6-stage deterministic safety firewall.

Across all four stages, engineering responsibilities are divided into five core roles:
- **Data Engineering:** Collection, schema definition, cleaning pipelines, privacy preservation, deterministic splitting, and leakage prevention.
- **EDA Engineering:** Statistical profiling, hypothesis testing, class balance audits, anti-shortcut verification, token/vocabulary distributions, and readiness gating.
- **Model Engineering (ML / DL / NLP / SLM):** Architecture selection, inductive bias alignment, feature representations, training objectives, and optimization.
- **Evaluation Engineering:** Metric selection aligned with clinical utility, locked test evaluation, error transition analysis, calibration, subgroup fairness auditing, and safety constraint verification.
- **Integration Engineering:** Service architecture, REST API design, schema contracts, offline deployment, deterministic fallbacks, drift monitoring, and automated circuit breakers.

---
---

# STAGE 1: MACHINE LEARNING — Tabular Toxicity Risk Prediction

---

## 1.1 Data Engineering Workflow

### 1.1.1 Data Collection & Ingestion
- **What We Did:** Ingested raw clinical oncology encounters from `oncology_uncleaned.csv` (8,754 rows, 35 raw columns) using `pandas` with strict UTF-8 encoding and schema assertion.
- **Why This Choice:** Clinical EMR exports frequently contain non-ASCII characters (e.g., Latin-1 symbols in investigational drug names, metric prefixes like $\mu\text{g}$, degree symbols in temperature). Enforcing UTF-8 prevents silent byte corruption or column misinterpretation at ingestion.
- **Comparison to Alternatives:**
  - *Standard auto-detection (chardet):* Inefficient and non-deterministic across operating systems; can misclassify rare medical character encodings.
  - *Dropping non-ASCII characters:* Destroys critical dosage and unit information ($\mu\text{g} \to \text{g}$, a 1,000,000x error).

### 1.1.2 Data Profiling & Anomalous Token Auditing
- **What We Did:** Programmatically scanned all fields for 13 domain-specific missing value tokens: `'NA'`, `'N/A'`, `'null'`, `'None'`, `'unknown'`, `'UNKNOWN'`, `'?'`, `'-'`, `'.' `, `''`, `'missing'`, `'not recorded'`, `'pending'`.
- **Why This Choice:** In electronic medical records, missing data is rarely represented as standard IEEE floating-point `NaN`. EMR drop-downs and manual clinical data entry inject semantic strings. If treated as strings, categorical encoders treat `"unknown"` as a valid patient phenotype, inducing severe synthetic bias.
- **Comparison to Alternatives:**
  - *Automated profiling packages (e.g., ydata-profiling):* Provide high-level summaries but fail to map proprietary clinical string tokens to null values without custom configuration.
  - *Global string replacement:* Risks replacing valid clinical tokens (e.g., a gene named `None` or blood type negation). Our dictionary lookup targeted specific columns with explicit type guards.

### 1.1.3 The 8-Step Deterministic Cleaning Pipeline
The raw tabular data passed through a strict sequential cleaning pipeline:
1. **Column Name Standardization:** Converted all headers to `lowercase_snake_case`. Eliminates casing inconsistencies (`Patient_Age` vs `patient_age`) that cause runtime silent drop failures in merge operations.
2. **Whitespace Trimming:** Stripped leading and trailing whitespace from all string features. Prevents accidental category splitting (`"Male"` vs `"Male "` resulting in two separate categories).
3. **Deterministic Categorical Mapping:** Standardized categorical variants via explicit lookup dictionaries (e.g., `{'M': 'Male', 'm': 'Male', 'F': 'Female', 'f': 'Female'}`).
   - *Why not Fuzzy Matching (Levenshtein Distance)?* Fuzzy string matching is dangerous in oncology. For instance, `"NSCLC"` (Non-Small Cell Lung Cancer) and `"SCLC"` (Small Cell Lung Cancer) have a Levenshtein distance of only 1, yet represent completely distinct oncological diagnoses with mutually exclusive chemotherapy protocols.
4. **Missing Token to NaN Conversion:** Converted the 13 verified missing string tokens into true `np.nan`.
5. **Numeric Parsing with Suffix Stripping:** Stripped measurement units embedded in strings (e.g., `"65 years"` $\to 65.0$, `"12.5 ng/mL"` $\to 12.5$, `"95%"` $\to 95.0$, `"120 mg"` $\to 120.0$) using compiled regular expressions before typecasting to `float64`.
6. **Physiological Boundary Enforcement:** Clipped or flagged physiologically impossible measurements based on medical consensus:
   - Age: $[0, 120]$ years
   - Heart Rate: $[30, 220]$ bpm
   - Systolic BP: $[50, 260]$ mmHg
   - Diastolic BP: $[30, 160]$ mmHg
   - $\text{SpO}_2$: $[70, 100]\%$
   - *Why Domain Boundaries over Statistical Outliers (Z-score / IQR)?* Statistical trimming (e.g., removing values $> 3\sigma$) in clinical data inadvertently eliminates true, critical emergencies (e.g., a patient in septic shock with a heart rate of 190 bpm or acute respiratory failure with $\text{SpO}_2$ of $72\%$). Domain-defined physiological impossibility boundaries filter hardware or logging errors while preserving critical patient pathologies.
7. **Median Imputation for Continuous Variables:** Imputed missing continuous values using the training set medians.
   - *Why Median over Mean?* Clinical lab values (e.g., mutation burden, serum ferritin, bilirubin) exhibit extreme right-skewness. The mean is pulled by extreme outliers, whereas the median reflects the true central tendency of the patient population.
   - *Why Not KNN or Iterative Imputer (MICE)?* KNN and multivariate imputation use feature correlations across samples. In cross-validation, performing KNN across the full dataset causes data leakage. Performing KNN inside each fold increases inference latency and memory requirements, and requires storing the entire training set for deployment. Median imputation is leakage-safe, fast, deterministic, and requires storing only a scalar per feature.
   - *Why Not Row Dropping (Complete Case Analysis)?* In real-world EMRs, almost every patient has at least one missing lab test. Dropping rows with missing values would discard $>40\%$ of the cohort, severely biasing against sicker patients who undergo more tests.
8. **Deduplication:** Dropped exact duplicate rows while preserving multiple clinical encounters for the same patient over time.

### 1.1.4 Target Validation & Hard Safety Gates
- Enforced that target variable `toxicity_risk` contained strictly valid classes: `{"Low", "Moderate", "High"}`.
- Zero null values or ambiguous labels permitted in the target column.

### 1.1.5 Data Leakage Prevention & Feature Isolation
- **Forbidden Predictor Elimination:** Explicitly removed `treatment_response` (e.g., `"Progressive Disease"`, `"Complete Response"`).
  - *Why this is catastrophic leakage:* `treatment_response` is an outcome measured *after* or *concurrently* with toxicity during the therapeutic cycle. If included, the model learns a trivial correlation rather than predicting future toxicity risk at baseline.
- **Metadata Exclusion:** Excluded non-predictive administrative identifiers: `patient_id`, `encounter_id`, `observation_date`.
- **Historical Predictors Retained:** Preserved `previous_toxicity_grade` and `previous_adverse_event` because these capture pre-treatment historical baseline risk known at the time of clinical decision-making.

---

## 1.2 EDA Engineering Workflow

### 1.2.1 Target Class Distribution Analysis
- **Empirical Ratios:** Low ($\sim 54\%$), Moderate ($\sim 27\%$), High ($\sim 19\%$).
- **Clinical & Modeling Consequence:** The dataset exhibits moderate class imbalance ($2.8:1$ majority-to-minority ratio). A naive classifier predicting the majority class ("Low") achieves $54\%$ accuracy while failing to identify $100\%$ of high-toxicity patients.
- **Downstream Decision:** Macro-averaged F1 was established as the primary evaluation metric (weighting all classes equally), with **High-Risk Recall** designated as the non-negotiable clinical safety metric.

### 1.2.2 Univariate Distribution & Skewness Profiling
- Evaluated parametric (mean, standard deviation) and non-parametric (median, IQR, skewness, kurtosis) statistics across all 20 numerical features.
- Identified heavy right-skewness in `mutation_burden` ($\text{skew} > 2.8$) and `ctdna_level` ($\text{skew} > 3.1$).
- Guided the feature engineering pipeline to implement logarithmic transformations ($\log(1 + x)$) to stabilize variance.

### 1.2.3 Bivariate & Multivariate Correlation Analysis
- Computed Pearson and Spearman correlation matrices across laboratory parameters and vital signs.
- Observed moderate-to-high collinearity between `systolic_bp` and `diastolic_bp` ($r \approx 0.72$), and between `drug_dose` and `treatment_cycle`.
- **Why We Retained Collinear Features:** Unlike ordinary least squares (OLS) linear regression where multicollinearity inflates coefficient standard errors, tree-based ensemble methods (LightGBM) are non-parametric and split greedily on individual features. Collinear features do not degrade gradient boosting predictive stability. Retaining them preserves clinical interpretability for SHAP tree explanations.
- **Why Not PCA (Principal Component Analysis)?** PCA transforms meaningful clinical variables (e.g., Blood Pressure, Creatinine) into uninterpretable orthogonal eigenvectors. An oncologist cannot act on a recommendation driven by "Component 3."

---

## 1.3 ML Engineering Workflow

### 1.3.1 Patient-Level Stratified Splitting (80/20)
- **What We Did:** Implemented `GroupShuffleSplit` grouping strictly by `patient_id`. 80% training ($\sim 7,000$ encounters across 4,800 unique patients), 20% locked test ($\sim 1,750$ encounters across 1,200 unique patients).
- **Why Patient-Level over Random Row-Level Split?** Patients frequently have multiple encounters across chemotherapy cycles. In a random row-level split, encounters from Patient $X$ appear in both the training set and the test set. The model memorizes patient-specific biological idiosyncrasies (e.g., baseline blood pressure, personal genetic quirks) rather than generalizable toxicity patterns. This produces severe data leakage and artificially inflated test scores that collapse in clinical production.
- **Verification Rule:** Programmatically verified $\text{Train}_{\text{pids}} \cap \text{Test}_{\text{pids}} = \emptyset$ (raises `RuntimeError` if violated).

### 1.3.2 Cross-Validation Strategy
- Implemented 5-Fold `StratifiedGroupKFold`.
  - *Stratified:* Preserves the 54:27:19 class ratio across all 5 validation folds.
  - *Grouped:* Enforces zero patient overlap between the 4 training folds and the 1 validation fold in every iteration.
  - *In-Fold Transformation:* All scalers, imputers, and target encoding mappings were fit strictly on the in-fold training data and applied to the validation fold, guaranteeing zero data leakage.

### 1.3.3 Domain-Specific Feature Engineering (11 Features, 2 Tiers)
Rather than relying solely on automated polynomial feature expansion, we engineered 11 clinically grounded composite biomarkers:

**Tier 1 — Baseline Interactions (6 Features):**
1. `blood_pressure_ratio` = $\frac{\text{systolic\_bp}}{\text{diastolic\_bp}}$ (vascular compliance indicator)
2. `pulse_pressure` = $\text{systolic\_bp} - \text{diastolic\_bp}$ (arterial stiffness marker)
3. `hematologic_risk_flag` = $(\text{hemoglobin} < 11.0) \lor (\text{platelet\_count} < 150)$ (bone marrow suppression marker)
4. `prior_toxicity_risk_flag` = $(\text{previous\_toxicity\_grade} \ge 2) \land (\text{previous\_adverse\_event} == \text{True})$ (anamnestic vulnerability)
5. `comorbidity_age_interaction` = $\text{comorbidity\_count} \times \text{age}$ (frailty index)
6. `tumor_biomarker_index` = $\log(1 + \text{mutation\_burden}) \times \text{ctdna\_level}$ (composite tumor load)

**Tier 2 — Expanded Physiological Interactions (5 Features):**
7. `cumulative_treatment_load` = $\text{drug\_dose} \times \text{treatment\_cycle} \times (\text{previous\_treatment\_count} + 1)$ (pharmacokinetic accumulation)
8. `organ_impairment_index` = $\text{creatinine\_level} + \frac{\text{liver\_function\_marker}}{20.0}$ (renal and hepatic clearance deficit)
9. `vital_instability_score` = Count of abnormal vital thresholds ($\text{HR} \notin [60, 100]$, $\text{SBP} \notin [90, 140]$, $\text{SpO}_2 < 95\%$)
10. `genomic_instability_score` = $\text{mutation\_burden} \times \frac{\text{gene\_expression\_score}}{50.0}$
11. `biomarker_severity_weight` = Trajectory trend mapped to categorical weights ($\text{Decreasing} = 0.5, \text{Stable} = 1.0, \text{Increasing} = 1.5$)

### 1.3.4 Model Selection: LightGBM
**Why LightGBM Over Alternatives:**
- *vs Logistic Regression:* Tabular clinical data contains complex, non-linear interactions (e.g., high drug dose combined with high creatinine causes toxicity, but either alone might be tolerated). Logistic regression requires manual specification of all multi-way interaction terms.
- *vs Random Forest:* Random Forest builds deep, unpruned trees independently. Gradient boosting builds shallow trees sequentially to minimize residual loss, providing superior calibration and discriminative power on tabular benchmarks.
- *vs XGBoost:* LightGBM uses Histogram-Based Feature Binning and Leaf-Wise (Best-First) tree growth rather than level-wise growth. It trains $4\times$ to $8\times$ faster, natively handles categorical features without one-hot expansion, and consumes a fraction of the RAM.
- *vs Tabular Deep Learning (TabNet, FT-Transformer):* On datasets with $<50,000$ rows, deep tabular architectures consistently underperform tree ensembles, require orders of magnitude more compute, and lack deterministic feature attribution.

### 1.3.5 Systematic Ablation Study (4 Empirical Experiments)
1. **Feature Set Ablation:**
   - *Set A (30 raw features):* Macro F1 = 0.512
   - *Set B (36 features = 30 raw + 6 Tier-1 engineered):* Macro F1 = **0.5288** (Optimal)
   - *Set C (41 features = 30 raw + 11 engineered):* Macro F1 = 0.5294 (Negligible gain, increased variance and overfitting risk)
   - *Selection:* Set B (parsimonious, interpretable, superior generalization).
2. **Class Imbalance Strategy:**
   - Tested: Unweighted, Balanced (`compute_class_weight`), Moderate-Boost, Aggressive High-Risk.
   - *Winner:* Balanced weighting. Heavy class weighting over-predicted High risk, inducing severe false positive rates that degrade clinical trust.
3. **Regularization Sweep (Generalization-First):**
   - Hyperparameters: `max_depth = 3`, `num_leaves = 8`, `n_estimators = 80`, `learning_rate = 0.10`, `min_child_samples = 30`, `reg_alpha (L1) = 0.5`, `reg_lambda (L2) = 1.0`, `subsample = 0.70`, `colsample_bytree = 0.70`.
   - *Philosophy:* Constraining tree depth to 3 eliminates high-order memorization, forcing the model to learn robust first- and second-order clinical heuristics.
4. **Post-Hoc Decision Rules:**
   - Multipliers: $W = [1.0, 1.05, 1.05]$. Gently boosts probability thresholds for Moderate and High classes without destabilizing precision.

---

## 1.4 Evaluation Engineering Workflow

### 1.4.1 Locked Test Set Performance (N=1,750 Encounters, 1,200 Patients)
- **Macro F1:** **0.5288** (95% Bootstrap CI: $[0.5035, 0.5521]$)
- **High-Risk Recall (Clinical Safety Metric):** **0.6287** (95% Bootstrap CI: $[0.5759, 0.6783]$)
- **Overall Accuracy:** **0.5766** (95% CI: $[0.5520, 0.6000]$)
- **Log Loss:** 0.9129 | **Brier Score:** 0.1817

### 1.4.2 Per-Class Breakdown
- **Low Risk:** Precision = 0.7364, Recall = 0.6911, F1 = 0.7130 (Support = 942)
- **Moderate Risk:** Precision = 0.3434, Recall = 0.3122, F1 = 0.3271 (Support = 474)
- **High Risk:** Precision = 0.4828, Recall = **0.6287**, F1 = 0.5462 (Support = 334)

### 1.4.3 Generalization Gap Audit
- Cross-Validation Macro F1: 0.5427 $\to$ Test Macro F1: 0.5288 (Gap: **0.0139** or $1.39\%$)
- Cross-Validation HR Recall: 0.6444 $\to$ Test HR Recall: 0.6287 (Gap: **0.0157** or $1.57\%$)
- *Significance:* A generalization gap $< 2\%$ demonstrates that the model did not memorize the training distribution and generalizes reliably to unseen clinical patients.

### 1.4.4 Error Transition Matrix & Clinical Risk Analysis
- **Critical Misses (High $\to$ Low):** 50 cases ($2.8\%$ of total test set). These represent false negatives where severe toxicity is missed. Mitigated downstream by SLM review.
- **Safe Over-Triage (Low $\to$ Moderate):** 209 cases ($11.9\%$). Clinically acceptable because over-monitoring causes no patient harm.
- **Model Progression:**
  - Candidate V1 (baseline): F1 = 0.5204, HR Recall = 0.5808
  - Candidate V3 (aggressive weights): F1 = 0.5188, HR Recall = 0.6018 (Precision collapsed)
  - Candidate V4 (regularized + gentle rules): F1 = **0.5288**, HR Recall = **0.6287** (Pareto optimal)

---

## 1.5 Integration Engineering Workflow (Deep Dive)

### 1.5.1 The `V4InferenceEngine` Encapsulation Architecture
The Integration Engineer's primary responsibility in Stage 1 is encapsulating the trained, evaluated, and frozen artifacts into a completely decoupled, high-performance inference engine (`stage-1-ml/integration/src/predictor.py`):
- **Frozen Artifacts Managed:**
  1. `model.joblib`: Frozen `ThresholdAdjustedClassifier` wrapping regularized LightGBM with decision multiplier weights $W = [1.0, 1.05, 1.05]$.
  2. `preprocessor.joblib`: Frozen `ColumnTransformer` handling 113 encoded feature dimensions (scaling, one-hot encodings, missing indicators).
  3. `target_mapping.json`: Explicit bidirectional mapping (`Low: 0`, `Moderate: 1`, `High: 2`).
  4. `v4_candidate_config.json`: Pinned feature schemas and metadata.
- **Strict Frozen Model Policy:** The integration layer enforces read-only access. It never retrains, fine-tunes, or mutates any weights, guaranteeing that production behavior is mathematically identical to locked test evaluation.

### 1.5.2 On-The-Fly Feature Engineering Engine
Hospital EMR systems do not store complex composite mathematical biomarkers. The integration layer bridges this gap by accepting **only the 30 raw clinical fields** and automatically computing the Tier 1 and Tier 2 engineered features on the fly before passing the tensor to the preprocessor:
```text
Raw EMR JSON (30 fields) ──► Input Validation (Pydantic)
                                    │
                                    ▼
                     Dynamic Feature Engineering
         (blood_pressure_ratio, organ_impairment_index, etc.)
                                    │
                                    ▼
                     Frozen Preprocessor (113 dims)
                                    │
                                    ▼
                     Frozen LightGBM (model.joblib)
                                    │
                                    ▼
                     Decision Rules (W = [1.0, 1.05, 1.05])
                                    │
                                    ▼
                     Structured Clinical Response JSON
```

### 1.5.3 Production FastAPI REST Microservice (`app.py`)
Exposes three standardized HTTP/JSON endpoints:
1. `GET /health`: Deep diagnostic probe verifying that model binaries are loaded, SHA-256 hashes match the release manifest, and RAM allocation is within normal parameters.
2. `POST /predict`: Single-encounter synchronous prediction. Returns predicted class, probability distribution (`{"Low": 0.12, "Moderate": 0.28, "High": 0.60}`), confidence score, and model version. Typical latency: **$< 4.5\text{ ms}$**.
3. `POST /predict/batch`: Vectorized array inference designed for nightly EMR batch scoring runs. Can score 10,000 hospital encounters in under 3.2 seconds.

### 1.5.4 Defensive Schema Validation & Fallback Handling
Implemented using Pydantic V2 (`schemas.py`):
- **Numerical Guards:** Explicit bounds assertions (e.g., Age $[0, 125]$, $\text{SpO}_2 [70, 100]\%$, Heart Rate $[30, 220]$).
- **Novel Categorical Fallback:** If an incoming patient record contains an unobserved categorical level (e.g., an unmapped drug code or new clinic department), the engine does NOT crash with an HTTP 500 error. Instead, it catches the unknown token, logs a structured warning, and dynamically maps it to the preprocessor's most frequent training category.

### 1.5.5 Architectural Trade-Off Analysis (Key Interview Topic)
- **REST API vs gRPC:**
  - *Why REST/JSON was chosen:* Universal interoperability with legacy hospital EHR/EMR systems (Epic, Cerner) which communicate via HTTP/1.1 and JSON. Enables transparent debugging via standard cURL and API gateways.
  - *When gRPC would be preferred:* For internal microservice-to-microservice high-frequency communication, gRPC over HTTP/2 with binary Protocol Buffers reduces payload size by $60\%$ and serializes faster.
- **Model Serialization: `joblib` vs `ONNX` vs `Pickle`:**
  - *Why `joblib`:* Standard for Scikit-Learn/LightGBM pipelines containing large NumPy arrays. Uses memory mapping (`mmap`) to share model weights across multiple worker processes without duplicating RAM.
  - *Why not raw `pickle`:* Security vulnerability (arbitrary code execution upon deserialization) and inefficient memory handling for dense numeric arrays.
  - *Why not `ONNX`:* While ONNX provides cross-language execution, custom Python feature engineering classes (e.g., our `FeatureEngineer`) require writing custom ONNX C++ runtime operators, introducing unnecessary operational complexity for tabular tree models.

---
---

# STAGE 2: DEEP LEARNING — Multimodal Progression & Pathology

---

## 2.1 Data Engineering Workflow

### 2.1.1 Synthetic Multimodal Cohort Generation
- **Pathology Vision:** 12,000 histopathology tiles ($224 \times 224 \times 3$ RGB) spanning three distinct tissue classes: Benign (glandular lumens), Malignant (pleomorphic nuclei, hyperchromasia), and Inflammation (dense lymphocytic infiltrate).
- **Longitudinal Biomarkers:** 16,012 laboratory observations across 1,000 patients tracking five circulating biomarkers: ctDNA VAF (Variant Allele Frequency), CEA, CA-125, LDH, and CRP over a 236-day longitudinal window.
- **Trajectory Archetypes:** Engineered five mathematical dynamic trajectories: gradual increase, rapid increase, stable, gradual decrease, and fluctuating.

### 2.1.2 Cohort Partitioning & Zero-Leakage Splitting
- Split: **700 Train / 150 Validation / 150 Test** patients.
- Grouped strictly by `patient_id`. Zero patient overlap across partitions ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$).
- **Why 70/15/15 over 80/20?** Neural networks require iterative epoch-by-epoch loss tracking, learning rate decay scheduling, and early stopping. Tuning these hyperparameters directly on the test set causes optimization leakage. A three-way split maintains a clean validation set for tuning and a pristine, locked test set for final audit.

### 2.1.3 Temporal Observation Windows
- **Historical Context:** Days $0 \le t \le 90$ utilized as input sequence.
- **Forecasting Window:** Predicting 30-day forward ctDNA trajectory and disease progression status for days $t > 90$.
- **Strict Temporal Barrier:** Data collection timestamps after Day 90 are masked during feature ingestion to prevent future-lookahead leakage.

---

## 2.2 EDA Engineering Workflow

### 2.2.1 Image Modality Profiling & Exposure Auditing
- Validated $100\%$ readable status of all 12,000 tiles; zero corrupt or missing image files.
- Computed per-channel RGB distributions: Mean channel intensities $[R=0.72, G=0.55, B=0.68]$ reflecting standard Hematoxylin & Eosin (H&E) staining coloration.
- Class distribution: Exactly balanced 1:1:1 ($4,000$ Benign, $4,000$ Malignant, $4,000$ Inflammation).

### 2.2.2 Anti-Shortcut Hypothesis Testing (Rigorous Scientific Controls)
- In deep learning computer vision, models frequently latch onto non-biological artifacts (e.g., background brightness, stroma tint, scanner vignetting) rather than true cellular morphology.
- **Statistical Tests Executed:**
  - One-Way ANOVA across classes for Background Brightness: $F = 0.14, p = 0.87$ (No significant difference).
  - ANOVA for Stroma Color Tint: $F = 0.40, p = 0.67$ (No significant difference).
  - ANOVA for Local Contrast: $F = 1.02, p = 0.36$ (No significant difference).
  - Chi-Square Test & Cramér's V for Staining Variations: $p = 0.83, V = 0.012$.
- **Conclusion:** Confirmed mathematically that low-level image exposure statistics do not correlate with diagnostic class labels, guaranteeing that deep vision backbones must learn cellular geometry and nuclear-to-cytoplasmic ratio rather than spurious shortcuts.

### 2.2.3 Longitudinal Biomarker Dynamics
- Evaluated temporal observation density: Median 16 observations per patient (range 10–24).
- Verified that missing lab entries occurred randomly (MCAR) and evaluated time interval distributions between blood draws (mean interval: 14.2 days).

---

## 2.3 DL Engineering Workflow

### 2.3.1 Computer Vision Backbone: ResNet-18 Transfer Learning
- **Architecture:** ResNet-18 pretrained on ImageNet-1K (11.7 million parameters).
- **Modification:** Replaced final fully connected layer with a specialized clinical classification head:
  $$\text{AdaptiveAvgPool2d}() \to \text{Linear}(512, 128) \to \text{ReLU}() \to \text{Dropout}(0.30) \to \text{Linear}(128, 3)$$
- **Why ResNet-18 Over Alternatives:**
  - *vs ResNet-50 / ResNet-101:* ResNet-50 contains 25.6M parameters. On 12,000 tiles, larger networks exhibit severe capacity over-parameterization, leading to memorization of synthetic stroma patterns. ResNet-18 provides the ideal inductive bias with sufficient capacity and rapid convergence.
  - *vs Vision Transformers (ViT-B/16):* ViTs lack intrinsic inductive biases (translation equivariance and locality) and require massive pretraining datasets ($>10^7$ images like JFT-300M) to outperform CNNs. On moderate-sized clinical cohorts, ViTs underperform CNNs and demand excessive GPU compute.
  - *vs Training from Scratch:* Pretraining on ImageNet equips early convolutional filters with Gabor-like edge detectors, texture maps, and gradient representations that accelerate convergence and stabilize feature extraction.
- **Backup Architecture (PathologyCNN):** A custom 4-stage convolutional network ($3 \to 32 \to 64 \to 128 \to 128$ channels with BatchNorm and MaxPool) developed to ensure CPU-friendly training and inference without external weight dependencies.

### 2.3.2 Temporal Architecture: Bidirectional LSTM Multi-Task Forecaster
- **Architecture:** 2-layer Bidirectional LSTM ($\text{input\_dim} = 5$, $\text{hidden\_dim} = 64$, output dimension $128$).
- **Multi-Task Dual Heads:**
  - *Head A (Continuous Regression):* $\text{Linear}(128, 32) \to \text{ReLU} \to \text{Linear}(32, 1)$ predicting 30-day forward ctDNA VAF. Loss: Mean Squared Error (MSE).
  - *Head B (Binary Classification):* $\text{Linear}(128, 32) \to \text{ReLU} \to \text{Linear}(32, 1) \to \text{Sigmoid}$ predicting disease progression status ($\ge 20\%$ increase in tumor burden). Loss: Binary Cross-Entropy (BCE).
- **Crucial Engineering Detail (Last Valid Timepoint Extraction):**
  Patients have variable numbers of longitudinal visits. When batching variable-length sequences with zero-padding, naive pooling takes the final padded vector ($t = T_{\max}$), which contains zeroes. Our model utilizes sequence length indexing:
  $$h_{\text{patient}} = h_i[\text{lengths}[i] - 1]$$
  This extracts the hidden state at the patient's true last recorded clinical visit, eliminating padding contamination.
- **Why BiLSTM Over Alternatives:**
  - *vs Unidirectional LSTM:* Unidirectional models only process chronological history ($t_0 \to t_n$). In clinical oncology, interpreting a sudden biomarker spike at Day 45 depends fundamentally on whether it subsequently stabilized or continued accelerating. Bidirectional processing contextualizes intermediate laboratory perturbations across the full trajectory window.
  - *vs GRU:* While GRUs have fewer parameters (no separate cell state), LSTMs maintain separate long-term memory cells ($c_t$) and hidden states ($h_t$), providing superior retention across long, irregularly spaced clinical time horizons.

### 2.3.3 Temporal Transformer Forecaster
- Built a comparable 2-layer TransformerEncoder ($\text{d\_model} = 64$, 4 attention heads, learned positional embeddings).
- Incorporates causal attention padding masks to prevent self-attention over padded time steps.
- Provides an empirical benchmark against the recurrent BiLSTM baseline.

---

## 2.4 Evaluation Engineering Workflow

### 2.4.1 Locked Test Set Evaluation (N=150 Patients)
- **Pathology CNN Classification:** Accuracy = **100.0%**, Macro F1 = **1.0000**, Macro ROC-AUC = **1.0000**.
- **Temporal ctDNA Regression:** Mean Absolute Error (MAE) = **0.3698%**, $R^2 = \mathbf{0.8236}$, Pearson $r = \mathbf{0.9702}$.
- **Temporal Progression Classification:** Binary Accuracy = **100.0%**, ROC-AUC = **1.0000**.

### 2.4.2 The "Synthetic Data Audit": Addressing Perfect Test Metrics
- **The Core Question:** Why did the models achieve near-perfect metrics, and is this realistic?
- **The Rigorous Engineering Explanation:**
  1. *Procedural Geometry:* The synthetic pathology tiles were generated with mathematically separated morphological patterns (e.g., circular non-overlapping rings for benign lumens, dense clustered high-frequency nuclei for malignancy). The CNN acts as an optimal filter bank, achieving perfect class separation.
  2. *Low Trajectory Noise:* Temporal biomarker series followed 5 clean parametric functions with low stochastic noise ($\sigma = 0.04$). After observing 6–8 timepoints, the underlying trajectory archetype is uniquely identifiable.
  3. *Interview Positioning:* "Near-perfect scores on synthetic data validate that our pipelines, architectures, and loss formulations are bug-free and learn correct mathematical representations. In real clinical EMR data, we expect substantial metric drops due to tissue preparation artifacts, staining variability, tumor heterogeneity, and patient non-adherence. The synthetic benchmark validates the engineering system, not real-world clinical efficacy."

### 2.4.3 Interpretability & Robustness Stress-Testing
- **Grad-CAM (Gradient-Weighted Class Activation Mapping):** Computed gradients from the final convolutional layer of ResNet-18, verifying that high activation heatmaps align with cellular nuclei clusters rather than stroma backgrounds.
- **Permutation Feature Importance:** Shuffled temporal features independently. Revealed that `ctdna_vaf` and `cea` accounted for $78\%$ of progression prediction importance, matching oncological reality.
- **Perturbation Testing:** Evaluated model degradation across 20 image corruptions (Gaussian noise, motion blur, brightness shifts) and temporal noise regimes, establishing performance boundaries.

---

## 2.5 Integration Engineering Workflow (Deep Dive)

### 2.5.1 The `MultimodalInferenceService`
The Stage 2 integration layer (`stage-2-dl/integration/src/inference_service.py`) unifies the two decoupled deep learning backbones into a single, cohesive, patient-level multimodal inference pipeline:
```text
                         Patient ID / Clinical Request
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
     Biopsy Pathology Tiles                        Temporal Series (<=90d)
                │                                             │
      Frozen ResNet-18 CNN                          Frozen BiLSTM Forecaster
     (best_pathology_cnn.pt)                        (best_temporal_lstm.pt)
                │                                             │
      Tile Pooling Engine                           Last Valid Timepoint Slice
     (Mean / Median / Max)                                    │
                │                                    Dual Head Evaluation:
                ▼                                   - Progression Risk (P_prog)
      P_malignant in [0, 1]                         - Forward 30d ctDNA VAF (%)
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       │
                                       ▼
                         Late Multimodal Fusion Engine
                                       │
                         Weighted Linear Alert Formula
                                       │
                                       ▼
                       Interactive Clinical Dashboard / API
```

### 2.5.2 Biopsy Tile Aggregation Mechanics
In clinical pathology, a patient biopsy consists of an entire Whole Slide Image (WSI) containing thousands of tissue coordinates. The integration service ingests all available tiles for a patient and implements three selectable pooling strategies:
- **Mean Pooling (Default):** Averages malignancy logits across tiles, providing a smooth, representative tissue summary.
- **Median Pooling:** Robust against extreme staining outliers or folded tissue artifacts.
- **Max Pooling (Worst-Tile Rule):** Identifies the single most aggressive malignant tile. Clinically vital because if even one tile shows high-grade cancer, the patient is malignant regardless of surrounding benign tissue.

### 2.5.3 Late Multimodal Fusion Formula & Alert Mapping
The engine scales forward ctDNA VAF into a normalized $[0, 1]$ risk index and computes the composite multimodal risk score:
$$\text{Multimodal Risk Score} = 0.35 \cdot P_{\text{malignant}} + 0.40 \cdot P_{\text{progression}} + 0.25 \cdot \text{ctDNA\_risk}$$
This score maps to standardized clinical alert tiers:
- **`LOW ALERT`**: Score $< 0.35$ (Schedule routine follow-up)
- **`MODERATE ALERT`**: $0.35 \le \text{Score} < 0.70$ (Increase biomarker monitoring frequency)
- **`HIGH ALERT`**: Score $\ge 0.70$ (Immediate oncology multidisciplinary team review)

### 2.5.4 Missing Modality Graceful Degradation Engine
In clinical reality, patients frequently have blood test results available weeks before a tissue biopsy is processed. A rigid multimodal model would crash. Our integration layer implements a dynamic modality router:
- **`FULL_MULTIMODAL`**: Both biopsy and temporal series available $\to$ calculates complete 3-component weighted score.
- **`PATHOLOGY_ONLY`**: Biopsy present, no blood series $\to$ re-normalizes alert score entirely to $P_{\text{malignant}}$.
- **`TEMPORAL_ONLY`**: Blood series present, no tissue biopsy $\to$ re-weights to:
  $$\text{Score} = 0.60 \cdot P_{\text{progression}} + 0.40 \cdot \text{ctDNA\_risk}$$
- **`INSUFFICIENT_DATA`**: Neither modality available $\to$ returns explicit clinical error payload with diagnostic recommendations.

### 2.5.5 Interactive Clinical Dashboard (`dashboard/app.py` & `dashboard.html`)
The integration module provides an interactive web dashboard for oncologists (`stage-2-dl/integration/dashboard/`):
- **Pathology Tile Viewer:** Visualizes H&E biopsy tiles with class prediction probabilities and Grad-CAM attention heatmaps.
- **Interactive Longitudinal Trajectory Charts:** Plots historical blood biomarkers (Days 0–90) and projects the forecasted 30-day ctDNA progression cone.
- **Dynamic Multimodal Gauge:** Real-time visual risk meter reflecting fused score and alert status.
- **Patient Simulation Sandbox:** Allows clinicians to toggle different tile aggregation strategies (mean vs max) and simulate how hypothetical biomarker changes affect overall risk.

### 2.5.6 DL Runtime Optimization & Memory Management
- **Inference Mode (`torch.no_grad()`):** Disables PyTorch autograd graph creation, reducing memory consumption by $65\%$ and speeding up forward passes by $2.5\times$.
- **Model Evaluation State (`model.eval()`):** Sets BatchNorm running stats and disables Dropout layers to guarantee deterministic outputs.
- **Device Agnostic Execution:** Automatically detects CUDA GPUs (`torch.cuda.is_available()`), with automatic CPU fallback leveraging OpenMP multi-threading for hospital workstation compatibility.

---
---

# STAGE 3: NLP — Clinical Narrative Comprehension & Feature Extraction

---

## 3.1 Data Engineering Workflow

### 3.1.1 Clinical NLP Dataset
- **Volume & Source:** 6,098 validated clinical oncology documents (`clinical_nlp_dataset_v1.parquet`) across 1,000 unique patients and 2,038 clinical encounters (consultation notes, inpatient progress notes, pathology summaries, discharge notes).
- **De-Identification & Privacy (HIPAA Safe Harbor):** All direct identifiers (names, MRNs, phone numbers, exact residential addresses) were stripped and replaced with deterministic synthetic surrogate tokens (`PT-000001`, `ENC-001024`).
- **Storage Optimization:** Serialized using Apache Parquet with Snappy compression, reducing disk footprint by $78\%$ compared to CSV while preserving precise data schemas and metadata types.

### 3.1.2 Deterministic Cohort Partitioning
- Partitioned into **700 Train / 150 Validation (N=909 docs) / 150 Locked Test (N=928 docs)** patients.
- Grouped strictly by `patient_id`. Zero patient or encounter leakage across splits.

---

## 3.2 EDA Engineering Workflow

### 3.2.1 Text Length & Context Window Auditing
- **Length Metrics:** Mean word count = **105.90** words (median: 104, min: 68, max: 143 words).
- **Token Analysis:** Evaluated BPE token distributions. Under standard medical tokenizers, $100\%$ of documents contain $< 210$ subword tokens.
- **Architectural Consequence:** Proved that a context window of `max_length = 256` tokens captures $100\%$ of clinical text without truncating any clinical narrative, while reducing Transformer self-attention compute by $75\%$ relative to default 512-token windows.

### 3.2.2 Vocabulary & Entity Profiling
- Corpus vocabulary: 3,181 unique terms. Follows standard Zipfian decay with heavy tail representation of antineoplastic agents (Cisplatin, Pembrolizumab, Trastuzumab, Fluorouracil) and genetic markers (EGFR, KRAS, BRAF, HER2).
- Document distribution: Consultation Notes ($34\%$), Progress Notes ($33\%$), Discharge Summaries ($33\%$).

### 3.2.3 The 100% Negation Prevalence Discovery
- **Key Finding:** **100.0% of clinical notes** contain at least one explicit clinical negation cue (mean: 2.30 negation cues per document).
- **Engineering Implication:** Standard Bag-of-Words or naive keyword searching is completely invalid in clinical medicine. A document stating *"Patient denies fever, no evidence of metastasis, negative for EGFR mutation"* contains three cancer keywords (`fever`, `metastasis`, `EGFR`), but all three are clinically absent. Negation-aware extraction is mandatory.

### 3.2.4 Severe Class Imbalance Auditing
- **Urgency Level (4 Classes):** `LOW` ($67.9\%$), `HIGH` ($13.7\%$), `CRITICAL` ($9.5\%$), `MEDIUM` ($8.9\%$). Imbalance ratio: **7.60 : 1**.
- **Hazard Type (8 Classes):** `NONE` ($79.5\%$), `HEPATIC` ($7.8\%$), `HEMATOLOGIC` ($4.8\%$), `RENAL` ($3.2\%$), `PULMONARY` ($2.1\%$), `INFECTION` ($1.1\%$), `NEUROLOGIC` ($0.9\%$), `CARDIAC` ($0.57\%$). Imbalance ratio: **138.43 : 1**.

---

## 3.3 NLP Engineering Workflow

### 3.3.1 Clinical Text Preprocessing Pipeline
1. **Conservative Text Cleaning:**
   - Preserves all digits, periods in decimal numbers, measurement symbols ($\%$, $\mu\text{g}$, $\text{mg/m}^2$), and structural delimiters.
   - *Why Conservative over Aggressive Cleaning?* Standard NLP cleaning removes punctuation and numbers. In oncology, stripping numbers destroys dosage ("Cisplatin 75 mg" $\to$ "Cisplatin mg") and changes staging ("Stage IV" $\to$ "Stage").
2. **NFKC Unicode Normalization:**
   - Normalizes compatibility ligatures (e.g., $\text{ﬁ} \to \text{fi}$, $\mu \to \mu$) and whitespace variants.
3. **Clinical Sentence Boundary Detection:**
   - Custom rule-based sentence tokenizer handling medical honorifics and abbreviations ("Dr.", "vs.", "q.d.", "b.i.d.", "mg.") without creating erroneous sentence breaks.

### 3.3.2 Deterministic Negation Detection (NegEx-Style Algorithm)
Implemented a rule-based clinical negation and status attribution engine operating across 4 distinct polarities:
- **Pre-Negation Triggers (Forward Scope):** `no`, `not`, `denies`, `without`, `ruled out`, `negative for`, `absence of`, `fails to show`. Scope: Up to 6 subsequent tokens or until a scope boundary.
- **Post-Negation Triggers (Backward Scope):** `unlikely`, `absent`, `resolved`, `negative`, `free`. Scope: Preceding 4 tokens.
- **Historical / Anamnestic Triggers:** `history of`, `prior`, `past medical history`, `previously treated with`.
- **Pseudo-Negation Guards (Non-Negating Expressions):** `no change`, `no increase`, `not only`, `no doubt`. These contain negation tokens but express positive clinical continuity.
- **Scope Delimiters:** Conjunctions (`but`, `however`, `although`, `except`) and punctuation (`;`, `.`) immediately terminate scope propagation.
- **Polarity Classes:** Every extracted medical entity is tagged as: `AFFIRMED`, `NEGATED`, `HISTORICAL`, or `RESOLVED`.
- **Why Rule-Based NegEx over Transformer-Based Negation?**
  1. *Zero Compute Overhead:* Executes in sub-millisecond CPU time via compiled regex passes.
  2. *Deterministic Auditability:* Clinicians can verify the exact lexical rule that flipped an entity polarity, essential for medico-legal liability.

### 3.3.3 Clinical Concept Extraction (NER)
- Extracts four entity classes: `DRUG`, `GENE`, `DOSAGE`, `ADVERSE_EVENT`.
- Links each entity with its sentence-level negation polarity.

### 3.3.4 Representation & Baseline Modeling
- **Negation-Aware TF-IDF:**
  - Standard TF-IDF treats negated words identically to affirmed words. Our vectorizer suppresses `NEGATED` and `HISTORICAL` entities, preventing non-existent symptoms from contributing positive weights to urgency classifiers.
- **Baseline Model:** Class-weighted Logistic Regression with L2 regularization.
  - *Why Logistic Regression Baseline?* Fast, interpretable, sets an empirical performance floor that the generative SLM must beat.

---

## 3.4 Evaluation Engineering Workflow

### 3.4.1 Locked Test Set Performance (N=928 Documents)
- **Urgency Level Classification:** Macro F1 = **0.7557**, **Critical Patient Recall = 94.57%**.
- **Hazard Type Classification:** Macro F1 = **0.5214** (severely constrained by extreme $138:1$ class imbalance in cardiac and neurologic hazards).
- **Named Entity Recognition:** Mean Span F1 = **76.70%**.

### 3.4.2 Metric Selection Rationales
- **Macro F1 over Micro F1 / Accuracy:** On a dataset where `LOW` urgency is $68\%$, a trivial model predicting `LOW` achieves $68\%$ accuracy. Macro F1 computes metrics per class independently before averaging, preventing majority-class dominance.
- **Critical Patient Recall as a Safety Constraint:** Missing a `CRITICAL` urgency note can result in unmanaged toxicity and patient mortality. We tuned decision thresholds to guarantee Critical Recall $>94\%$.

---

## 3.5 Integration Engineering Workflow (Deep Dive)

### 3.5.1 The Inter-Stage Schema Contract (`contract_validation.py`)
In enterprise machine learning, silent schema drift between upstream feature extraction and downstream generative models is a primary source of silent outages. Stage 3 defines and strictly enforces a formal **Inter-Stage Integration Contract** (`stage-3-nlp-slm/integration/contract_validation.py`):
- **Version Enforcement:** Pins `SUPPORTED_SCHEMA_VERSION = "1.0.0"`. Any minor or major version discrepancy raises a fatal validation exception.
- **Strict Pydantic Model (`extra="forbid"`):** Forbids any unmodeled attributes, ensuring downstream models never receive unexpected or unauthorized fields.
- **Hex SHA-256 Integrity Hashes:** Computes a SHA-256 hash over the raw clinical narrative text to establish a cryptographic audit chain.

### 3.5.2 Standardized Output Payloads
1. **`TriageUrgencyPayload`**:
   - `predicted_class`: Strictly constrained to `Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]`.
   - `confidence`: Floating-point scalar bounded in $[0.0, 1.0]$.
   - `class_probabilities`: Explicit dictionary of calibrated probabilities for all 4 classes.
2. **`HazardPayload`**:
   - `predicted_hazard`: One of 8 validated organ hazard types (`NONE`, `HEMATOLOGIC`, `HEPATIC`, `RENAL`, `CARDIAC`, `PULMONARY`, `NEUROPATHIC`, `DERMATOLOGIC`).
3. **`EntityPayload`**:
   - Array of extracted entity objects, each tagged with `entity_type` (`GENE_MUTATION`, `DRUG_NAME`, `DOSAGE`, `ADVERSE_EVENT`), start/end character offsets, and verified `polarity` (`AFFIRMED`, `NEGATED`, `HISTORICAL`, `RESOLVED`).

### 3.5.3 Confidence Gating & Selective Human Routing (`confidence_gate.py`)
Not all clinical notes can be classified with high certainty. The integration module implements an automated confidence gate:
- **Threshold Policy:** If $\max(P_{\text{urgency}}) < \tau_{\text{NLP}} = 0.65$, the note is flagged with `review_required = True`.
- **Selective Routing:** High-confidence notes proceed directly to the Stage 4 SLM prompt generator. Low-confidence notes are routed to a human oncologist triage queue, preventing ambiguous text from misleading the generative model.

### 3.5.4 Structured Failure Handlers (`failure_mode_handlers.py`)
Implements defensive exception routing for production anomalies:
- **Empty or Whitespace-Only Notes:** Caught before NLP processing; returns a clean validation error payload rather than crashing tokenizer libraries.
- **Context Length Overflow ($> 256$ tokens):** Employs intelligent sentence-boundary truncation that preserves initial clinical assessment sentences and terminal plan sections while trimming repetitive middle text.
- **Corrupted Characters:** Filters non-printable control characters via NFKC normalization before regex parsing.

### 3.5.5 Immutable Audit Logging (`integration_audit_logger.py`)
Every transaction is recorded into an append-only JSONL ledger (`integration_audit.jsonl`):
- Records timestamp (UTC ISO-8601), document ID, input text SHA-256 hash, extracted entity array, predicted urgency, confidence score, and processing latency in milliseconds.
- **HIPAA Compliance:** Raw patient text is not stored in the log file; only its SHA-256 hash is logged. This enables full forensic reproducibility without storing unencrypted Protected Health Information (PHI) in operational log storage.

---
---

# STAGE 4: SLM — Small Language Model Fine-Tuning & Offline CDSS

---

## 4.1 Data Engineering Workflow

### 4.1.1 Clinical Instruction Dataset Construction
- **Dataset:** 5,706 accepted instruction-completion pairs (`slm_finetune_dataset_v1.parquet`) generated from validated Stage 3 records.
- **Instruction Schema (17 Features):** Structured prompt containing `patient_id`, `note_id`, `document_type`, `clinical_note`, alongside gold-standard completion triads:
  - `target_risk`: Quantitative toxicity risk classification (`Low`, `Moderate`, `High`).
  - `target_key_finding`: Exactly two factual sentences synthesizing affirmed clinical entities and vitals.
  - `target_action`: Concrete, guideline-adherent oncological recommendations (e.g., *"Hold Cisplatin; administer intravenous hydration; re-check serum creatinine in 48 hours"*).

### 4.1.2 Entity Preservation Quality Gate
- Ran programmatic entity extraction on both the source note and the generated target text.
- Verified that target completions contained zero "invented" entities (hallucinations) and captured all critical antineoplastic agents and adverse events.
- Enforced a dataset-level rejection circuit breaker: If $>5\%$ of pairs violated entity preservation, pipeline generation halted automatically.

### 4.1.3 Zero-Leakage Cohort Isolation
- Split: **700 Train / 150 Validation / 150 Locked Test** patients. Verified zero cross-split patient overlap across 23 automated unit tests.

---

## 4.2 EDA Engineering Workflow

### 4.2.1 BPE Token Distributions & Context Overflow Auditing
- Tokenized source instructions and target completions using the Qwen2.5 byte-pair encoding (BPE) tokenizer (151,643 vocabulary size).
- **Source Token Length:** Mean 198 tokens, 99th percentile 342 tokens, maximum 488 tokens.
- **Target Token Length:** Mean 42 tokens, 99th percentile 62 tokens, maximum 71 tokens.
- **Context Allocation:** Total sequence length comfortably bounded within **1024 tokens**. Allocating 1024 tokens guarantees $0.00\%$ context truncation while avoiding the memory footprint of 2048 or 4096 windows.

### 4.2.2 Subword Fragmentation Analysis
- Audited how the BPE tokenizer segments complex oncological pharmaceuticals (e.g., `Pembrolizumab` $\to$ `['Pem', 'brol', 'izumab']`, `Fluorouracil` $\to$ `['Flu', 'or', 'our', 'acil']`).
- Verified that fragmentation does not exceed 4 tokens per drug name, confirming that Qwen's extensive vocabulary represents medical morphology efficiently without degradation into character-level tokens.

### 4.2.3 Negation Flip Audit
- Cross-compared entity negation polarity between source notes and target summaries across all 5,706 pairs.
- Identified 0 instances of polarity inversion (e.g., a negated symptom in the note being summarized as affirmed in the action recommendation).

---

## 4.3 SLM Engineering Workflow

### 4.3.1 Base Model Selection: Qwen2.5-1.5B-Instruct
**Why Qwen2.5-1.5B-Instruct Over Alternatives:**
- *vs API-Based LLMs (GPT-4, Claude 3.5 Sonnet):*
  1. **Data Privacy (HIPAA / GDPR):** In clinical oncology, transmitting Protected Health Information (PHI) to third-party cloud APIs violates patient confidentiality and hospital governance.
  2. **Deterministic Availability:** Cloud APIs suffer from rate limits, network outages, and silent upstream weight changes.
  3. **Economic Scalability:** At thousands of daily encounters, API fees become exorbitant. An SLM incurs zero marginal per-inference query cost.
- *vs Larger Open LLMs (LLaMA-3-8B, BioMistral-7B):*
  An 8B parameter model requires $\approx 16\text{ GB}$ of VRAM in FP16, necessitating costly enterprise server GPUs (A100/H100). Qwen2.5-1.5B fits comfortably in $<2\text{ GB}$ RAM when quantized, enabling high-speed inference on standard, existing hospital workstations and CPUs.
- *vs Smaller Models (GPT-2, SmollLM-360M):*
  Models $<1\text{B}$ parameters lack the internal attention capacity for complex multi-hop clinical reasoning and frequently fail to adhere to rigid JSON triad formatting. Qwen2.5-1.5B represents the empirical "Goldilocks" sweet spot.
- *Architectural Advantages:* Utilizes SwiGLU (Swish Gated Linear Unit) activation functions, Rotary Position Embeddings (RoPE), and Grouped-Query Attention (GQA), providing superior inference efficiency and long-context stability.

### 4.3.2 Fine-Tuning Strategy: PEFT LoRA (Low-Rank Adaptation)
- **Mathematical Principle:** Freezes base weights $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable rank decomposition matrices:
  $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A), \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k)$$
- **Hyperparameter Configuration:**
  - Rank: $r = 16$ | Scaling Factor: $\alpha = 32$ ($\frac{\alpha}{r} = 2.0$)
  - Dropout: $0.05$ | Learning Rate: $2 \times 10^{-4}$ with Cosine Annealing and 5% Warmup
  - Batch Size: 4 (Effective batch size 16 via gradient accumulation steps = 4)
  - Mixed Precision: FP16 with Gradient Checkpointing
- **Target Modules:** Injected adapters into **all 7 linear projections**:
  - Attention layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - MLP feed-forward layers: `gate_proj`, `up_proj`, `down_proj`
- **Why Target All 7 Modules over Attention Only?**
  Earlier LoRA literature focused solely on query and value projections ($q, v$). Empirical research on language models demonstrates that factual domain knowledge and clinical reasoning circuits reside primarily in the MLP feed-forward blocks (`gate_proj`, `up_proj`, `down_proj`). Adapting all linear layers yielded superior clinical entity retention without increasing rank $r$.
- **Prompt Token Loss Masking:** Set `labels = -100` for all prompt and instruction tokens. The cross-entropy loss was calculated strictly on completion tokens (`target_risk`, `key_finding`, `action`), preventing the model from wasting gradient updates memorizing prompt scaffolding.

### 4.3.3 Edge Deployment: GGUF Quantization & llama.cpp
- **Format:** Converted fine-tuned LoRA weights into unified GGUF (GPT-Generated Unified Format) using **Q4_K_M** quantization.
- **Memory Footprint:** Reduced model file size from $3.1\text{ GB}$ (FP16) to **$986\text{ MB}$** ($4$-bit).
- **Inference Runtime:** Powered by `llama.cpp`, a pure C/C++ inference engine with zero external Python/PyTorch dependencies. Executes at $>25\text{ tokens/second}$ on commodity quad-core hospital CPUs.
- **Why GGUF Over ONNX?**
  - ONNX Runtime is optimized for tensor graph execution on GPU/NPU hardware but has cumbersome memory overhead for autoregressive text generation on CPUs.
  - `llama.cpp` + GGUF uses specialized AVX-512 vector CPU intrinsics and k-quantization (maintaining higher bit precision on critical attention norms while quantizing non-critical feed-forward weights to 4-bit), offering superior perplexity-to-memory trade-offs.

---

## 4.4 Evaluation Engineering Workflow

### 4.4.1 Locked Test Set Evaluation (N=861 Pairs, 150 Patients)
Direct comparison between Stage 3 Statistical Baseline and Stage 4 Fine-Tuned SLM:

| Performance Metric | Stage 3 Baseline | Stage 4 Fine-Tuned SLM | Relative Improvement |
|:---|:---:|:---:|:---:|
| **Entity Preservation (Span F1)** | 76.70% | **80.17%** | **+3.47%** |
| **Critical Patient Recall** | 94.57% | **100.00%** | **+5.43%** |
| **Hallucination Rate** | Unmeasured | **0.00%** | **Zero Defect** |
| **ROUGE-1** | — | **52.42%** [51.67%, 53.21%] | — |
| **ROUGE-L** | — | **50.83%** [50.01%, 51.67%] | — |
| **BLEU-4** | — | **28.37%** | — |

### 4.4.2 Zero Hallucination Verification & The 6-Stage Clinical Safety Firewall
**Why is the Hallucination Rate 0.00%?**
In clinical medicine, a single hallucinated drug or dosage can be fatal. We implemented a **Defense-in-Depth post-inference regex firewall**:
1. **Gate 1 — Drug Entity Boundary:** Every drug recommended in `target_action` must exist verbatim in the input clinical note or approved institutional formulary.
2. **Gate 2 — Physiological Dosage Bounds:** Regex matches all numeric dosages and cross-references them against maximum therapeutic oncology thresholds (e.g., Cisplatin $\le 100\text{ mg/m}^2$).
3. **Gate 3 — Mutation & Biomarker Verification:** Verifies that recommended targeted therapies match the patient's verified genomic profile (e.g., blocking Osimertinib if EGFR is negative).
4. **Gate 4 — Semantic Contradiction Detector:** Asserts logical coherence (e.g., flagging output if `target_risk` is `"Low"` while `target_action` recommends `"Emergency ICU Transfer"`).
5. **Gate 5 — Non-Clinical Phrase Filter:** Blocks conversational filler, colloquialisms, or non-deterministic disclaimers.
6. **Gate 6 — Schema & Structural Validator:** Validates that output conforms strictly to the parsed Triad JSON schema.

### 4.4.3 Empirical Calibration & Selective Prediction
- Evaluated confidence calibration using Expected Calibration Error (ECE) and Brier score.
- Established an empirical validation threshold: $\tau^* = 0.500$.
- **Selective Prediction Rule:** If the model's generation confidence score is below $\tau^*$, the system abstains from generative output and automatically falls back to the deterministic Stage 3 baseline.

### 4.4.4 Subgroup Fairness Auditing Across 26 Demographic & Clinical Strata
Audited performance across:
- Cancer Types (6): Breast, Lung, Colorectal, Prostate, Melanoma, Pancreatic
- Cancer Stages (4): Stage I, II, III, IV
- Patient Sex (2), Smoking History (3), Primary Mutations (6)
- **Result:** $100\%$ pass rate across active strata; Span F1 $\ge 0.70$ across all cohorts; $0.00\%$ hallucination rate maintained across all 23 active subgroups.

---

## 4.5 Integration Engineering Workflow (Deep Dive)

### 4.5.1 Production Offline Architecture
The Stage 4 Integration Engineer packages the fine-tuned model into a complete, standalone, production-ready clinical application running locally with **zero external API dependencies** (`stage-4-slm/integration-engineer/`):
```text
CLINICAL NOTE (EMR)
         │
         ▼
[Input Sanitizer & Canonical PromptBuilder v1.0.0]
         │
         ▼
[ModelManager: llama.cpp CPU Runtime (Q4_K_M GGUF)]
         │
         ▼
[OutputParser: JSON Triad + 3-Line Voice Spoken Summary]
         │
         ▼
[SafetyGateway: 6-Stage Clinical Safety Firewall]
         │
         ├── Passes All Gates ──► [Approved Output & Cryptographic Provenance]
         │
         └── Fails Any Gate   ──► [Automatic Fallback to Stage 3 Baseline]
```

### 4.5.2 The 7 Key Integration Components
1. **`api.py` (FastAPI Production Service):**
   - Serves clinical endpoints with CORS and static UI mounting:
     - `POST /generate/decision-support`: Main endpoint returning full validated triad, safety badge, and audio-ready summary text.
     - `GET /health`: Deep diagnostic probe asserting model loading, memory usage, and GGUF binary checksums.
2. **`inference_service.py` (Pipeline Orchestrator):**
   - Coordinates prompt synthesis, C++ inference execution, output parsing, safety verification, and audit logging into a single cohesive pipeline.
3. **`model_manager.py` (Local C++ Runtime Manager):**
   - Loads `merged-model-Q4_K_M.gguf` using `llama.cpp`.
   - Manages CPU thread pools, context window buffers, and temperature settings ($T = 0.1$ for deterministic, reproducible generation).
4. **`prompt_builder.py` (Canonical Prompt Formatter v1.0.0):**
   - Formats raw clinical notes into strict instruction-following prompts with prompt token masking and XML/markdown delimiters.
   - Sanitizes input text against prompt-injection attacks.
5. **`output_parser.py` (Structured Parser & Voice Summary Engine):**
   - Extracts `target_risk`, `target_key_finding`, and `target_action` into structured Pydantic models.
   - Synthesizes a natural, 3-line spoken summary for oncologist voice interfaces (hands-free clinical dictation in surgical suites).
6. **`safety_gateway.py` (Clinical Safety Firewall Enforcement):**
   - Evaluates the 6 post-inference safety gates. If an entity is unverified or dosage exceeds safety limits, intercepts the response before it reaches the clinician.
7. **`provenance.py` & `audit_logger.py` (Cryptographic Provenance):**
   - Computes SHA-256 hashes of base model weights, LoRA adapter, GGUF binary, input prompt, and output completion.
   - Writes append-only transactions to `audit_log.jsonl`, establishing complete legal and clinical chain of custody.

### 4.5.3 Accessible Offline Web Dashboard & Terminal CLI
- **Clinical Web Dashboard (`frontend/index.html`, `app.js`, `styles.css`):**
  - Zero-dependency, lightweight web interface served directly by FastAPI.
  - Interactive risk cards, expandable entity provenance, safety gate badges (Green/Red), and one-click Web Speech API audio playback.
- **Terminal CLI (`cli.py`):**
  - Standalone command-line tool allowing oncology researchers to test notes directly in terminals:
    ```powershell
    python cli.py --note "Patient on osimertinib 80mg with elevated liver enzymes."
    ```

### 4.5.4 Air-Gapped Containerization (`deployment/`)
- Includes production `Dockerfile` and `docker-compose.yml`.
- Configured with `network_mode: none` to guarantee air-gapped security. The container operates with zero network cards attached, proving to hospital security auditors that patient PHI cannot physically leave the server.

---
---

# 5. CROSS-STAGE ARCHITECTURAL COMPARISON & SYSTEM SYNTHESIS

---

## 5.1 End-to-End Multimodal Data Flow

```
                                  CLINICAL PATIENT ENCOUNTER
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         │                                    │                                    │
         ▼                                    ▼                                    ▼
   STAGE 1: ML                          STAGE 2: DL                          STAGE 3: NLP
Tabular Labs & Vitals             Pathology Tiles & ctDNA               Unstructured Clinical Notes
         │                                    │                                    │
         ▼                                    ▼                                    ▼
8-Step Cleaning Pipeline             ResNet-18 Vision Backbone             Conservative Text Preprocessing
         │                                    │                                    │
11 Engineered Features               BiLSTM Multi-Task Forecaster          NegEx Polarity Attribution
         │                                    │                                    │
LightGBM Regularized Model           Late Fusion Risk Alert                Negation-Aware TF-IDF Vectorizer
         │                                    │                                    │
Output: Toxicity Risk                Output: Progression Alert             Output: Extracted Entities & Urgency
(Low / Moderate / High)              (Malignancy & ctDNA Trend)            (Span F1 = 80.2%, Recall = 94.6%)
         │                                    │                                    │
         └────────────────────────────────────┼────────────────────────────────────┘
                                              │
                                              ▼
                                         STAGE 4: SLM
                            Instruction-Tuning Prompt Synthesis
                                              │
                                              ▼
                                   Qwen2.5-1.5B (PEFT LoRA)
                                4-Bit GGUF / llama.cpp on CPU
                                              │
                                              ▼
                                   6-Stage Safety Firewall
                                              │
                                              ▼
                                  CLINICAL DECISION SUPPORT
                                 Risk | Findings | Action
```

---

## 5.2 Architectural Complexity & Paradigm Evolution

| Architectural Dimension | Stage 1: Classical ML | Stage 2: Deep Learning | Stage 3: Statistical NLP | Stage 4: Generative SLM |
|:---|:---|:---|:---|:---|
| **Input Modality** | Tabular (Labs, Vitals, Genomics) | Multimodal (RGB Tiles + Time Series) | Unstructured Clinical Text | Structured Prompts (Text + Entities) |
| **Core Algorithm** | Gradient Boosted Trees (LightGBM) | ResNet-18 CNN + BiLSTM / Transformer | Class-Weighted Logistic Regression | Qwen2.5-1.5B Foundation Model |
| **Model Size** | $\sim 10,000$ Decision Trees | $11.7\text{M} + 0.5\text{M}$ Parameters | $\sim 50,000$ Feature Weights | $1.5\text{ Billion}$ Parameters |
| **Inference Compute** | CPU (Sub-millisecond) | CPU / GPU ($\sim 50\text{ ms}$) | CPU ($\sim 5\text{ ms}$) | CPU ($<500\text{ ms}$ via 4-bit GGUF) |
| **Inductive Bias** | Axis-aligned hierarchical splits | Spatial translation + temporal order | Linear additive feature log-odds | Transformer self-attention & causal LM |
| **Primary Failure Mode** | High-dimensional multicollinearity | Spurious texture & stroma shortcuts | Semantic negation misattribution | Hallucination & ungrounded generation |
| **Mitigation Strategy** | Feature selection + regularized depth | Anti-shortcut testing + Grad-CAM | NegEx rule engine + scope guards | 6-Stage Regex Firewall + Fallbacks |

---

## 5.3 System-Wide Design Principles

1. **Strict Patient-Level Isolation:** Zero patient overlap between training, validation, and testing partitions across all four stages.
2. **Pristine Locked Test Sets:** Evaluated strictly once for final reporting. Never utilized for early stopping or hyperparameter exploration.
3. **Defense-in-Depth Safety:** Priority placed on high-recall safety metrics (High-Risk Recall in Stage 1, Critical Recall in Stage 3, 100% Critical Recall in Stage 4).
4. **100% Offline Capability:** Operates with zero cloud API dependencies, ensuring strict compliance with hospital data security, HIPAA, and GDPR regulations.
5. **Deterministic Auditability:** Every recommendation can be inspected back to its underlying lab values, pathology regions (Grad-CAM), or text entities.

---

## 5.4 Top 5 Master Technical Interview Q&A

### Q1: "Why did you build four modular stages rather than a single end-to-end multimodal deep learning model?"
**Answer:**  
"In high-stakes clinical oncology, **modularity is a prerequisite for safety, regulatory approval, and operational reliability**.
1. *Error Isolation:* If an end-to-end black-box model predicts 'Hold Chemotherapy', it is impossible to determine whether that prediction was triggered by an abnormal lab value, a malignant pathology tile, or an entity in the clinical note. In our modular architecture, each subsystem can be debugged, validated, and monitored independently.
2. *Asynchronous Clinical Workflows:* Patient data does not arrive simultaneously. A blood draw is processed in 2 hours; a pathology biopsy takes 4 days; an oncologist's clinical note is logged after the visit. A modular architecture allows each stage to execute as data becomes available rather than blocking the entire pipeline.
3. *Graceful Degradation:* If the computer vision pipeline fails due to a corrupted image, Stages 1, 3, and 4 continue functioning normally. An end-to-end architecture suffers from single-point-of-failure vulnerability."

---

### Q2: "In Stage 1, your LightGBM model achieves a Macro F1 of 0.5288. In many ML domains, 0.53 F1 is considered low. How do you defend this model in a clinical setting?"
**Answer:**  
"First, context matters: this is a **moderately imbalanced 3-class problem** where random chance yields a Macro F1 of $0.33$.
Second, and most importantly, our optimization objective was explicitly **clinical safety rather than symmetric accuracy**. In oncology, the cost of a False Negative (classifying a High-Toxicity patient as Low Risk, resulting in unmonitored toxic shock) is orders of magnitude higher than the cost of a False Positive (over-monitoring a Low-Risk patient).
Our Candidate V4 model achieved **62.87% High-Risk Recall** with a generalization gap of only **1.57%** on the locked test set. Furthermore, Stage 1 does not operate in a vacuum—its predictions feed into the Stage 4 SLM, where our multi-stage safety firewall achieves **100.0% Critical Patient Recall**."

---

### Q3: "Why choose LoRA fine-tuning on a 1.5B Small Language Model rather than zero-shot prompting on an industry LLM like GPT-4?"
**Answer:**  
"This decision was dictated by four engineering and regulatory constraints:
1. *Data Sovereignty & HIPAA Compliance:* Clinical notes contain sensitive Protected Health Information. Sending patient narratives to commercial third-party cloud APIs violates strict healthcare compliance in many jurisdictions. Our SLM runs 100% locally behind the hospital firewall.
2. *Latency & Cost:* Cloud APIs incur latency jitter (500ms to 5s depending on traffic) and high ongoing operational costs. Our quantized 4-bit GGUF model executes locally on existing hospital CPU infrastructure with zero network overhead and zero marginal query cost.
3. *Domain Specialization:* General LLMs are prone to verbose, conversational outputs that resist strict schema compliance. Fine-tuning Qwen 1.5B via LoRA directly on clinical triads conditioned the model to output precise, structured oncology recommendations.
4. *Hallucination Mitigation:* General LLMs hallucinate plausible-sounding medical facts. Our fine-tuned SLM, constrained by prompt token masking and backed by our 6-stage post-inference regex firewall, achieved a mathematically audited **0.00% hallucination rate**."

---

### Q4: "How does your system guarantee zero data leakage across the lifecycle?"
**Answer:**  
"We enforced a four-layer anti-leakage architecture:
1. *Patient-Level Grouping:* We strictly used `GroupShuffleSplit` and `StratifiedGroupKFold` on `patient_id`. We programmatically asserted that the intersection of training, validation, and test patient IDs was the empty set ($\emptyset$).
2. *Temporal Barrier Isolation:* In Stage 2, longitudinal biomarkers were split into a historical observation window ($t \le 90$ days) and a forecasting window ($t > 90$ days). Future data points were strictly masked.
3. *In-Fold Preprocessing:* All imputation medians, scalers, and TF-IDF vocabularies were computed strictly on in-fold training data. Test sets were locked and never touched during hyperparameter tuning.
4. *Feature Exclusions:* Confounding outcome features like `treatment_response` were removed from Stage 1 predictors because they encode post-treatment results."

---

### Q5: "If this system is deployed in a hospital tomorrow, what are the primary failure modes you anticipate, and how are they handled?"
**Answer:**  
"We designed the system around defensive engineering:
1. *Distribution Drift (Covariate Shift):* If the hospital introduces a new chemotherapy agent not in the training set, the model may experience uncertainty. Our `ProductionDriftMonitor` tracks token distributions and entity density; if divergence exceeds thresholds, the system flags the encounter.
2. *Out-of-Distribution Inputs:* If an unformatted or corrupted clinical note enters Stage 4, the SLM's output confidence will fall below our calibration threshold ($\tau^* = 0.500$), or it will trip one of the 6 safety firewall gates.
3. *Graceful Fallback:* Upon tripping any gate, the system instantly triggers an automated fallback to the deterministic Stage 3 baseline.
4. *Human-in-the-Loop:* Crucially, our system is architected as a **Decision Support Tool**, not an autonomous agent. The final recommendation is presented to the treating oncologist with highlighted entity provenance, ensuring human clinical judgment remains the final decision gate."

---
---

# 6. MASTER INTEGRATION ENGINEERING PLAYBOOK & COMPARATIVE ANALYSIS

---

## 6.1 Serving Protocols: REST vs gRPC vs Kafka / Message Queues

| Architecture Dimension | REST (HTTP/1.1 + JSON) | gRPC (HTTP/2 + Protocol Buffers) | Message Queues (Apache Kafka / RabbitMQ) |
|:---|:---|:---|:---|
| **Primary Use Case** | External clinical client integration, hospital web portals | Internal low-latency inter-service microservice RPC | Asynchronous, event-driven decoupled clinical telemetry |
| **Data Serialization** | Textual JSON (Human-readable, verbose) | Binary Protocol Buffers (Compact, strictly typed) | Binary byte arrays (Avro / JSON schemas) |
| **Transport Protocol** | HTTP/1.1 (Single request per TCP connection) | HTTP/2 (Multiplexed bidirectional streaming) | TCP persistent broker connections |
| **Latency Benchmark** | $\sim 5 - 20\text{ ms}$ overhead | $\sim 0.8 - 2.5\text{ ms}$ overhead | $\sim 10 - 50\text{ ms}$ (Queue buffering dependent) |
| **Hospital Integration** | **Native Compatibility:** All EHR/EMR systems (Epic, Cerner) support REST. | **Requires Gateway:** Modern microservices use gRPC behind a REST reverse proxy. | **Night-Runs & Feeds:** Ideal for streaming continuous ICU telemetry feeds. |
| **Decision in Project** | **Adopted for Stages 1, 2, and 4 FastAPI services.** Prioritizes universal hospital interoperability and zero-dependency debugging. | Evaluated for Stage 2 $\to$ Stage 4 internal piping; rejected to maintain clean, inspectable HTTP/JSON boundaries. | Suitable for hospital-wide batch queueing in production expansion. |

---

## 6.2 Serving Paradigms: Real-Time vs Batch vs Streaming

```
                                CLINICAL INFERENCE PARADIGMS
                                             │
         ┌───────────────────────────────────┼───────────────────────────────────┐
         │                                   │                                   │
         ▼                                   ▼                                   ▼
Real-Time Synchronous               Batch Asynchronous                  Streaming (SSE / Chunked)
- Latency: < 500 ms                 - Latency: Minutes / Hours          - Latency: Real-time tokens
- Trigger: Doctor clicks 'Assess'   - Trigger: Nightly EMR Cron Job     - Trigger: SLM generation starts
- Use Case: Point-of-Care Clinic    - Use Case: Population Health Triage- Use Case: Interactive UI Audio
- Endpoints: /predict               - Endpoints: /predict/batch         - Endpoints: /generate/stream
```

1. **Real-Time Synchronous (`/predict`):**
   - *Clinical Scenario:* An oncologist sits in an outpatient consultation room with a patient, types a progress note, and requires an immediate toxicity risk score and action recommendation before prescribing chemotherapy.
   - *Requirements:* Hard latency SLA ($P_{95} < 500\text{ ms}$ for tabular, $< 2.5\text{ s}$ for generative SLM).
2. **Batch Asynchronous (`/predict/batch`):**
   - *Clinical Scenario:* Every night at 02:00 AM, the hospital EMR exports the last 24 hours of laboratory records and vital signs for all 3,500 admitted oncology inpatients.
   - *Requirements:* High vectorized throughput, chunked memory processing (preventing OOM errors), and database bulk inserts.
3. **Token Streaming (Server-Sent Events):**
   - *Clinical Scenario:* The generative SLM outputs words sequentially. Streaming tokens to the browser frontend enables the oncologist to read key findings immediately while remaining tokens generate, eliminating perceived latency.

---

## 6.3 Model Serialization & Runtime Matrix

| Serialization Format | Target Framework | File Size (Stage 4) | CPU Inference Speed | Trade-Offs & Best Fit |
|:---|:---|:---:|:---:|:---|
| **Joblib (`.joblib`)** | Scikit-Learn / LightGBM | $12\text{ MB}$ (Stage 1) | Instant ($<2\text{ ms}$) | **Optimal for tabular ML.** Uses memory mapping for NumPy arrays. Python-dependent. |
| **PyTorch Native (`.pt`)** | PyTorch (CNN + LSTM) | $48\text{ MB}$ (Stage 2) | Fast ($\sim 25\text{ ms}$) | **Optimal for DL research.** Preserves complete state dicts. Requires PyTorch runtime. |
| **TorchScript (`.pt`)** | PyTorch C++ Runtime | $46\text{ MB}$ | Fast ($\sim 20\text{ ms}$) | Graph traced. Runs in standalone C++ environments without Python interpreter. |
| **ONNX (`.onnx`)** | ONNX Runtime | $47\text{ MB}$ | Very Fast | Cross-platform graph representation. Difficult to express complex custom Python feature logic. |
| **GGUF (`.gguf` Q4_K_M)** | `llama.cpp` (C++) | **$986\text{ MB}$ (Stage 4)** | **$>25\text{ tok/s}$ on CPU** | **Winner for offline SLM edge serving.** Unified single-file binary with embedded weights, vocabulary, and metadata. |
| **vLLM / TGI** | High-end GPU Clusters | $\approx 3.1\text{ GB}$ (FP16) | Extremely Fast ($>100\text{ tok/s}$) | Requires expensive server GPUs (A100/H100) and CUDA drivers. Incompatible with cheap hospital CPUs. |

---

## 6.4 Defensive Systems Engineering: Fallback Architecture

```
                       INCOMING CLINICAL PATIENT PAYLOAD
                                       │
                                       ▼
                       [Contract Validation Layer (SemVer)]
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
             [Valid Schema]                        [Malformed Schema]
                    │                                     │
                    ▼                                     ▼
        [Stage 4 SLM Generation]                [Structured Error 422 Payload]
                    │
                    ▼
     [Confidence Gate: Confidence >= 0.500?]
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
       [YES]                  [NO] ────────────────────────┐
         │                                                 │
         ▼                                                 │
[6-Stage Safety Firewall]                                  │
         │                                                 │
   ┌─────┴─────┐                                           │
   ▼           ▼                                           │
[PASS]       [FAIL] ───────────────────────────────────────┤
   │                                                       │
   ▼                                                       ▼
[Approved SLM Output]                      [DETERMINISTIC FALLBACK ENGINE]
- Risk Assessment                          - Triggers Stage 3 Baseline
- Key Clinical Finding                     - Invokes Rule-Based NegEx NER
- Guideline-Adherent Action                - Flags Record for Human Review
                                           - Logs Safety Audit Event
```

---

## 6.5 Medico-Legal Provenance & HIPAA Air-Gap Compliance

1. **Air-Gapped Container Isolation:**
   - Deployed within Docker containers configured with `network_mode: none`.
   - The Linux kernel disables all socket creation to external interfaces, providing mathematical proof to healthcare compliance officers that patient PHI cannot be exfiltrated.
2. **Cryptographic SHA-256 Chain of Custody:**
   - `Hash(Input Note)` $\to$ `Hash(Prompt Template)` $\to$ `Hash(Model Weights)` $\to$ `Hash(Generated Triad)`.
   - Any modification to model weights or data inputs instantly invalidates downstream cryptographic signatures.
3. **Non-Anonymized PHI Elimination:**
   - Operational logs store only one-way cryptographic SHA-256 hashes and metadata, satisfying HIPAA Safe Harbor standards while preserving full deterministic auditability.
