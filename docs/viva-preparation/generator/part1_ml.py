"""
PART 1 — MACHINE LEARNING
Contains all 24 topics formatted strictly as requested:
Concept, Simple Definition, Example (Oncology Project), Viva Answer, Remember.
"""

ML_CONCEPTS = [
    {
        "num": 1,
        "title": "What is Machine Learning?",
        "def": "Machine Learning (ML) is a branch of Artificial Intelligence where computer systems learn patterns from historical data instead of following hardcoded rules. The system automatically improves its predictions as it is exposed to more data. In simple terms, traditional programming takes Rules + Data to produce Answers, while Machine Learning takes Data + Answers to learn the Rules.",
        "example": "In our Oncology Precision Medicine project, instead of a doctor manually writing if-else rules for 30 lab tests, our ML model learns patterns from 1,750 patient encounters to predict whether a chemotherapy patient will suffer an adverse drug toxicity event.",
        "viva": "Machine Learning is a subset of AI where mathematical algorithms learn statistical relationships from past data to make automated predictions on unseen data without explicit rule-based programming.",
        "remember": "1. Traditional programming: Rules + Data = Answers.<br>2. Machine Learning: Data + Answers = Learned Rules."
    },
    {
        "num": 2,
        "title": "AI vs ML vs Data Science",
        "def": "Artificial Intelligence is the broad umbrella concept of creating machines capable of intelligent human-like behavior. Machine Learning is a specific technique within AI that uses statistical algorithms to learn from data. Data Science is an interdisciplinary field combining statistics, domain expertise (like oncology), data engineering, and ML to extract actionable insights and solve business or medical problems.",
        "example": "In our project: <strong>Data Science</strong> analyzes patient demographics, lab tests, and genomics; <strong>Machine Learning</strong> trains a LightGBM model to predict toxicity risk; and <strong>AI</strong> integrates this model into an autonomous clinical decision support system.",
        "viva": "AI is the broad vision of smart machines; ML is the data-driven engine that powers AI; and Data Science is the overarching discipline that cleans, explores, analyzes, and translates data into predictive solutions.",
        "remember": "Think of nested circles: AI is the largest outer circle, ML is inside AI, and Data Science overlaps both while adding domain expertise and data engineering."
    },
    {
        "num": 3,
        "title": "How Machine Learning Works",
        "def": "Machine Learning works through a 5-step lifecycle: Data Collection, Preprocessing, Model Training, Evaluation, and Inference. During training, the algorithm takes historical inputs, generates trial predictions, calculates the error (loss) against true outcomes, and iteratively adjusts internal parameters to minimize that error.",
        "example": "In our pipeline, we ingest patient lab records (creatinine, liver enzymes, drug dose), preprocess them, feed them into a LightGBM classifier to minimize log-loss error, validate on a locked test set, and deploy as a REST API for real-time inference.",
        "viva": "ML works by iteratively feeding preprocessed data into an algorithm, comparing predictions against actual labels to compute loss, adjusting internal parameters via optimization, and deploying the finalized model to score new inputs.",
        "remember": "The core cycle is: <strong>Input Data &rarr; Hypothesis Function &rarr; Loss Calculation &rarr; Parameter Optimization &rarr; Deployed Inference</strong>."
    },
    {
        "num": 4,
        "title": "Types of Machine Learning",
        "def": "Machine Learning is primarily categorized into three paradigms based on feedback: <strong>Supervised Learning</strong> (learns from labeled input-output pairs), <strong>Unsupervised Learning</strong> (discovers hidden patterns or clusters in unlabeled data), and <strong>Reinforcement Learning</strong> (learns optimal actions through trial-and-error rewards and penalties in an environment).",
        "example": "In oncology: Supervised Learning predicts known patient toxicity grades (Low/Moderate/High); Unsupervised Learning clusters patients into novel genomic sub-phenotypes; Reinforcement Learning optimizes multi-cycle chemo dosing schedules through simulated patient responses.",
        "viva": "The three main types are Supervised Learning (labeled data with targets), Unsupervised Learning (unlabeled data finding natural groupings), and Reinforcement Learning (agent learning via reward feedback).",
        "remember": "Supervised = Teacher present (Ground Truth Labels); Unsupervised = No teacher (Find structure); Reinforcement = Learn from rewards and penalties."
    },
    {
        "num": 5,
        "title": "Supervised Learning: Classification vs Regression",
        "def": "Supervised learning maps input features to a known target variable. When the target is a distinct category or label, it is <strong>Classification</strong>. When the target is a continuous numerical quantity, it is <strong>Regression</strong>.",
        "example": "In our project: Predicting whether a patient's toxicity risk is <strong>Low, Moderate, or High</strong> is Classification. Predicting the patient's exact <strong>serum creatinine level (e.g., 1.42 mg/dL)</strong> or exact survival days is Regression.",
        "viva": "Classification predicts discrete qualitative classes (e.g., benign vs malignant tumor), whereas Regression predicts continuous quantitative values (e.g., drug concentration or blood pressure).",
        "remember": "Classification outputs categorical labels (classes); Regression outputs continuous numerical real values."
    },
    {
        "num": 6,
        "title": "Unsupervised Learning: Clustering vs Dimensionality Reduction",
        "def": "Unsupervised learning works on data without ground truth labels. <strong>Clustering</strong> groups similar data points together based on feature distance. <strong>Dimensionality Reduction</strong> compresses a large number of correlated features into fewer informative components while preserving key variance.",
        "example": "In our project: Using <strong>K-Means Clustering</strong> to group 1,000 cancer patients into 3 distinct metabolic risk cohorts based on 25 lab tests; using <strong>PCA (Principal Component Analysis)</strong> to reduce 5,000 gene expression scores down to 10 principal components for visualization.",
        "viva": "Clustering groups unlabeled samples into meaningful clusters based on similarity; Dimensionality reduction compresses high-dimensional feature spaces to eliminate redundancy, avoid the curse of dimensionality, and enable 2D/3D visualization.",
        "remember": "Clustering groups <em>samples/rows</em>; Dimensionality Reduction compresses <em>features/columns</em>."
    },
    {
        "num": 7,
        "title": "Common ML Algorithms",
        "def": "Machine learning provides several fundamental algorithms tailored to different data relationships: linear boundaries, tree-based partitions, margin maximizers, and probabilistic models.",
        "example": "In our project, we benchmarked Logistic Regression, Decision Trees, and Random Forests before selecting regularized LightGBM (Candidate V4) as our best performing clinical toxicity model.",
        "viva": "Here is how the 8 standard algorithms operate:<br>"
               "• <strong>Linear Regression:</strong> Fits a best-fit straight line (y = mx + c) to predict continuous values by minimizing sum of squared residuals.<br>"
               "• <strong>Logistic Regression:</strong> Uses the Sigmoid function to output probability [0, 1] for binary or multiclass classification via log-odds.<br>"
               "• <strong>Decision Tree:</strong> Hierarchical if-else tree that splits features on thresholds using Gini Impurity or Information Gain.<br>"
               "• <strong>Random Forest:</strong> An ensemble (Bagging) of multiple decorrelated decision trees that votes on the final class, reducing variance and overfitting.<br>"
               "• <strong>K-Nearest Neighbors (KNN):</strong> Instance-based lazy learner that classifies a new sample based on the majority vote of its 'k' closest neighbors using Euclidean distance.<br>"
               "• <strong>Support Vector Machine (SVM):</strong> Finds the optimal hyperplane that maximizes the geometric margin between different classes in high-dimensional space.<br>"
               "• <strong>Naive Bayes:</strong> Fast probabilistic classifier based on Bayes' Theorem, assuming all input features are conditionally independent given the class.<br>"
               "• <strong>K-Means:</strong> Iterative clustering algorithm that partitions data into K clusters by assigning points to the nearest centroid and recalculating centroids until convergence.",
        "remember": "Tree ensembles (Random Forest, LightGBM) handle tabular clinical features best; Logistic Regression and SVM are strong linear/kernel baselines; K-Means is unsupervised."
    },
    {
        "num": 8,
        "title": "Features and Target",
        "def": "A <strong>Feature</strong> (independent variable, X) is an individual measurable property or characteristic of the phenomenon being observed. The <strong>Target</strong> (dependent variable, y) is the outcome or ground-truth label that the machine learning model is trained to predict.",
        "example": "In our Oncology dataset: Features include <code>age</code>, <code>cancer_type</code>, <code>creatinine_level</code>, <code>liver_function_marker</code>, and <code>drug_dose</code>. The Target is <code>toxicity_risk</code> (Low, Moderate, High).",
        "viva": "Features are the input variables (X) describing the patient, while the Target is the ground-truth outcome (y) that the model learns to predict.",
        "remember": "Features = Inputs (X matrix); Target = Ground Truth Output (y vector)."
    },
    {
        "num": 9,
        "title": "Training, Validation, and Testing Data",
        "def": "To properly build and evaluate models, data is split into three distinct sets: <strong>Training Data</strong> (used by the algorithm to learn weights/parameters), <strong>Validation Data</strong> (used to tune hyperparameters and prevent overfitting during development), and <strong>Testing Data</strong> (a locked, held-out dataset used strictly once to assess real-world performance).",
        "example": "In our project, we enforced a strict <strong>Patient-Level Split</strong>: 700 patients for Training, 150 for Validation, and 150 for Locked Testing. This guarantees 0% patient leakage (no patient's visits exist in both train and test).",
        "viva": "Training data fits the model weights; Validation data guides hyperparameter tuning and model selection; Testing data is a locked unseen set providing an unbiased evaluation of generalization.",
        "remember": "Never touch testing data during training or feature engineering! In clinical datasets, always split by patient ID, not just random row rows."
    },
    {
        "num": 10,
        "title": "Data Preprocessing",
        "def": "Data Preprocessing is the process of converting raw, messy real-world data into a clean, structured format suitable for machine learning algorithms. Raw datasets frequently contain missing values, inconsistent units, outliers, text categories, and skewed scales.",
        "example": "In our Oncology Data Engineering pipeline, we removed duplicate encounter IDs, formatted dates (YYYY-MM-DD), imputed missing lab values, encoded categorical cancer types, and scaled continuous biomarkers.",
        "viva": "Data preprocessing transforms raw, noisy data into clean numerical feature matrices through imputation, encoding, outlier filtering, and scaling before model training.",
        "remember": "'Garbage In, Garbage Out' — a high-capacity model trained on uncleaned data will produce unreliable, dangerous predictions."
    },
    {
        "num": 11,
        "title": "Missing Values Handling",
        "def": "Missing values occur when no data value is stored for an observation in a column. Strategies to handle them include: dropping rows/columns (if <2% missing), <strong>Mean/Median Imputation</strong> for numerical data, <strong>Mode or 'Unknown' Category</strong> for categorical data, and advanced model-based KNN/MICE imputation.",
        "example": "In our project: For skewed clinical labs (e.g., <code>creatinine_level</code>, <code>ctdna_level</code>), we used <strong>Median Imputation</strong> to resist outlier distortion; for categorical <code>cancer_type</code> and <code>smoking_history</code>, we imputed a dedicated <code>'Unknown'</code> category.",
        "viva": "Missing values can be handled by deletion, statistical imputation (mean for symmetric data, median for skewed data), or explicit categorization ('Unknown'), fitted strictly on the training fold to prevent data leakage.",
        "remember": "Always fit imputers on the training set and transform the test set! Using test set statistics causes data leakage."
    },
    {
        "num": 12,
        "title": "Encoding Categorical Data",
        "def": "Machine learning algorithms require numerical inputs and cannot directly process text categories. <strong>One-Hot Encoding</strong> creates binary (0/1) columns for nominal categories with no order. <strong>Ordinal Encoding</strong> assigns ranked integers (e.g., 1, 2, 3, 4) to categories with inherent natural ordering.",
        "example": "In our project: <code>cancer_type</code> (NSCLC, Colorectal, Breast) has no order, so we used <strong>One-Hot Encoding</strong>; <code>cancer_stage</code> (Stage I, II, III, IV) has an inherent severity progression, so we used <strong>Ordinal Encoding</strong> (1 to 4).",
        "viva": "Categorical data is converted to numbers using One-Hot Encoding for non-ordered nominal variables (creating binary indicator columns) and Ordinal Encoding for ranked variables (preserving natural ordering).",
        "remember": "Avoid Ordinal Encoding on nominal data (e.g., red=1, green=2, blue=3), as it forces a false mathematical order on the algorithm."
    },
    {
        "num": 13,
        "title": "Feature Scaling",
        "def": "Feature Scaling is the technique of adjusting the numerical range of independent variables so that features with large numerical magnitudes do not disproportionately dominate features with small numerical values during distance or gradient calculations.",
        "example": "In our oncology dataset, <code>platelet_count</code> ranges from 150,000 to 450,000, while <code>creatinine_level</code> ranges from 0.6 to 2.5. Without scaling, distance-based algorithms like KNN and SVM would treat platelet variations as 100,000x more important than creatinine variations.",
        "viva": "Feature scaling standardizes numerical feature ranges to ensure equal contribution during gradient descent optimization and distance calculations in algorithms like KNN, SVM, and Neural Networks.",
        "remember": "Tree-based models (Decision Trees, Random Forest, LightGBM) are scale-invariant, but distance and gradient-based models (KNN, SVM, Logistic Regression, Neural Networks) strictly require scaling."
    },
    {
        "num": 14,
        "title": "Feature Engineering",
        "def": "Feature Engineering is the creative process of using domain knowledge to extract new, informative variables from raw data to improve machine learning model accuracy and clinical interpretability.",
        "example": "In our project, we engineered a <strong>Renal Load Ratio</strong> (<code>creatinine_level / drug_dose</code>) and a <strong>Biomarker Acceleration Index</strong> (change in ctDNA level over treatment cycles). These engineered features had higher feature importance than raw variables alone.",
        "viva": "Feature engineering uses domain knowledge to transform, combine, and construct new high-signal features from raw data, enabling models to capture non-linear relationships more effectively.",
        "remember": "'Better features beat better algorithms.' Clean, domain-rich engineered features often provide bigger performance jumps than hyperparameter tuning."
    },
    {
        "num": 15,
        "title": "Normalization vs Standardization",
        "def": "Both are feature scaling methods. <strong>Normalization (Min-Max Scaling)</strong> rescales values into a fixed [0, 1] range: X_norm = (X - X_min) / (X_max - X_min). <strong>Standardization (Z-Score Scaling)</strong> transforms values so they have a mean of 0 and a standard deviation of 1: Z = (X - μ) / σ.",
        "example": "In our project: We used <strong>Normalization</strong> for our image pixel inputs (dividing RGB values 0–255 by 255.0 to get [0, 1]); we used <strong>Standardization</strong> for patient blood lab tests (creatinine, hemoglobin) to preserve extreme outlier signals without compressing the distribution.",
        "viva": "Normalization bounds features strictly between 0 and 1 (sensitive to outliers), while Standardization centers data to mean 0 and unit variance without bounding the range (robust to outliers).",
        "remember": "Use Min-Max Normalization when data has a bounded range (like image pixels); use Z-score Standardization when data has outliers or follows a Gaussian distribution."
    },
    {
        "num": 16,
        "title": "Overfitting vs Underfitting",
        "def": "<strong>Overfitting</strong> occurs when a model learns training data details and noise too closely, scoring near 100% on training data but failing completely on unseen test data (high variance). <strong>Underfitting</strong> occurs when a model is too simple to capture underlying patterns, performing poorly on both training and test data (high bias).",
        "example": "In our project: An unconstrained Decision Tree of depth 50 memorized all 1,200 training patients (Train Acc = 99.8%, Test Acc = 48.1%) — this was Overfitting. A linear model with 2 features had 49% accuracy on both — this was Underfitting. Our regularized LightGBM achieved a balanced 57.6% test accuracy.",
        "viva": "Overfitting is when a model memorizes training noise and fails to generalize to test data; Underfitting is when a model is too simplistic to capture even the training patterns.",
        "remember": "Overfitting = High Train Accuracy, Low Test Accuracy. Underfitting = Low Train Accuracy, Low Test Accuracy. Good Fit = High Train & Test Accuracy."
    },
    {
        "num": 17,
        "title": "Bias vs Variance",
        "def": "<strong>Bias</strong> is the error introduced by approximating a complex real-world problem with an overly simplistic model (causing underfitting). <strong>Variance</strong> is the error from the model's sensitivity to small fluctuations and noise in the training set (causing overfitting). The goal is the optimal <strong>Bias-Variance Tradeoff</strong>.",
        "example": "A single linear model has High Bias (assumes linear toxicity across all drug doses). An unpruned 100-leaf Decision Tree has High Variance (erratic predictions based on single patient anomalies). An ensemble like Random Forest balances both.",
        "viva": "Bias represents simplifying assumptions leading to underfitting, while Variance represents excessive model sensitivity to training noise leading to overfitting. The objective is finding the sweet spot that minimizes total error.",
        "remember": "Total Error = Bias² + Variance + Irreducible Error."
    },
    {
        "num": 18,
        "title": "Model Training and Prediction",
        "def": "<strong>Model Training</strong> (fitting) is the phase where an algorithm analyzes training data, computes prediction errors against true labels, and iteratively tunes its internal parameters (weights, coefficients, tree split thresholds). <strong>Prediction</strong> (inference) is when the finalized, frozen model accepts new, unseen feature inputs and outputs estimated targets or probabilities.",
        "example": "We trained Candidate V4 using LightGBM on 1,200 patient records using CPU/GPU resources (taking 3 minutes). At inference time, our FastAPI REST service loads <code>model.joblib</code> and predicts a new patient's toxicity in under 15 milliseconds.",
        "viva": "Training is the offline computationally heavy phase of learning parameters from labeled data; Prediction is the lightweight real-time inference phase where new inputs pass through the frozen model to produce outputs.",
        "remember": "Training happens once or periodically offline; Prediction runs continuously in production."
    },
    {
        "num": 19,
        "title": "Hyperparameters vs Parameters",
        "def": "<strong>Parameters</strong> are the internal variables learned and adjusted automatically by the algorithm during training (e.g., weights, biases, tree split thresholds). <strong>Hyperparameters</strong> are the external configuration settings chosen manually by the engineer before training begins to govern the learning process.",
        "example": "In our project: The coefficients assigned to <code>creatinine_level</code> and the split thresholds in LightGBM trees are <strong>Parameters</strong>. The number of trees (<code>n_estimators=300</code>), tree depth (<code>max_depth=5</code>), and learning rate (<code>learning_rate=0.05</code>) are <strong>Hyperparameters</strong>.",
        "viva": "Parameters are learned internally from data during training (weights, biases), while Hyperparameters are external configuration knobs tuned by the engineer before training (learning rate, tree depth, K in KNN).",
        "remember": "Parameters = Learned by Model (Internal). Hyperparameters = Set by Engineer (External)."
    },
    {
        "num": 20,
        "title": "Cross-Validation",
        "def": "Cross-Validation is a resampling technique used to evaluate how well an ML model will generalize to an independent dataset. In <strong>K-Fold Cross-Validation</strong>, the data is split into K equal folds; the model is trained on K-1 folds and validated on the remaining fold, repeating this K times so every sample is evaluated once.",
        "example": "In our clinical project, we used <strong>5-Fold Stratified Group K-Fold Cross-Validation</strong>, grouping by <code>patient_id</code>. This ensured that all visits of any patient remained in the same fold and class proportions (Low/Mod/High) were preserved across all folds.",
        "viva": "Cross-validation splits data into K subsets to train on K-1 folds and evaluate on the hold-out fold, averaging performance across all K iterations to provide an unbiased, robust estimate of generalization.",
        "remember": "Always use <strong>Stratified K-Fold</strong> for imbalanced classification and <strong>Group K-Fold</strong> when multiple records belong to the same patient/user to prevent leakage."
    },
    {
        "num": 21,
        "title": "ML Pipeline",
        "def": "An ML Pipeline is an automated, end-to-end sequence of data processing and modeling steps assembled into a single reproducible workflow. It binds preprocessing (imputation, encoding, scaling) together with the estimator (model) so that all transformations are applied consistently to training, validation, and test data.",
        "example": "Using Scikit-Learn's <code>Pipeline</code>, we bundled: <code>ColumnTransformer</code> (Median Imputer + StandardScaler for labs, SimpleImputer + OneHotEncoder for cancer types) &rarr; <code>LightGBM Classifier</code>, exported as a single frozen <code>pipeline.joblib</code> file.",
        "viva": "An ML pipeline chains data ingestion, imputation, transformation, and model estimation into an atomic, reproducible object, preventing data leakage and guaranteeing identical preprocessing during production inference.",
        "remember": "Fitting a pipeline on raw training data ensures that test data is transformed strictly using training statistics without manual intervention."
    },
    {
        "num": 22,
        "title": "Model Evaluation Metrics",
        "def": "Evaluation metrics quantitatively measure how well a trained model performs on unseen test data. Different tasks require different metrics to capture specific types of errors.",
        "example": "In our project: Stage 1 ML evaluated Classification metrics (Accuracy, Macro F1, High-Risk Recall, Confusion Matrix) on 1,750 test encounters; Stage 2 evaluated Regression metrics (MAE, RMSE, R²) for 30-day forward ctDNA prediction.",
        "viva": "Here is the clear definition of the standard metrics:<br>"
               "• <strong>Accuracy:</strong> Total correct predictions divided by total predictions: (TP + TN) / Total. Misleading when classes are imbalanced.<br>"
               "• <strong>Precision:</strong> Of all predicted positive cases, how many were truly positive? TP / (TP + FP). Measures false alarm rate.<br>"
               "• <strong>Recall (Sensitivity):</strong> Of all actual positive cases, how many did the model capture? TP / (TP + FN). Safety-critical in oncology.<br>"
               "• <strong>F1-Score:</strong> Harmonic mean of Precision and Recall: 2 * (P * R) / (P + R). Balances both false positives and false negatives.<br>"
               "• <strong>Confusion Matrix:</strong> A 2D/3D table showing True Positives, True Negatives, False Positives, and False Negatives across all classes.<br>"
               "• <strong>MAE (Mean Absolute Error):</strong> Average absolute difference between predicted and actual values: (1/n) * Σ |y - ŷ|.<br>"
               "• <strong>MSE (Mean Squared Error):</strong> Average squared difference: (1/n) * Σ (y - ŷ)². Penalizes large outlier errors heavily.<br>"
               "• <strong>RMSE (Root Mean Squared Error):</strong> Square root of MSE, returning error units back to the original target scale.<br>"
               "• <strong>R² (Coefficient of Determination):</strong> Proportion of target variance explained by the model, ranging from 0 to 1 (1.0 = perfect fit).",
        "remember": "For clinical safety, <strong>Recall</strong> is king (missing a high-risk patient is fatal). In regression, <strong>RMSE</strong> penalizes dangerous large errors."
    },
    {
        "num": 23,
        "title": "When Each Evaluation Metric Should Be Used",
        "def": "Choosing the right metric depends entirely on the business and clinical consequences of specific errors (False Positives vs False Negatives) and the class balance of the dataset.",
        "example": "In our cancer toxicity model: Predicting Low risk for a patient who actually suffers High toxicity (False Negative) could cause fatal organ failure. Thus, we prioritized <strong>High-Risk Recall (0.6287)</strong> and <strong>Macro F1 (0.5288)</strong> over overall Accuracy (0.5766).",
        "viva": "• Use <strong>Accuracy</strong> only when classes are perfectly balanced and all error types have identical real-world cost.<br>"
               "• Use <strong>Precision</strong> when False Positives are very costly (e.g., spam detection, canceling non-toxic chemotherapy).<br>"
               "• Use <strong>Recall</strong> when False Negatives are dangerous or fatal (e.g., cancer detection, high-toxicity risk prediction, fraud detection).<br>"
               "• Use <strong>F1-Score / Macro F1</strong> when data is imbalanced and you need a balance between finding positives and avoiding false alarms.<br>"
               "• Use <strong>MAE</strong> for regression when you want an easily interpretable average error robust to outliers.<br>"
               "• Use <strong>RMSE</strong> for regression when large errors are disproportionately catastrophic and must be heavily penalized.<br>"
               "• Use <strong>R²</strong> to explain what percentage of variance the regression model captures compared to a baseline mean model.",
        "remember": "Never use Accuracy alone for medical diagnosis or imbalanced data! An examiner will always ask: 'What if 95% of patients are healthy?'"
    },
    {
        "num": 24,
        "title": "Simple Real-World Examples for Important Concepts",
        "def": "To master viva examinations, every core machine learning concept should be immediately anchored to a practical, intuitive example from our real-world oncology project.",
        "example": "In our project: <strong>Features</strong> = Patient age, tumor stage, creatinine, drug dose. <strong>Target</strong> = Toxicity Risk (Low, Moderate, High). <strong>Baseline Model</strong> = Logistic Regression (Macro F1 = 0.46). <strong>Final Model</strong> = Candidate V4 LightGBM (Macro F1 = 0.5288, High-Risk Recall = 0.6287).",
        "viva": "In our Precision Oncology project, we take 25 clinical features, handle missing labs via median imputation, apply stratified group splitting by patient ID, train a regularized LightGBM classifier, and evaluate safety using High-Risk Recall to ensure no vulnerable patient suffers unmonitored severe adverse toxicity.",
        "remember": "Keep your project examples front and center: examiners reward candidates who explain concepts through their own actual implemented systems."
    }
]
