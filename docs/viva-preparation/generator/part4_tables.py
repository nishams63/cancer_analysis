"""
PART 4 — IMPORTANT DIFFERENCES
Contains all 13 requested comparison tables in clean, structured dictionary format.
"""

COMPARISON_TABLES = [
    {
        "num": 1,
        "title": "AI vs ML vs DL",
        "headers": ["Dimension", "Artificial Intelligence (AI)", "Machine Learning (ML)", "Deep Learning (DL)", "Our Oncology Project Context"],
        "rows": [
            [
                "Core Concept",
                "Broad science of creating machines that mimic human intelligence and decision-making.",
                "Subset of AI where algorithms learn statistical patterns from structured data.",
                "Subset of ML using multi-layered artificial neural networks for raw unstructured data.",
                "AI is the autonomous clinical system; ML predicts toxicity; DL analyzes pathology tiles."
            ],
            [
                "Scope",
                "Superset encompassing ML, DL, robotics, NLP, and expert rule engines.",
                "Subset of AI; relies on data and statistical optimization.",
                "Specialized subset of ML; relies on deep neural architectures.",
                "Integrated multi-agent decision support system."
            ],
            [
                "Feature Engineering",
                "Manually engineered rules or data algorithms depending on approach.",
                "Crucial human-engineered features (ratios, encodings, scaling).",
                "Automatic hierarchical feature representation learned directly from raw inputs.",
                "Stage 1 ML uses hand-crafted lab features; Stage 2 DL uses raw 224x224 RGB pixels."
            ],
            [
                "Data Requirement",
                "Can operate on rules, knowledge graphs, or data.",
                "Performs effectively on small-to-medium tabular datasets (1,000s of samples).",
                "Demands massive data volumes (10,000s to millions of samples).",
                "1,750 encounters for LightGBM vs 12,000 tiles and 16,012 temporal observations for DL."
            ],
            [
                "Hardware",
                "Standard CPU or specialized processors depending on system.",
                "Standard multi-core CPUs; low compute footprint.",
                "High-end GPUs / TPUs required for matrix tensor parallel operations.",
                "Stage 1 trained on CPU in 3 minutes; Stage 2 CNN trained on GPU."
            ]
        ]
    },
    {
        "num": 2,
        "title": "ML vs DL",
        "headers": ["Factor", "Machine Learning (Classical ML)", "Deep Learning (DL)", "Project Application"],
        "rows": [
            [
                "Data Representation",
                "Structured tabular matrices (rows = patients, columns = lab features).",
                "High-dimensional raw sensory data (2D image grids, 3D volumes, sequential time series).",
                "Stage 1: Tabular patient CSV<br>Stage 2: 224x224 H&E biopsy tiles + serial ctDNA."
            ],
            [
                "Feature Extraction",
                "Manual domain-driven feature engineering by human data scientists.",
                "Automated end-to-end representation learning via convolutional or attention layers.",
                "Stage 1: Engineered creatinine-to-dose ratio.<br>Stage 2: CNN learns tumor cell morphology automatically."
            ],
            [
                "Model Interpretability",
                "High interpretability (tree split thresholds, SHAP values, feature importance).",
                "Low to moderate ('Black box'); requires Grad-CAM or integrated gradients to visualize.",
                "LightGBM feature importances show clinicians exactly why a toxicity score was assigned."
            ],
            [
                "Performance Scaling",
                "Plateaus after reaching moderate dataset sizes.",
                "Continues improving accuracy as dataset volume and model parameters increase.",
                "LightGBM reached peak performance at 1,200 patients; CNN improves with more slide tiles."
            ],
            [
                "Primary Algorithms",
                "Linear/Logistic Regression, Decision Trees, Random Forest, LightGBM, SVM, KNN.",
                "CNNs (ResNet, ConvNet), RNNs, LSTMs, Transformers.",
                "Stage 1: LightGBM Candidate V4.<br>Stage 2: 2D CNN + 1D LSTM."
            ]
        ]
    },
    {
        "num": 3,
        "title": "Classification vs Regression",
        "headers": ["Criteria", "Classification", "Regression", "Project Example"],
        "rows": [
            [
                "Target Variable Type",
                "Discrete, qualitative categorical classes or labels.",
                "Continuous, quantitative real numerical values.",
                "Classification: Low/Mod/High Toxicity.<br>Regression: 30-day ctDNA level (ng/mL)."
            ],
            [
                "Model Output",
                "Class label (or predicted probability distribution across classes).",
                "Specific floating-point numerical prediction.",
                "Class: [P(Low)=0.12, P(Mod)=0.25, P(High)=0.63]<br>Regress: ctDNA = 2.45 ng/mL."
            ],
            [
                "Evaluation Metrics",
                "Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.",
                "MAE (Mean Absolute Error), MSE, RMSE, R² (Coefficient of Determination).",
                "Evaluated using Macro F1 (0.5288) for risk vs RMSE (0.18 ng/mL) for ctDNA levels."
            ],
            [
                "Clinical Objective",
                "Sorting patients into treatment risk categories or diagnostic groups.",
                "Forecasting future lab values, drug concentration, or survival days.",
                "Classification flags high-risk patients; Regression predicts exact progression rate."
            ]
        ]
    },
    {
        "num": 4,
        "title": "Supervised vs Unsupervised Learning",
        "headers": ["Aspect", "Supervised Learning", "Unsupervised Learning", "Project Application"],
        "rows": [
            [
                "Ground Truth Target",
                "Explicit ground-truth target labels (y) are provided for every sample.",
                "No target labels (y); only input features (X) are provided.",
                "Supervised: Patient has known toxicity outcome.<br>Unsupervised: Raw patient biomarkers without labels."
            ],
            [
                "Primary Goal",
                "Map inputs X to known outputs y (learn f(X) &approx; y).",
                "Discover hidden structures, groupings, or feature compressions in X.",
                "Supervised: Predict toxicity risk.<br>Unsupervised: Cluster patients into novel sub-phenotypes."
            ],
            [
                "Primary Tasks",
                "Classification and Regression.",
                "Clustering, Dimensionality Reduction, and Anomaly Detection.",
                "Supervised: LightGBM classification.<br>Unsupervised: K-Means patient clustering & PCA."
            ],
            [
                "Feedback / Evaluation",
                "Direct error feedback comparing predictions against true labels.",
                "Subjective or heuristic evaluation (Silhouette score, Inertia, explained variance).",
                "Supervised: Confusion matrix against clinical records.<br>Unsupervised: Silhouette analysis."
            ]
        ]
    },
    {
        "num": 5,
        "title": "Training vs Testing",
        "headers": ["Parameter", "Training Set", "Testing Set (Locked)", "Project Execution"],
        "rows": [
            [
                "Purpose",
                "Used by the ML algorithm to fit weights, learn coefficients, and split trees.",
                "Used exclusively to assess unbiased real-world generalization performance.",
                "Train on 700 patients (1,000+ visits); Test on 150 independent locked patients."
            ],
            [
                "Model Exposure",
                "Seen repeatedly by the model across iterations/epochs.",
                "Completely unseen; locked away until final candidate evaluation.",
                "Model weights were frozen before touching the locked test cohort."
            ],
            [
                "Data Leakage Risk",
                "Imputers and scalers are fitted strictly on this data.",
                "Must NEVER be used to compute mean, median, or scaling parameters.",
                "Preprocessors were fitted on Train and only applied to Test via transform()."
            ],
            [
                "Splitting Method",
                "Typically 70%–80% of total data.",
                "Typically 15%–20% of total data.",
                "Patient-level grouping ensured 0% patient leakage between Train and Test."
            ]
        ]
    },
    {
        "num": 6,
        "title": "Parameter vs Hyperparameter",
        "headers": ["Comparison", "Model Parameter", "Hyperparameter", "Project Example"],
        "rows": [
            [
                "Origin",
                "Learned internally and adjusted automatically from data during training.",
                "Set externally by the engineer before model training begins.",
                "Parameter: Tree split thresholds.<br>Hyperparameter: max_depth, learning_rate."
            ],
            [
                "Manual Setting",
                "Cannot be set manually by the engineer.",
                "Configured manually or tuned via Grid Search, Random Search, or Optuna.",
                "Engineers tuned learning_rate to 0.05 using cross-validation."
            ],
            [
                "Storage",
                "Saved inside the model artifact file (e.g., <code>model.joblib</code> or <code>.pt</code>).",
                "Defined in configuration scripts or training arguments (JSON / YAML).",
                "Candidate V4 parameters saved in joblib; config in <code>v4_candidate_config.json</code>."
            ],
            [
                "Examples",
                "Weights, biases, linear coefficients, tree split values.",
                "Tree depth (max_depth), n_estimators, learning rate, K in KNN, batch size.",
                "LightGBM: 300 trees (hyperparameter); exact split at creatinine=1.45 (parameter)."
            ]
        ]
    },
    {
        "num": 7,
        "title": "Overfitting vs Underfitting",
        "headers": ["Characteristic", "Underfitting (High Bias)", "Overfitting (High Variance)", "Optimal Fit (Our Project)"],
        "rows": [
            [
                "Training Error",
                "High training error (poor fit).",
                "Very low / near zero training error (memorization).",
                "Low, balanced training error."
            ],
            [
                "Testing Error",
                "High testing error (fails to predict).",
                "High testing error (fails to generalize).",
                "Low testing error matching training performance."
            ],
            [
                "Model Complexity",
                "Too simple (e.g., linear model on non-linear data).",
                "Too complex (e.g., 50-deep unpruned decision tree).",
                "Balanced complexity with regularization (LightGBM max_depth=5)."
            ],
            [
                "Cause",
                "Overly rigid assumptions; insufficient features.",
                "Model learns training noise and specific patient anomalies.",
                "Feature selection, cross-validation, and conservative thresholds."
            ],
            [
                "Remedy",
                "Add more features, reduce regularization, use more complex models.",
                "Add regularization, dropout, pruning, feature selection, or more data.",
                "Candidate V4 used L2 regularization and conservative decision rules."
            ]
        ]
    },
    {
        "num": 8,
        "title": "Normalization vs Standardization",
        "headers": ["Feature", "Normalization (Min-Max Scaling)", "Standardization (Z-Score Scaling)", "Project Application"],
        "rows": [
            [
                "Mathematical Formula",
                "X_norm = (X - X_min) / (X_max - X_min)",
                "Z = (X - &mu;) / &sigma;",
                "Both scale numerical features to comparable ranges."
            ],
            [
                "Output Bounding",
                "Strictly bounded between [0, 1] (or [-1, 1]).",
                "Unbounded; centered at mean = 0 with standard deviation = 1.",
                "Pixels bounded to [0, 1]; labs standardized to mean 0."
            ],
            [
                "Outlier Sensitivity",
                "Highly sensitive: outliers compress normal data into tiny ranges.",
                "Robust: preserves relative outlier distances in standard deviation units.",
                "Used Standardization for clinical lab tests containing acute spike outliers."
            ],
            [
                "Primary Use Cases",
                "Image pixel values (0–255 &rarr; 0–1), neural network inputs, KNN.",
                "Algorithms assuming Gaussian distributions (Logistic Regression, SVM, PCA).",
                "Stage 2 CNN images normalized; Stage 1 blood biomarkers standardized."
            ]
        ]
    },
    {
        "num": 9,
        "title": "Data Engineer vs Data Scientist",
        "headers": ["Focus Area", "Data Engineer", "Data Scientist", "Project Collaboration"],
        "rows": [
            [
                "Primary Mission",
                "Build scalable, reliable data pipelines, databases, and clean datasets.",
                "Analyze data, uncover clinical insights, and build predictive statistical models.",
                "DE provides clean data; DS formulates hypotheses and trains baseline models."
            ],
            [
                "Core Deliverable",
                "Clean data lakes, ETL pipelines, validated master tables (e.g., <code>master_patient_dataset.csv</code>).",
                "Statistical reports, feature significance, prototype models, and insight presentations.",
                "DE generated the 1,750-encounter dataset; DS verified clinical predictive power."
            ],
            [
                "Primary Mindset",
                "Software engineering, data integrity, pipeline reliability, schema validation.",
                "Scientific exploration, hypothesis testing, math/statistics, domain research.",
                "DE ensures 0% patient leakage; DS tests whether biomarkers predict toxicity."
            ],
            [
                "Primary Tools",
                "SQL, Spark, Airflow, Pydantic, Kafka, Docker, PostgreSQL, S3.",
                "Python/R, Pandas, NumPy, Scikit-Learn, SciPy, Jupyter Notebooks.",
                "DE used Pydantic/Pandas for schema checks; DS used Scikit-Learn in notebooks."
            ]
        ]
    },
    {
        "num": 10,
        "title": "Data Scientist vs ML Engineer",
        "headers": ["Aspect", "Data Scientist", "ML Engineer", "Project Hand-off"],
        "rows": [
            [
                "Primary Focus",
                "Statistical experimentation, exploratory analysis, and model prototyping.",
                "Productionizing, scaling, optimizing, and deploying ML models as robust services.",
                "DS tested 3 model types in notebooks; MLE productionized Candidate V4."
            ],
            [
                "Environment",
                "Interactive notebooks (Jupyter/Colab) and statistical dashboards.",
                "Modular Python codebases, CI/CD pipelines, Docker containers, REST microservices.",
                "DS delivered <code>experiment.ipynb</code>; MLE wrote <code>train.py</code> and <code>app.py</code>."
            ],
            [
                "Code Requirements",
                "Focused on rapid experimentation and proving feasibility.",
                "High software quality: unit tests, type hinting, logging, latency optimization, CI/CD.",
                "MLE created automated test suites (40 passing tests) and Pydantic schemas."
            ],
            [
                "Deployment Ownership",
                "Typically hands off model weights or prototype code to the engineering team.",
                "Owns end-to-end model serving, REST APIs, Dockerization, and production monitoring.",
                "MLE packaged <code>model.joblib</code> and deployed FastAPI inference endpoints."
            ]
        ]
    },
    {
        "num": 11,
        "title": "EDA Engineer vs ML Engineer",
        "headers": ["Role Attribute", "EDA Engineer", "ML Engineer", "Project Hand-off"],
        "rows": [
            [
                "Primary Objective",
                "Understand data characteristics, distributions, anomalies, and clinical correlations.",
                "Engineer features, train algorithms, tune hyperparameters, and deploy models.",
                "EDA found right-skewed creatinine; MLE applied log-transform/median imputation."
            ],
            [
                "Key Deliverables",
                "Statistical summary reports, correlation matrices, outlier lists, distribution charts.",
                "Trained model artifacts (<code>model.joblib</code>), reproducible pipelines, prediction APIs.",
                "EDA delivered <code>eda_report.md</code>; MLE delivered Candidate V4 model."
            ],
            [
                "Analytical Approach",
                "Descriptive and diagnostic analytics (What happened? What is in the data?).",
                "Predictive and prescriptive analytics (What will happen? How do we automate it?).",
                "EDA analyzed past toxicity rates; MLE built automated future toxicity predictor."
            ]
        ]
    },
    {
        "num": 12,
        "title": "ML Engineer vs Evaluation Engineer",
        "headers": ["Dimension", "ML Engineer", "Evaluation Engineer", "Separation of Duties"],
        "rows": [
            [
                "Role Bias",
                "Creator bias: Motivated to optimize training metrics and reach high performance.",
                "Auditor neutrality: Unbiased verification of real-world safety and failure edge cases.",
                "MLE built Candidate V4; Eval Engineer verified it on locked test patients."
            ],
            [
                "Test Data Access",
                "Uses Train and Validation sets; NEVER touches the locked test set.",
                "Holds the keys to the locked test set; executes official independent benchmarks.",
                "Prevents test set snooping and accidental overfitting during model tuning."
            ],
            [
                "Core Responsibilities",
                "Algorithm selection, training loops, hyperparameter tuning, model packaging.",
                "Confidence interval calculation, 26-cohort subgroup analysis, error transition analysis.",
                "Eval Engineer verified High-Risk Recall is 0.6287 with 95% bootstrap CIs."
            ],
            [
                "Failure Analysis",
                "Adjusts loss functions and regularizations to fix training bottlenecks.",
                "Dissects confusion matrices to catch clinically dangerous false negatives.",
                "Eval Engineer flagged misclassified elderly patients for conservative override rules."
            ]
        ]
    },
    {
        "num": 13,
        "title": "ML Engineer vs Integration Engineer",
        "headers": ["Dimension", "ML Engineer", "Integration Engineer", "Clinical Bridge"],
        "rows": [
            [
                "Core Expertise",
                "Machine learning algorithms, loss functions, hyperparameter optimization, PyTorch.",
                "Software architecture, REST/gRPC APIs, backend systems, frontend UI, containerization.",
                "MLE builds the brain (model); Integration Engineer builds the nervous system (API/UI)."
            ],
            [
                "Primary Output",
                "Frozen model weights (<code>model.joblib</code>, <code>best_model.pt</code>).",
                "Production REST API service (FastAPI <code>/predict</code>), dashboard, multimodal fusion engine.",
                "Integration Engineer integrated <code>model.joblib</code> into <code>app.py</code> and <code>dashboard.html</code>."
            ],
            [
                "Latency Concern",
                "Focuses on training throughput (samples/sec) and convergence speed.",
                "Focuses on real-time inference latency (milliseconds per API call) and concurrent user load.",
                "Integration Engineer benchmarked API to ensure <15ms response time for doctors."
            ],
            [
                "System Scope",
                "Individual models (e.g., LightGBM or CNN).",
                "Unified multimodal systems (Late Fusion combining tabular ML + vision DL + time series).",
                "Built <code>fusion_service.py</code> to synthesize multi-stage predictions into clinical alerts."
            ]
        ]
    }
]
