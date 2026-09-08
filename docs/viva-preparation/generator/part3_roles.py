"""
PART 3 — DATA SCIENCE ENGINEERING ROLES
Explains the 5 core engineering roles in simple terms, completely grounded in our
Personalized Precision Medicine for Oncology Treatment Optimization project.
"""

ROLES_DATA = [
    {
        "role": "Data Engineer",
        "tag": "FOUNDATION & INFRASTRUCTURE",
        "summary": "The Data Engineer designs, builds, and maintains the data infrastructure, pipelines, and databases that ingest raw, messy records and transform them into clean, reliable, and standardized datasets for downstream analysis and machine learning.",
        "what_they_do": "Data Engineers focus on the plumbing of data science. They ensure data arrives reliably from electronic health record (EHR) systems, hospital databases, and digital scanners, is validated against strict schemas, cleaned of corrupt entries, and stored securely in queryable databases.",
        "data_collection": "Automates the continuous extraction of raw clinical records, laboratory blood panels, genomic mutation profiles, and pathology image files from heterogeneous hospital source systems.",
        "etl_elt": "• <strong>ETL (Extract, Transform, Load):</strong> Data is extracted from source systems, transformed (cleaned, standardized, validated) on a processing server, and then loaded into a target data warehouse.<br>• <strong>ELT (Extract, Load, Transform):</strong> Raw data is immediately loaded into a scalable data lake/warehouse (like Snowflake or BigQuery) and transformed inside the warehouse using SQL.",
        "data_pipelines": "Automated DAGs (Directed Acyclic Graphs) that run scheduled data ingestion, unit testing, schema validation, and partition generation without manual human intervention.",
        "databases": "• <strong>Relational (SQL - PostgreSQL, MySQL):</strong> Stores structured patient demographic, visit encounter, and drug dosage tables with strict foreign-key integrity.<br>• <strong>NoSQL / Object Storage (MongoDB, AWS S3, MinIO):</strong> Stores semi-structured JSON telemetry, genomic variant files, and multi-gigabyte Whole Slide Images (WSI) and tiles.",
        "data_cleaning_storage": "Enforces type safety, deduplicates encounter records, standardizes units (e.g., converting all hemoglobin values to g/dL), handles corrupt image files, and creates immutable versioned data artifacts.",
        "tools": "Python, Pandas, Polars, SQL, Apache Spark, Apache Airflow, Pydantic, PostgreSQL, Docker, AWS S3 / MinIO.",
        "project_example": "In our Oncology project (<code>stage-1-ml/data-engineering</code>), the Data Engineer wrote automated pipelines that ingested raw EHR visits, validated schemas with Pydantic, imputed missing values according to strict clinical rules, generated the clean <code>master_patient_dataset.csv</code>, and enforced a <strong>strict patient-level split (700 Train / 150 Val / 150 Test) guaranteeing 0% data leakage</strong>.",
        "viva_q1": "What does a Data Engineer do?",
        "viva_a1": "A Data Engineer builds and maintains the automated data ingestion pipelines, ETL/ELT workflows, databases, and storage architectures that transform raw, messy data into clean, validated, and high-quality datasets for data scientists and ML models.",
        "viva_q2": "How did the Data Engineer prevent data leakage in our oncology project?",
        "viva_a2": "The Data Engineer implemented a patient-level split (using patient_id rather than random row splitting). This ensured that all historical visits and encounters for any given patient existed strictly within either the training set or the test set, preventing the model from 'cheating' by memorizing patient-specific baseline profiles."
    },
    {
        "role": "EDA Engineer (Exploratory Data Analysis)",
        "tag": "DATA UNDERSTANDING & DIAGNOSTICS",
        "summary": "The EDA Engineer investigates, visualizes, and statistically summarizes datasets before any modeling begins. They uncover underlying distributions, detect anomalies, check for target imbalances, verify data integrity, and discover predictive signals.",
        "what_they_do": "The EDA Engineer acts like a detective inspecting the crime scene. They do not assume the data is correct. They formulate hypotheses, compute summary statistics, plot visual distributions, identify extreme outliers, and check whether features correlate with the target or with each other.",
        "why_eda": "To understand data characteristics, discover anomalies/errors, ensure assumptions of ML models are met, prevent data leakage, and guide effective feature engineering. Training a model without EDA is like driving with your eyes closed.",
        "components": "• <strong>Data Distribution:</strong> Checking whether variables follow Gaussian (normal) bell curves or skewed distributions (e.g., right-skewed creatinine and liver enzyme levels).<br>"
                      "• <strong>Missing Values Analysis:</strong> Visualizing missingness patterns using heatmaps to determine if data is Missing Completely at Random (MCAR) or systematically missing.<br>"
                      "• <strong>Outlier Detection:</strong> Identifying physiologically impossible values (e.g., heart rate = 0 while patient is active) using the Interquartile Range (IQR) rule (Q1 - 1.5*IQR to Q3 + 1.5*IQR) and Z-scores (|Z| > 3).<br>"
                      "• <strong>Correlation Analysis:</strong> Computing Pearson (linear) and Spearman (monotonic rank) correlation matrices to detect multi-collinearity between clinical biomarkers.<br>"
                      "• <strong>Visualization:</strong> Histograms, KDE density plots, box plots, violin plots, and correlation heatmaps.<br>"
                      "• <strong>Univariate Analysis:</strong> Analyzing 1 variable at a time (e.g., histogram of patient ages).<br>"
                      "• <strong>Bivariate Analysis:</strong> Analyzing relationships between 2 variables (e.g., drug dose vs toxicity risk category using grouped box plots).<br>"
                      "• <strong>Multivariate Analysis:</strong> Analyzing 3 or more variables simultaneously (e.g., pairplots, 3D PCA projections, or correlation heatmaps across all 25 labs).",
        "tools": "Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, SciPy, SweetViz, YData-Profiling.",
        "project_example": "In <code>stage-1-ml/eda</code>, the EDA Engineer generated correlation heatmaps showing that elevated <code>creatinine_level</code> and high <code>drug_dose</code> strongly correlated with <strong>High Toxicity Risk</strong>. They also verified that the target variable had 42% Low, 36% Moderate, and 22% High risk encounters, guiding our decision to adopt Macro F1 instead of raw accuracy.",
        "viva_q1": "What is EDA and why is it essential before training an ML model?",
        "viva_a1": "EDA is the critical preliminary process of performing initial statistical investigations and visual analyses on data to uncover patterns, spot anomalies, test clinical hypotheses, and check distribution assumptions before building machine learning models.",
        "viva_q2": "How would you detect outliers in patient blood laboratory tests?",
        "viva_a2": "I would use box plots to visually identify outliers and apply the statistical Interquartile Range (IQR) method: any value below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is flagged as an outlier and inspected clinically to determine whether it is an instrumentation error or a genuine acute physiological crisis."
    },
    {
        "role": "ML Engineer (Machine Learning Engineer)",
        "tag": "MODELING & PRODUCTION PIPELINES",
        "summary": "The ML Engineer bridges the gap between theoretical data science prototypes and robust production systems. They design, train, tune, optimize, and package machine learning models into reliable, reproducible, and deployable software pipelines.",
        "what_they_do": "An ML Engineer takes cleaned data from the Data Engineer and insights from the EDA Engineer to build high-performance predictive models. They experiment with multiple algorithms, formulate loss functions, tune hyperparameters, implement cross-validation, prevent overfitting, serialize models into deployable artifacts (e.g., <code>model.joblib</code> or ONNX), and monitor performance.",
        "core_tasks": "• <strong>Data Preparation:</strong> Building Scikit-Learn transformers and pipelines that bundle imputation, encoding, and scaling.<br>"
                      "• <strong>Model Selection:</strong> Benchmarking diverse algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM) to find the best speed-accuracy-interpretability tradeoff.<br>"
                      "• <strong>Hyperparameter Tuning:</strong> Searching parameter spaces using Grid Search, Random Search, or Bayesian Optimization (Optuna) to maximize validation Macro F1.<br>"
                      "• <strong>Model Deployment:</strong> Freezing model weights, packaging artifacts with exact dependency versions, and enabling fast local/cloud inference.<br>"
                      "• <strong>Monitoring:</strong> Tracking model drift, concept drift, latency, and prediction distribution shifts in live production.",
        "ds_vs_mle": "• <strong>Data Scientist:</strong> Focuses on statistical exploration, hypothesis testing, training prototype models in Jupyter notebooks, and extracting clinical insights.<br>"
                    "• <strong>ML Engineer:</strong> Focuses on software engineering, reproducible training pipelines, code optimization, model serving, CI/CD, scalability, and latency.",
        "tools": "Scikit-Learn, LightGBM, XGBoost, PyTorch, MLflow, Optuna, Joblib, Docker, Git.",
        "project_example": "In <code>stage-1-ml/ml</code>, the ML Engineer trained and compared 4 model candidates. Candidate V4 (regularized LightGBM with <code>learning_rate=0.05</code>, <code>max_depth=5</code>, and conservative decision rules) was selected as the champion model. They exported the complete preprocessor and model as <code>model.joblib</code> with 100% reproducible training scripts.",
        "viva_q1": "What is the difference between a Data Scientist and an ML Engineer?",
        "viva_a1": "A Data Scientist focuses on data analysis, statistical experimentation, business questions, and prototyping models in notebooks; an ML Engineer focuses on building production-grade software pipelines, optimizing training code, deploying models via APIs, and monitoring model reliability in production.",
        "viva_q2": "Why did you choose LightGBM (Candidate V4) over a deep neural network for Stage 1?",
        "viva_a2": "For structured tabular clinical data with 1,750 records and 25 features, tree-based gradient boosting (LightGBM) outperforms deep neural networks because it handles mixed categorical/numerical features naturally, is robust to unscaled outliers, requires significantly less tuning and compute, and provides clear feature importances for clinical interpretability."
    },
    {
        "role": "Evaluation Engineer",
        "tag": "INDEPENDENT AUDITING & CLINICAL SAFETY",
        "summary": "The Evaluation Engineer acts as an independent auditor of model quality. They design rigorous, unbiased testing protocols, calculate statistical confidence intervals, conduct error analysis, and ensure the model performs safely, fairly, and reliably across all patient subgroups.",
        "what_they_do": "The Evaluation Engineer does not build or tune the model. By maintaining complete independence from the ML Engineer, they prevent confirmation bias. They take the finalized frozen model artifact, test it exclusively on the locked test cohort, compute primary and safety metrics, generate confusion matrices, run bootstrap confidence intervals, and analyze edge-case failures.",
        "why_important": "A model that achieves 90% training accuracy may fail catastrophically in clinical practice. The Evaluation Engineer ensures that models are clinically safe, robust against distributional shifts, and do not suffer from hidden failure modes in vulnerable sub-populations.",
        "core_tasks": "• <strong>Metric Selection:</strong> Selecting domain-appropriate metrics (prioritizing High-Risk Recall and Macro F1 over simple accuracy for oncology safety).<br>"
                      "• <strong>Independent Locked Testing:</strong> Running evaluation scripts strictly on unseen held-out test data that was never touched during training or tuning.<br>"
                      "• <strong>Confidence Intervals:</strong> Computing 95% Bootstrap Confidence Intervals (e.g., resampling test data 1,000 times) to prove statistical stability.<br>"
                      "• <strong>Subgroup Analysis:</strong> Benchmarking performance across 26 distinct demographic and clinical cohorts (e.g., Elderly patients >70, Stage IV metastatic cancer, EGFR mutants) to guarantee fairness and safety.<br>"
                      "• <strong>Error Transition Analysis:</strong> Dissecting the confusion matrix to investigate dangerous false negatives (patients who were predicted Low Risk but actually suffered High Toxicity).",
        "tools": "Scikit-Learn metrics, SciPy, Bootstrap Resampling, Matplotlib/Seaborn for confusion matrices, Pandas, PyTest.",
        "project_example": "In <code>stage-1-ml/evaluation</code>, the Evaluation Engineer independently benchmarked Candidate V4 on the locked test set (1,750 encounters across 1,200 unique patients). They certified that **Macro F1 was 0.5288 [95% CI: 0.5035, 0.5521]**, **High-Risk Safety Recall was 0.6287 [95% CI: 0.5759, 0.6783]**, and verified consistent performance across all 26 clinical cohorts.",
        "viva_q1": "Why do we need a separate Evaluation Engineer instead of letting the ML Engineer evaluate their own model?",
        "viva_a1": "A separate Evaluation Engineer prevents developer bias and overfitting to the test set. Just like software testing requires independent QA, clinical AI requires an independent auditor to rigorously stress-test the model on locked data and ensure patient safety before clinical deployment.",
        "viva_q2": "What was the most critical metric evaluated in your project and why?",
        "viva_a2": "High-Risk Recall (0.6287). In oncology precision medicine, missing a high-risk patient (False Negative) could lead to catastrophic, unmonitored chemotherapy toxicity or organ failure. Therefore, maximizing Recall for the High-Risk cohort is our primary clinical safety constraint."
    },
    {
        "role": "Integration Engineer",
        "tag": "SYSTEM BRIDGING & CLINICAL DEPLOYMENT",
        "summary": "The Integration Engineer connects trained AI/ML models into real-world hospital software ecosystems. They build high-speed REST APIs, handle input/output schema validation, integrate frontend clinical dashboards, connect databases, manage model latency, and fuse multimodal predictions.",
        "what_they_do": "An ML model in a file is useless until doctors and hospital software can use it. The Integration Engineer embeds the frozen model into production microservices, defines strict input validation schemas (so bad data cannot crash the service), handles serialization/deserialization, optimizes inference latency, and builds intuitive user interfaces for medical staff.",
        "core_tasks": "• <strong>API Development:</strong> Creating production RESTful API endpoints using FastAPI or Flask (e.g., <code>/health</code>, <code>/predict</code>, <code>/predict/batch</code>).<br>"
                      "• <strong>Request/Response Validation:</strong> Using Pydantic models to strictly enforce data types, physiological ranges, and required fields before passing data to the model.<br>"
                      "• <strong>Multimodal Late Fusion:</strong> Combining vision predictions from the CNN pathology service with temporal predictions from the LSTM biomarker service into a single unified clinical risk alert.<br>"
                      "• <strong>Dashboard & UI Integration:</strong> Connecting the backend API to clinical frontends (HTML5, JavaScript, modern dashboards) so oncologists can review risk scores and alerts in real-time.<br>"
                      "• <strong>Latency Benchmarking & Reliability:</strong> Guaranteeing sub-50ms inference latency, writing automated integration tests, and orchestrating containers with Docker.",
        "tools": "FastAPI, Uvicorn, Pydantic, Requests, Docker, HTML/CSS/JavaScript, PyTest, NGINX, Redis.",
        "project_example": "In <code>stage-1-ml/integration</code> and <code>stage-2-dl/integration</code>, the Integration Engineer built a high-speed FastAPI REST microservice with Pydantic validation. They implemented <code>fusion_service.py</code> to combine Stage 1 tabular toxicity risk + Stage 2 CNN pathology score + LSTM ctDNA forecast into a composite patient alert tier (<strong>SAFE, MONITOR, CRITICAL</strong>) displayed on an interactive clinical dashboard (<code>dashboard.html</code>), validated with 40 automated passing tests.",
        "viva_q1": "What is the role of an Integration Engineer in an AI/ML project?",
        "viva_a1": "The Integration Engineer deploys ML models into production by building secure, high-speed REST APIs, enforcing strict input validation, integrating models with frontend dashboards and hospital databases, and ensuring sub-second inference latency with robust automated testing.",
        "viva_q2": "Explain the API endpoints you built in your precision oncology system.",
        "viva_a2": "We built a FastAPI service with three core endpoints: <code>GET /health</code> to verify model status and memory health, <code>POST /predict</code> to accept a single patient encounter payload and return toxicity probabilities in under 15ms, and <code>POST /predict/batch</code> to process multiple patient records concurrently for hospital ward batches."
    }
]
