"""Task Decomposer translating analytical objectives into discrete tasks."""
from __future__ import annotations
from typing import List
from schemas.task import (
    Task,
    TaskType,
    TaskStatus,
    TaskPriority,
    FailurePolicy,
    RetryPolicy,
    KnowledgeRetrievalSpec,
)
from schemas.condition import BranchCondition, ComparisonOperator, BranchAction
from planner.goal_parser import ParsedGoal


class TaskDecomposer:
    """Generates discrete, well-formed task definitions matching analytical intent."""

    @classmethod
    def decompose(cls, parsed_goal: ParsedGoal) -> List[Task]:
        """Select and instantiate task sequence based on intent type."""
        if parsed_goal.intent_type == "revenue_decline":
            return cls._build_revenue_decline_tasks(parsed_goal)
        elif parsed_goal.intent_type == "customer_analysis":
            return cls._build_customer_tasks(parsed_goal)
        elif parsed_goal.intent_type == "anomaly_investigation":
            return cls._build_anomaly_tasks(parsed_goal)
        elif parsed_goal.intent_type == "predictive_analysis":
            return cls._build_predictive_tasks(parsed_goal)
        else:
            return cls._build_general_analysis_tasks(parsed_goal)

    @staticmethod
    def _build_revenue_decline_tasks(goal: ParsedGoal) -> List[Task]:
        """Multi-branch revenue decline investigation tasks."""
        return [
            Task(
                task_id="T001",
                name="Load Sales Dataset",
                description=f"Ingest raw {goal.dataset} into analytical session.",
                task_type=TaskType.INGESTION,
                tool="load_dataset",
                dependencies=[],
                required_inputs=[goal.dataset],
                expected_outputs=["raw_dataframe"],
                priority=TaskPriority.CRITICAL,
            ),
            Task(
                task_id="T002",
                name="Profile Sales Dataset",
                description="Compute summary statistics, row counts, missing rates, and data types.",
                task_type=TaskType.PROFILING,
                tool="profile_dataset",
                dependencies=["T001"],
                required_inputs=["raw_dataframe"],
                expected_outputs=["profile_report", "missing_rate", "quality_score"],
            ),
            Task(
                task_id="T003",
                name="Validate Data Quality",
                description="Verify presence of required revenue columns, timestamps, and validity ranges.",
                task_type=TaskType.VALIDATION,
                tool="validate_data_quality",
                dependencies=["T002"],
                required_inputs=["profile_report"],
                expected_outputs=["quality_verdict", "needs_cleaning"],
                conditions=[
                    BranchCondition(
                        condition_id="COND-REV-CLEANING",
                        metric_name="needs_cleaning",
                        operator=ComparisonOperator.EQ,
                        threshold=True,
                        action=BranchAction.DATA_CLEANING,
                        target_task_id="T004",
                        description="Route to cleaning if missing values or invalid data detected",
                    ),
                    BranchCondition(
                        condition_id="COND-REV-CLEAN",
                        metric_name="needs_cleaning",
                        operator=ComparisonOperator.EQ,
                        threshold=False,
                        action=BranchAction.CONTINUE_ANALYSIS,
                        target_task_id="T005",
                        description="Proceed directly to trend analysis if data is clean",
                    ),
                ],
                knowledge_retrieval=KnowledgeRetrievalSpec(
                    query="Validation of missing numerical values and hard boundary limits",
                    category="data_quality",
                    top_k=2,
                ),
            ),
            Task(
                task_id="T004",
                name="Clean Dataset",
                description="Impute missing numerical/categorical values and trim anomalous recording errors.",
                task_type=TaskType.CLEANING,
                tool="clean_dataset",
                dependencies=["T003"],
                required_inputs=["raw_dataframe", "quality_verdict"],
                expected_outputs=["cleaned_dataframe"],
                knowledge_retrieval=KnowledgeRetrievalSpec(
                    query="Median imputation for skewed numerical data and robust winsorization",
                    category="data_cleaning",
                    top_k=2,
                ),
            ),
            Task(
                task_id="T005",
                name="Analyze Revenue Trend",
                description="Calculate longitudinal revenue trajectory and identify turning points.",
                task_type=TaskType.EDA,
                tool="eda_analysis",
                dependencies=["T003", "T004"],
                required_inputs=["cleaned_dataframe"],
                expected_outputs=["revenue_trend_metrics", "decline_magnitude", "decline_start_date"],
                conditions=[
                    BranchCondition(
                        condition_id="COND-DECLINE-CONFIRM",
                        metric_name="decline_magnitude",
                        operator=ComparisonOperator.GT,
                        threshold=0.0,
                        action=BranchAction.CONTINUE_ANALYSIS,
                        target_task_id=None,
                        description="Continue root-cause analysis branches if revenue decline confirmed",
                    )
                ],
            ),
            Task(
                task_id="T006",
                name="Analyze Product Performance Branch",
                description="Decompose revenue change by product hierarchy and SKU volume.",
                task_type=TaskType.ROOT_CAUSE_ANALYSIS,
                tool="root_cause_analysis",
                dependencies=["T005"],
                required_inputs=["revenue_trend_metrics"],
                expected_outputs=["product_contribution_delta"],
            ),
            Task(
                task_id="T007",
                name="Analyze Customer Segment Branch",
                description="Decompose revenue variance across customer tiers and tenure cohorts.",
                task_type=TaskType.ROOT_CAUSE_ANALYSIS,
                tool="root_cause_analysis",
                dependencies=["T005"],
                required_inputs=["revenue_trend_metrics"],
                expected_outputs=["customer_contribution_delta"],
            ),
            Task(
                task_id="T008",
                name="Analyze Regional & Channel Branch",
                description="Decompose revenue changes by geographic markets and sales channels.",
                task_type=TaskType.ROOT_CAUSE_ANALYSIS,
                tool="root_cause_analysis",
                dependencies=["T005"],
                required_inputs=["revenue_trend_metrics"],
                expected_outputs=["region_channel_contribution_delta"],
            ),
            Task(
                task_id="T009",
                name="Detect Revenue Anomalies Branch",
                description="Identify specific dates or accounts exhibiting abnormal revenue drops.",
                task_type=TaskType.ANOMALY_DETECTION,
                tool="detect_anomalies",
                dependencies=["T005"],
                required_inputs=["revenue_trend_metrics"],
                expected_outputs=["revenue_anomaly_list"],
                knowledge_retrieval=KnowledgeRetrievalSpec(
                    query="Time-series anomaly detection and residual spike isolation",
                    category="anomaly_detection",
                    top_k=2,
                ),
            ),
            Task(
                task_id="T010",
                name="Perform Rate vs Volume Mix Analysis",
                description="Consolidate all branch deltas and isolate pure price changes vs volume mix shifts.",
                task_type=TaskType.ROOT_CAUSE_ANALYSIS,
                tool="root_cause_analysis",
                dependencies=["T006", "T007", "T008", "T009"],
                required_inputs=[
                    "product_contribution_delta",
                    "customer_contribution_delta",
                    "region_channel_contribution_delta",
                    "revenue_anomaly_list",
                ],
                expected_outputs=["consolidated_rca_drivers", "root_cause_confidence"],
                knowledge_retrieval=KnowledgeRetrievalSpec(
                    query="Rate vs Volume Mix Decomposition and factor tree attribution",
                    category="root_cause",
                    top_k=2,
                ),
            ),
            Task(
                task_id="T011",
                name="Synthesize Evidence-Based Findings",
                description="Synthesize quantitative driver rankings with confidence intervals and hypothesis tests.",
                task_type=TaskType.STATISTICAL_ANALYSIS,
                tool="statistical_analysis",
                dependencies=["T010"],
                required_inputs=["consolidated_rca_drivers"],
                expected_outputs=["ranked_findings", "statistical_significance"],
            ),
            Task(
                task_id="T012",
                name="Generate Strategic Recommendations",
                description="Derive actionable business remediation strategies addressing verified root causes.",
                task_type=TaskType.RECOMMENDATION,
                tool="generate_recommendation",
                dependencies=["T011"],
                required_inputs=["ranked_findings"],
                expected_outputs=["strategic_action_plan", "business_impact_level"],
            ),
            Task(
                task_id="T013",
                name="Generate Final Executive Report",
                description="Compile final analytical report with visual waterfall chart and findings summary.",
                task_type=TaskType.REPORTING,
                tool="generate_report",
                dependencies=["T012"],
                required_inputs=["ranked_findings", "strategic_action_plan"],
                expected_outputs=["final_executive_report_markdown"],
            ),
        ]

    @staticmethod
    def _build_customer_tasks(goal: ParsedGoal) -> List[Task]:
        """Customer behavior, segmentation, and churn investigation tasks."""
        return [
            Task(
                task_id="T001",
                name="Load Customer Dataset",
                description=f"Load customer transaction records from {goal.dataset}.",
                task_type=TaskType.INGESTION,
                tool="load_dataset",
                dependencies=[],
                expected_outputs=["raw_dataframe"],
            ),
            Task(
                task_id="T002",
                name="Profile Customer Dataset",
                description="Profile customer identifiers, activity timestamps, and monetary attributes.",
                task_type=TaskType.PROFILING,
                tool="profile_dataset",
                dependencies=["T001"],
                expected_outputs=["profile_report"],
            ),
            Task(
                task_id="T003",
                name="Validate Data Quality",
                description="Ensure customer IDs are unique and monetary quantities are non-negative.",
                task_type=TaskType.VALIDATION,
                tool="validate_data_quality",
                dependencies=["T002"],
                expected_outputs=["quality_verdict"],
            ),
            Task(
                task_id="T004",
                name="Compute RFM Metrics",
                description="Calculate Recency, Frequency, and Monetary values per customer entity.",
                task_type=TaskType.EDA,
                tool="eda_analysis",
                dependencies=["T003"],
                expected_outputs=["rfm_features"],
            ),
            Task(
                task_id="T005",
                name="Segment Customers",
                description="Execute clustering algorithm to identify behavioral customer personas.",
                task_type=TaskType.SEGMENTATION,
                tool="segment_customers",
                dependencies=["T004"],
                expected_outputs=["customer_segments"],
                knowledge_retrieval=KnowledgeRetrievalSpec(
                    query="Unsupervised Clustering Algorithm Selection for customer segmentation",
                    category="machine_learning",
                    top_k=2,
                ),
            ),
            Task(
                task_id="T006",
                name="Analyze Churn & Retention Curves",
                description="Compute cohort retention decay and identify leading churn indicators.",
                task_type=TaskType.STATISTICAL_ANALYSIS,
                tool="statistical_analysis",
                dependencies=["T005"],
                expected_outputs=["retention_cohort_matrix", "churn_risk_scores"],
                knowledge_retrieval=KnowledgeRetrievalSpec(
                    query="Cohort Retention and Churn Rate Analysis",
                    category="business_analysis",
                    top_k=2,
                ),
            ),
            Task(
                task_id="T007",
                name="Generate Customer Retention Recommendations",
                description="Produce targeted retention campaigns for high-risk, high-value tiers.",
                task_type=TaskType.RECOMMENDATION,
                tool="generate_recommendation",
                dependencies=["T006"],
                expected_outputs=["retention_playbook"],
            ),
            Task(
                task_id="T008",
                name="Generate Customer Insights Report",
                description="Compile comprehensive customer intelligence report.",
                task_type=TaskType.REPORTING,
                tool="generate_report",
                dependencies=["T007"],
                expected_outputs=["customer_insights_report"],
            ),
        ]

    @staticmethod
    def _build_anomaly_tasks(goal: ParsedGoal) -> List[Task]:
        """Anomaly detection and investigation tasks."""
        return [
            Task(
                task_id="T001",
                name="Load Dataset",
                description="Load operational data stream.",
                task_type=TaskType.INGESTION,
                tool="load_dataset",
                dependencies=[],
                expected_outputs=["raw_dataframe"],
            ),
            Task(
                task_id="T002",
                name="Detect Anomalies",
                description="Run multi-method statistical and machine learning anomaly detection.",
                task_type=TaskType.ANOMALY_DETECTION,
                tool="detect_anomalies",
                dependencies=["T001"],
                expected_outputs=["detected_anomalies", "anomaly_scores"],
                knowledge_retrieval=KnowledgeRetrievalSpec(
                    query="Isolation Forest and Tukey IQR fences for anomaly detection",
                    category="anomaly_detection",
                    top_k=2,
                ),
            ),
            Task(
                task_id="T003",
                name="Validate Anomaly Integrity",
                description="Verify whether flagged anomalies are measurement artifacts or genuine events.",
                task_type=TaskType.VALIDATION,
                tool="validate_data_quality",
                dependencies=["T002"],
                expected_outputs=["validated_anomalies"],
            ),
            Task(
                task_id="T004",
                name="Historical Baseline Comparison",
                description="Compare anomalous window against rolling historical distributions.",
                task_type=TaskType.STATISTICAL_ANALYSIS,
                tool="statistical_analysis",
                dependencies=["T003"],
                expected_outputs=["historical_deviation_metrics"],
            ),
            Task(
                task_id="T005",
                name="Investigate Anomaly Root Causes",
                description="Rank correlated features and isolate driving factors for abnormal state.",
                task_type=TaskType.ROOT_CAUSE_ANALYSIS,
                tool="root_cause_analysis",
                dependencies=["T004"],
                expected_outputs=["ranked_causes", "root_cause_confidence"],
            ),
            Task(
                task_id="T006",
                name="Generate Anomaly Incident Report",
                description="Compile technical diagnostic report detailing causes and mitigations.",
                task_type=TaskType.REPORTING,
                tool="generate_report",
                dependencies=["T005"],
                expected_outputs=["incident_report_markdown"],
            ),
        ]

    @staticmethod
    def _build_predictive_tasks(goal: ParsedGoal) -> List[Task]:
        """Predictive modeling pipeline tasks."""
        return [
            Task(
                task_id="T001",
                name="Load Feature Dataset",
                description="Ingest training tabular features.",
                task_type=TaskType.INGESTION,
                tool="load_dataset",
                dependencies=[],
                expected_outputs=["raw_dataframe"],
            ),
            Task(
                task_id="T002",
                name="Profile Dataset and Target",
                description="Evaluate feature distributions and target variable balance/skew.",
                task_type=TaskType.PROFILING,
                tool="profile_dataset",
                dependencies=["T001"],
                expected_outputs=["profile_report", "target_type"],
            ),
            Task(
                task_id="T003",
                name="Engineer Features and Normalization",
                description="Encode categorical variables, handle missing data, and apply scaling.",
                task_type=TaskType.CLEANING,
                tool="clean_dataset",
                dependencies=["T002"],
                expected_outputs=["processed_feature_matrix"],
            ),
            Task(
                task_id="T004",
                name="Select Cross-Validation Strategy",
                description="Determine optimal split protocol (Stratified K-Fold or TimeSeriesSplit).",
                task_type=TaskType.VALIDATION,
                tool="statistical_analysis",
                dependencies=["T003"],
                expected_outputs=["cv_folds_config"],
                knowledge_retrieval=KnowledgeRetrievalSpec(
                    query="Cross-Validation and Data Splitting Protocols",
                    category="machine_learning",
                    top_k=2,
                ),
            ),
            Task(
                task_id="T005",
                name="Train Candidate Models",
                description="Fit baseline and gradient boosted models using cross-validation.",
                task_type=TaskType.MACHINE_LEARNING,
                tool="train_model",
                dependencies=["T004"],
                expected_outputs=["trained_model_artifacts", "cv_metrics"],
                knowledge_retrieval=KnowledgeRetrievalSpec(
                    query="Supervised Classification Algorithm Selection",
                    category="machine_learning",
                    top_k=2,
                ),
            ),
            Task(
                task_id="T006",
                name="Evaluate Model & Feature Importance",
                description="Compute permutation feature importance, calibration, and error residuals.",
                task_type=TaskType.MACHINE_LEARNING,
                tool="eda_analysis",
                dependencies=["T005"],
                expected_outputs=["feature_importance_ranking", "test_evaluation_metrics"],
            ),
            Task(
                task_id="T007",
                name="Generate Predictive Model Report",
                description="Document performance benchmarks, key feature drivers, and deployment guidance.",
                task_type=TaskType.REPORTING,
                tool="generate_report",
                dependencies=["T006"],
                expected_outputs=["model_performance_report"],
            ),
        ]

    @staticmethod
    def _build_general_analysis_tasks(goal: ParsedGoal) -> List[Task]:
        """General dataset profiling and exploratory analysis tasks."""
        return [
            Task(
                task_id="T001",
                name="Load Dataset",
                description="Ingest dataset into analytical session.",
                task_type=TaskType.INGESTION,
                tool="load_dataset",
                dependencies=[],
                expected_outputs=["raw_dataframe"],
            ),
            Task(
                task_id="T002",
                name="Profile Dataset",
                description="Compute distributions, missingness, and data types across all columns.",
                task_type=TaskType.PROFILING,
                tool="profile_dataset",
                dependencies=["T001"],
                expected_outputs=["profile_report", "missing_rate", "quality_score"],
            ),
            Task(
                task_id="T003",
                name="Validate Data Quality",
                description="Check schema constraints, valid ranges, and duplicate records.",
                task_type=TaskType.VALIDATION,
                tool="validate_data_quality",
                dependencies=["T002"],
                expected_outputs=["quality_verdict", "needs_cleaning"],
            ),
            Task(
                task_id="T004",
                name="Clean Dataset Conditionally",
                description="Impute missing values and standardize formats if necessary.",
                task_type=TaskType.CLEANING,
                tool="clean_dataset",
                dependencies=["T003"],
                expected_outputs=["cleaned_dataframe"],
            ),
            Task(
                task_id="T005",
                name="Perform Exploratory Data Analysis",
                description="Analyze univariate and bivariate correlations and distributions.",
                task_type=TaskType.EDA,
                tool="eda_analysis",
                dependencies=["T003", "T004"],
                expected_outputs=["eda_summary_metrics"],
            ),
            Task(
                task_id="T006",
                name="Perform Statistical Testing",
                description="Execute hypothesis tests to evaluate significant differences between groups.",
                task_type=TaskType.STATISTICAL_ANALYSIS,
                tool="statistical_analysis",
                dependencies=["T005"],
                expected_outputs=["statistical_test_results"],
                knowledge_retrieval=KnowledgeRetrievalSpec(
                    query="Parametric and Non-Parametric Two-Sample Hypothesis Testing",
                    category="statistics",
                    top_k=2,
                ),
            ),
            Task(
                task_id="T007",
                name="Detect Data Anomalies",
                description="Screen numerical columns for extreme multi-sigma outliers.",
                task_type=TaskType.ANOMALY_DETECTION,
                tool="detect_anomalies",
                dependencies=["T005"],
                expected_outputs=["anomaly_flags"],
            ),
            Task(
                task_id="T008",
                name="Perform Pattern & Segment Analysis",
                description="Extract clustering groupings and feature interactions.",
                task_type=TaskType.SEGMENTATION,
                tool="eda_analysis",
                dependencies=["T006", "T007"],
                expected_outputs=["segment_patterns"],
            ),
            Task(
                task_id="T009",
                name="Perform Root Cause Analysis",
                description="Identify principal factors driving observed disparities.",
                task_type=TaskType.ROOT_CAUSE_ANALYSIS,
                tool="root_cause_analysis",
                dependencies=["T008"],
                expected_outputs=["primary_drivers"],
            ),
            Task(
                task_id="T010",
                name="Synthesize Analytical Findings",
                description="Combine statistical and exploratory discoveries into clear takeaways.",
                task_type=TaskType.REPORTING,
                tool="generate_report",
                dependencies=["T009"],
                expected_outputs=["findings_summary"],
            ),
            Task(
                task_id="T011",
                name="Generate Recommendations",
                description="Formulate actionable business conclusions.",
                task_type=TaskType.RECOMMENDATION,
                tool="generate_recommendation",
                dependencies=["T010"],
                expected_outputs=["actionable_recommendations"],
            ),
            Task(
                task_id="T012",
                name="Compile Final Analytical Report",
                description="Produce publication-ready executive report.",
                task_type=TaskType.REPORTING,
                tool="generate_report",
                dependencies=["T011"],
                expected_outputs=["final_report_markdown"],
            ),
        ]
