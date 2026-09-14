"""Deterministic Analytical Goal Parser without external LLM dependencies."""
from __future__ import annotations
import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ParsedGoal(BaseModel):
    """Structured representation of user analytical intent."""
    raw_goal: str
    objective: str
    intent_type: str  # revenue_decline, customer_analysis, anomaly_investigation, predictive_analysis, dataset_analysis
    domain: str       # sales, customer, operations, predictive, general
    dataset: Optional[str] = None
    time_scope: Optional[str] = None
    analysis_requirements: List[str] = Field(default_factory=list)
    key_metrics: List[str] = Field(default_factory=list)


class GoalParser:
    """Extracts analytical requirements from natural language input deterministically."""

    # Regex patterns for time scope detection
    TIME_PATTERNS = [
        (r"\b(last|past)\s+(quarter|month|year|week|decade)\b", lambda m: m.group(0).lower()),
        (r"\b(q[1-4]|quarter\s+[1-4])(\s+\d{4})?\b", lambda m: m.group(0).lower()),
        (r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b", lambda m: m.group(0).lower()),
        (r"\b(fy\s*\d{2,4}|20\d{2})\b", lambda m: m.group(0).lower()),
        (r"\b(yoy|mom|qoq)\b", lambda m: f"period-over-period ({m.group(0).lower()})"),
    ]

    # Patterns for dataset detection
    DATASET_PATTERNS = [
        r"\b([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)?\s+(?:dataset|data|table|records|file|csv))\b",
    ]

    def parse(self, goal_text: str) -> ParsedGoal:
        """Parse natural language query into deterministic analytical requirements."""
        text_clean = goal_text.strip()
        lower = text_clean.lower()

        # 1. Extract Time Scope
        time_scope = None
        for pattern, extractor in self.TIME_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                time_scope = extractor(match)
                break

        # 2. Extract Dataset reference
        dataset = None
        for pattern in self.DATASET_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                raw_ds = match.group(1).strip()
                raw_ds = re.sub(r"^(?:this|the|that|a|an)\s+", "", raw_ds, flags=re.IGNORECASE)
                dataset = raw_ds
                break

        # 3. Classify Intent and Domain
        if any(w in lower for w in ["revenue", "sales", "arr", "mrr", "income", "margin"]) and any(w in lower for w in ["decrease", "decline", "drop", "fell", "down", "why", "loss"]):
            intent_type = "revenue_decline"
            domain = "sales"
            objective = "Determine the causes of revenue decline and decompose performance drivers."
            requirements = [
                "revenue trend analysis",
                "product analysis",
                "customer segment analysis",
                "geographic/channel performance analysis",
                "pricing and volume mix decomposition",
                "revenue anomaly detection",
                "root cause analysis and evidence ranking",
            ]
            metrics = ["revenue", "sales_volume", "average_order_value", "unit_price"]

        elif any(w in lower for w in ["predict", "forecast", "classification", "regression", "model"]):
            intent_type = "predictive_analysis"
            domain = "predictive"
            objective = "Build, validate, and evaluate predictive model pipeline on target variable."
            requirements = [
                "target distribution validation",
                "feature engineering and scaling",
                "cross-validation protocol selection",
                "model training and baseline comparison",
                "feature importance and error analysis",
            ]
            metrics = ["rmse", "mae", "roc_auc", "f1_score", "r_squared"]

        elif any(w in lower for w in ["anomal", "outlier", "spike", "fraud", "unusual", "deviat"]):
            intent_type = "anomaly_investigation"
            domain = "operations"
            objective = "Detect, validate, and isolate root causes for statistical anomalies."
            requirements = [
                "statistical anomaly detection",
                "multivariate outlier validation",
                "historical baseline comparison",
                "cohort anomaly segmentation",
                "root cause ranking and explanation",
            ]
            metrics = ["anomaly_score", "z_score", "isolation_path_length", "residual_error"]

        elif any(w in lower for w in ["customer", "churn", "retention", "cohort", "rfm", "user", "subscriber"]):
            intent_type = "customer_analysis"
            domain = "customer"
            objective = "Analyze customer behavior, segmentation, and retention/churn patterns."
            requirements = [
                "customer profiling",
                "rfm segmentation",
                "behavioral trend analysis",
                "churn risk and anomaly detection",
                "cross-segment comparison",
                "root cause analysis of churn",
            ]
            metrics = ["churn_rate", "retention_rate", "ltv", "recency", "frequency", "monetary"]

        else:
            intent_type = "dataset_analysis"
            domain = "general"
            objective = "Perform end-to-end dataset profiling, exploratory analysis, and hypothesis discovery."
            requirements = [
                "data quality audit",
                "univariate and bivariate profiling",
                "correlation and association analysis",
                "statistical hypothesis testing",
                "key findings and executive report generation",
            ]
            metrics = ["row_count", "column_count", "quality_score", "correlation_matrix"]

        return ParsedGoal(
            raw_goal=text_clean,
            objective=objective,
            intent_type=intent_type,
            domain=domain,
            dataset=dataset or "input_dataset",
            time_scope=time_scope,
            analysis_requirements=requirements,
            key_metrics=metrics,
        )
