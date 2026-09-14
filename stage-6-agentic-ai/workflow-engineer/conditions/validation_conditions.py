"""Data validation condition definitions."""
from typing import List, Optional
from schemas.condition import BranchCondition, ComparisonOperator, BranchAction


def build_column_presence_check(
    required_columns_count_metric: str = "missing_required_columns_count",
) -> BranchCondition:
    return BranchCondition(
        condition_id="COND-REQUIRED-COLS",
        metric_name=required_columns_count_metric,
        operator=ComparisonOperator.GT,
        threshold=0,
        action=BranchAction.HUMAN_REVIEW,
        description="Escalate if any required columns are missing from the schema",
    )


def build_row_count_check(
    min_rows: int = 10,
) -> BranchCondition:
    return BranchCondition(
        condition_id="COND-MIN-ROWS",
        metric_name="row_count",
        operator=ComparisonOperator.LT,
        threshold=min_rows,
        action=BranchAction.TERMINATE,
        description=f"Terminate if dataset contains fewer than {min_rows} rows",
    )
