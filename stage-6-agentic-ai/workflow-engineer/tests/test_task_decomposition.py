"""Unit tests for analytical TaskDecomposer."""
from planner.goal_parser import GoalParser
from planner.task_decomposer import TaskDecomposer
from schemas.task import TaskType


def test_decompose_revenue_decline():
    parser = GoalParser()
    goal = parser.parse("Why did revenue decrease in the sales dataset?")
    tasks = TaskDecomposer.decompose(goal)

    assert len(tasks) == 13
    task_ids = [t.task_id for t in tasks]
    assert task_ids == [f"T{i:03d}" for i in range(1, 14)]

    # Check multi-branch dependencies on T010
    t010 = next(t for t in tasks if t.task_id == "T010")
    assert set(t010.dependencies) == {"T006", "T007", "T008", "T009"}

    # Check Knowledge Engineer integration presence
    t003 = next(t for t in tasks if t.task_id == "T003")
    assert t003.knowledge_retrieval is not None
    assert t003.knowledge_retrieval.category == "data_quality"


def test_decompose_customer_analysis():
    parser = GoalParser()
    goal = parser.parse("Analyze customer behavior and churn cohort retention.")
    tasks = TaskDecomposer.decompose(goal)

    assert len(tasks) == 8
    assert tasks[0].task_type == TaskType.INGESTION
    assert any(t.task_type == TaskType.SEGMENTATION for t in tasks)


def test_decompose_anomaly_investigation():
    parser = GoalParser()
    goal = parser.parse("Investigate statistical anomalies in transaction stream.")
    tasks = TaskDecomposer.decompose(goal)

    assert len(tasks) == 6
    assert any(t.task_type == TaskType.ANOMALY_DETECTION for t in tasks)
