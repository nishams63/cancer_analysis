"""Unit tests for DAG dependency resolution and topological sort."""
import pytest
from schemas.task import Task, TaskType
from planner.dependency_resolver import DependencyResolver


def test_topological_sort_linear(sample_linear_tasks):
    order = DependencyResolver.topological_sort(sample_linear_tasks)
    assert order == ["T001", "T002", "T003"]


def test_cycle_detection(sample_cyclic_tasks):
    has_cycle, cycle_err = DependencyResolver.detect_cycles(sample_cyclic_tasks)
    assert has_cycle is True
    assert "Circular dependency" in cycle_err

    with pytest.raises(ValueError, match="cycles"):
        DependencyResolver.topological_sort(sample_cyclic_tasks)


def test_missing_dependency_detected():
    tasks = [
        Task(
            task_id="T001",
            name="Load",
            description="Valid task description",
            task_type=TaskType.INGESTION,
            tool="load",
            dependencies=["T999_NON_EXISTENT"],
        )
    ]
    has_cycle, err = DependencyResolver.detect_cycles(tasks)
    assert has_cycle is True
    assert "non-existent dependency" in err


def test_unreachable_tasks():
    tasks = [
        Task(task_id="T001", name="Entry", description="Entry task description", task_type=TaskType.INGESTION, tool="load"),
        Task(task_id="T002", name="Child", description="Child task description", task_type=TaskType.PROFILING, tool="prof", dependencies=["T001"]),
        Task(task_id="T003", name="Orphan", description="Orphan task description", task_type=TaskType.EDA, tool="eda", dependencies=[]),
    ]
    unreachable = DependencyResolver.find_unreachable_tasks(tasks, entry_task_id="T001")
    assert unreachable == ["T003"]
