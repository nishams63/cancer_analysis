"""End-to-End Deterministic Workflow Planner."""
from __future__ import annotations
import uuid
from typing import Optional, Dict, Any, List
from schemas.workflow import Workflow
from schemas.task import Task
from planner.goal_parser import GoalParser, ParsedGoal
from planner.task_decomposer import TaskDecomposer
from planner.dependency_resolver import DependencyResolver
from conditions.escalation_conditions import (
    build_poor_quality_escalation,
    build_low_confidence_rca_escalation,
    build_similar_cause_scores_escalation,
    build_missing_columns_escalation,
    build_max_retries_escalation,
    build_high_impact_action_escalation,
)


class WorkflowPlanner:
    """Synthesizes complete, validated, executable Workflow objects from user goals."""

    def __init__(self, goal_parser: Optional[GoalParser] = None):
        self.parser = goal_parser or GoalParser()
        self.decomposer = TaskDecomposer()
        self.resolver = DependencyResolver()

    def plan(self, goal_text: str, custom_id: Optional[str] = None) -> Workflow:
        """Convert high-level natural language goal into deterministic executable Workflow."""
        parsed_goal = self.parser.parse(goal_text)
        tasks = self.decomposer.decompose(parsed_goal)

        # Validate DAG dependencies and detect cycles
        has_cycle, cycle_err = self.resolver.detect_cycles(tasks)
        if has_cycle:
            raise ValueError(f"Workflow decomposition produced invalid graph: {cycle_err}")

        # Compute topological order
        exec_order = self.resolver.topological_sort(tasks)
        entry_task = exec_order[0]

        # Identify terminal tasks (tasks with no outgoing dependencies)
        adj = self.resolver.build_adjacency_list(tasks)
        terminal_tasks = [t.task_id for t in tasks if not adj.get(t.task_id)]
        if not terminal_tasks:
            terminal_tasks = [tasks[-1].task_id]

        workflow_id = custom_id or f"WF-{parsed_goal.intent_type.upper()}-001"
        name = f"{parsed_goal.domain.title()} - {parsed_goal.intent_type.replace('_', ' ').title()}"

        escalation_rules = [
            build_poor_quality_escalation(threshold=0.50),
            build_low_confidence_rca_escalation(threshold=0.60),
            build_similar_cause_scores_escalation(margin=0.05),
            build_missing_columns_escalation(),
            build_max_retries_escalation(max_retries=3),
            build_high_impact_action_escalation(),
        ]

        return Workflow(
            workflow_id=workflow_id,
            name=name,
            version="1.0",
            goal=parsed_goal.objective,
            entry_task=entry_task,
            terminal_tasks=terminal_tasks,
            tasks=tasks,
            escalation_rules=escalation_rules,
            metadata={
                "raw_goal": parsed_goal.raw_goal,
                "domain": parsed_goal.domain,
                "intent_type": parsed_goal.intent_type,
                "dataset": parsed_goal.dataset,
                "time_scope": parsed_goal.time_scope,
                "analysis_requirements": parsed_goal.analysis_requirements,
                "topological_order": exec_order,
            },
        )
