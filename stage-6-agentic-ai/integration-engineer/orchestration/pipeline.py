"""Authoritative end-to-end AADA integration pipeline."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from .lifecycle import RunLifecycleState
from .session_manager import AADARunSession, SessionManager, get_session_manager
from events.event_bus import EventBus, get_global_event_bus
from events.publisher import EventPublisher
from events.event_types import IntegrationEventType
from streaming.trace_stream import TraceStreamBridge

from workflow_engineer.planner.workflow_planner import WorkflowPlanner
from workflow_engineer.schemas.workflow import Workflow
from agent_engineer.agent.executor import AgentExecutor
from agent_engineer.trace.repository import TraceRepository
from agent_engineer.schemas.agent import RunStatus
from agent_engineer.schemas.result import AgentResult

from evaluation_engineer.experiments.runner import ExperimentRunner
from evaluation_engineer.schemas.scenario import EvaluationScenario, ScenarioCategory, ScenarioDifficulty
from evaluation_engineer.schemas.evaluation import ScenarioEvaluationResult


class AADAIntegrationPipeline:
    """Coordinates Workflow Engineer, Agent Engineer, and Evaluation Engineer."""

    def __init__(
        self,
        planner: Optional[WorkflowPlanner] = None,
        executor: Optional[AgentExecutor] = None,
        eval_runner: Optional[ExperimentRunner] = None,
        session_manager: Optional[SessionManager] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.planner = planner or WorkflowPlanner()
        self.executor = executor or AgentExecutor()
        self.eval_runner = eval_runner or ExperimentRunner()
        self.session_manager = session_manager or get_session_manager()
        self.bus = event_bus or get_global_event_bus()

    def run_pipeline(
        self,
        goal: str,
        custom_run_id: Optional[str] = None,
        auto_evaluate: bool = True,
    ) -> AADARunSession:
        """Execute complete pipeline: Goal -> Workflow -> Agent -> Evaluation -> Result."""
        run_id = custom_run_id or f"RUN-{uuid.uuid4().hex[:8].upper()}"
        session = self.session_manager.create_session(run_id=run_id, goal=goal)
        publisher = EventPublisher(run_id=run_id, bus=self.bus)

        # 1. RUN_STARTED
        session.update_status(RunLifecycleState.PLANNING)
        self.session_manager.save_session(session)
        publisher.emit(
            event_type=IntegrationEventType.RUN_STARTED,
            message=f"AADA session started for goal: '{goal}'",
            payload={"goal": goal},
        )

        # 2. Planning via Workflow Engineer
        try:
            workflow: Workflow = self.planner.plan(goal_text=goal)
            session.workflow_id = workflow.workflow_id
            session.tasks_total = len(workflow.tasks)
            session.workflow_metadata = {
                "name": workflow.name,
                "domain": workflow.metadata.get("domain", "general"),
                "intent_type": workflow.metadata.get("intent_type", "analysis"),
                "tasks": [
                    {
                        "task_id": t.task_id,
                        "name": t.name,
                        "tool": t.tool,
                        "dependencies": t.dependencies,
                    }
                    for t in workflow.tasks
                ],
            }
            session.update_status(RunLifecycleState.READY)
            self.session_manager.save_session(session)

            publisher.emit(
                event_type=IntegrationEventType.WORKFLOW_CREATED,
                message=f"Workflow '{workflow.workflow_id}' generated with {len(workflow.tasks)} tasks.",
                payload={"workflow_id": workflow.workflow_id, "tasks_count": len(workflow.tasks)},
            )
        except Exception as plan_err:
            session.update_status(RunLifecycleState.FAILED)
            session.error = {"code": "WORKFLOW_GENERATION_FAILED", "message": str(plan_err)}
            self.session_manager.save_session(session)
            publisher.emit(
                event_type=IntegrationEventType.RUN_FAILED,
                message=f"Workflow generation failed: {plan_err}",
                severity="ERROR",
            )
            return session

        # 3. Execution via Agent Engineer
        session.update_status(RunLifecycleState.RUNNING)
        self.session_manager.save_session(session)

        try:
            bridge = TraceStreamBridge(run_id=run_id, bus=self.bus)
            agent_result: AgentResult = self.executor.run(
                workflow=workflow,
                custom_run_id=run_id,
            )

            # Bridge recorded trace events
            trace_events = self.executor.repo.get_trace(run_id)
            for evt in trace_events:
                bridge.bridge_event(evt)

            # Update session with agent results
            session.agent_status = agent_result.status
            session.tasks_completed = agent_result.completed_tasks
            session.confidence = agent_result.confidence
            session.findings = agent_result.findings
            session.evidence = agent_result.evidence
            session.recommendations = agent_result.recommendations

            # Check if paused for Human-in-the-Loop
            if agent_result.status == RunStatus.WAITING_FOR_HUMAN.value or agent_result.escalated:
                session.update_status(RunLifecycleState.WAITING_FOR_HUMAN)
                pending_appr = self.executor.repo.get_run(run_id) or {}
                state_dict = pending_appr.get("state", {})
                approvals = state_dict.get("pending_approvals", [])

                reason = approvals[0].get("message") if approvals else "Low confidence or ambiguous findings require analyst review."
                task_id = approvals[0].get("task_id") if approvals else state_dict.get("current_task_id")

                session.human_review.required = True
                session.human_review.status = "PENDING"
                session.human_review.reason = reason
                session.human_review.task_id = task_id
                session.human_review.confidence = agent_result.confidence
                session.human_review.evidence = agent_result.evidence

                self.session_manager.save_session(session)
                publisher.emit(
                    event_type=IntegrationEventType.HUMAN_REVIEW_REQUIRED,
                    message=f"Human review required: {reason}",
                    task_id=task_id,
                    payload={"confidence": agent_result.confidence, "reason": reason},
                    severity="WARNING",
                )
                return session

            if agent_result.status == RunStatus.FAILED.value:
                session.update_status(RunLifecycleState.FAILED)
                session.error = {"code": "AGENT_EXECUTION_FAILED", "message": "Agent execution failed."}
                self.session_manager.save_session(session)
                publisher.emit(
                    event_type=IntegrationEventType.RUN_FAILED,
                    message="Agent execution failed.",
                    severity="ERROR",
                )
                return session

        except Exception as exec_err:
            session.update_status(RunLifecycleState.FAILED)
            session.error = {"code": "AGENT_EXECUTION_ERROR", "message": str(exec_err)}
            self.session_manager.save_session(session)
            publisher.emit(
                event_type=IntegrationEventType.RUN_FAILED,
                message=f"Agent execution encountered error: {exec_err}",
                severity="ERROR",
            )
            return session

        # 4. Evaluation via Evaluation Engineer
        if auto_evaluate:
            session.update_status(RunLifecycleState.EVALUATING)
            self.session_manager.save_session(session)
            publisher.emit(
                event_type=IntegrationEventType.EVALUATION_STARTED,
                message="Evaluation Engineer assessing agent process, outcome, and safety.",
            )

            try:
                scenario = self._resolve_or_create_scenario(workflow, agent_result)
                trace_events = self.executor.repo.get_trace(run_id)
                eval_res: ScenarioEvaluationResult = self.eval_runner.evaluate_run(
                    scenario=scenario,
                    agent_result=agent_result,
                    trace_events=trace_events,
                )

                session.evaluation_status = "COMPLETED"
                session.evaluation_passed = eval_res.passed
                session.process_score = eval_res.process_score
                session.outcome_score = eval_res.outcome_score
                session.overall_score = eval_res.overall_score
                session.evaluation_details = {
                    "verdict": eval_res.verdict_rationale,
                    "critical_violations": eval_res.safety_metrics.critical_violations,
                    "high_violations": eval_res.safety_metrics.high_violations,
                    "failures_count": len(eval_res.failures),
                }

                publisher.emit(
                    event_type=IntegrationEventType.EVALUATION_COMPLETED,
                    message=f"Evaluation complete: Verdict={eval_res.verdict_rationale} (Score: {int(eval_res.overall_score * 100)}%)",
                    payload={
                        "passed": eval_res.passed,
                        "overall_score": eval_res.overall_score,
                        "verdict": eval_res.verdict_rationale,
                    },
                )
            except Exception as eval_err:
                session.evaluation_status = "FAILED"
                session.evaluation_details = {"error": str(eval_err)}

        # 5. Complete Session
        session.update_status(RunLifecycleState.COMPLETED)
        self.session_manager.save_session(session)
        publisher.emit(
            event_type=IntegrationEventType.RUN_COMPLETED,
            message=f"AADA Run {run_id} completed successfully.",
            payload={"confidence": session.confidence},
        )
        return session

    def _resolve_or_create_scenario(self, workflow: Workflow, agent_result: AgentResult) -> EvaluationScenario:
        """Find matching catalog scenario or create synthetic evaluation contract."""
        intent = workflow.metadata.get("intent_type", "revenue_decline")
        # Match against known catalog keys
        catalog_map = {
            "revenue_decline": "SC-REV-001",
            "revenue_growth": "SC-REV-002",
            "customer_churn": "SC-CUST-001",
            "segmentation": "SC-CUST-002",
            "anomaly_detection": "SC-ANOM-001",
        }
        matched_id = catalog_map.get(intent)
        if matched_id and matched_id in self.eval_runner._scenarios:
            return self.eval_runner._scenarios[matched_id]

        # Synthetic contract for unmapped goals
        return EvaluationScenario(
            scenario_id=f"SC-SYNTH-{workflow.workflow_id}",
            name=workflow.name,
            description=workflow.goal,
            category=ScenarioCategory.SALES,
            dataset=workflow.metadata.get("dataset") or "data/default.csv",
            user_goal=workflow.goal,
            expected_workflow=workflow.workflow_id,
            expected_tasks=[t.task_id for t in workflow.tasks],
            expected_tools=[t.tool for t in workflow.tasks if t.tool],
            difficulty=ScenarioDifficulty.MEDIUM,
        )
