"""Handles human approvals, interacting directly with AgentExecutor."""
from typing import Dict, Any, Optional
from orchestration.session_manager import AADARunSession, get_session_manager
from orchestration.lifecycle import RunLifecycleState
from events.event_bus import get_global_event_bus
from events.publisher import EventPublisher
from events.event_types import IntegrationEventType
from agent_engineer.agent.executor import AgentExecutor
from agent_engineer.schemas.agent import AgentState, RunStatus
from agent_engineer.schemas.result import AgentResult
from workflow_engineer.schemas.workflow import Workflow


class ApprovalManager:
    """Resumes paused agents upon human approval or handles rejection."""

    def __init__(self, executor: Optional[AgentExecutor] = None, session_manager=None, event_bus=None):
        self.executor = executor or AgentExecutor()
        self.session_manager = session_manager or get_session_manager()
        self.bus = event_bus or get_global_event_bus()

    def approve(self, run_id: str, workflow: Workflow, reason: str = "Approved by analyst") -> AgentResult:
        session = self.session_manager.get_session(run_id)
        if not session:
            raise ValueError(f"Run session '{run_id}' not found.")

        publisher = EventPublisher(run_id=run_id, bus=self.bus)
        publisher.emit(
            event_type=IntegrationEventType.HUMAN_APPROVED,
            message=f"Human approved execution for task {session.current_task}: {reason}",
            task_id=session.current_task,
            payload={"reason": reason},
        )

        session.human_review.status = "APPROVED"
        session.update_status(RunLifecycleState.RUNNING)
        self.session_manager.save_session(session)

        # Ensure repo has the run state before resuming
        if not self.executor.repo.get_run(run_id):
            st = AgentState(run_id=run_id, workflow_id=workflow.workflow_id, status=RunStatus.WAITING_FOR_HUMAN)
            self.executor.repo.save_run(st)

        # Resume agent execution using authoritative AgentExecutor interface
        result = self.executor.resume_after_approval(
            run_id=run_id,
            workflow=workflow,
            decision="approve",
            reason=reason,
        )
        return result

    def reject(self, run_id: str, workflow: Workflow, reason: str = "Rejected by analyst") -> AgentResult:
        session = self.session_manager.get_session(run_id)
        if not session:
            raise ValueError(f"Run session '{run_id}' not found.")

        publisher = EventPublisher(run_id=run_id, bus=self.bus)
        publisher.emit(
            event_type=IntegrationEventType.HUMAN_REJECTED,
            message=f"Human rejected execution: {reason}",
            task_id=session.current_task,
            payload={"reason": reason},
            severity="WARNING",
        )

        session.human_review.status = "REJECTED"
        session.update_status(RunLifecycleState.CANCELLED)
        self.session_manager.save_session(session)

        # Ensure repo has the run state before resuming/terminating
        if not self.executor.repo.get_run(run_id):
            st = AgentState(run_id=run_id, workflow_id=workflow.workflow_id, status=RunStatus.WAITING_FOR_HUMAN)
            self.executor.repo.save_run(st)

        # Call AgentExecutor with terminate/reject decision
        result = self.executor.resume_after_approval(
            run_id=run_id,
            workflow=workflow,
            decision="reject",
            reason=reason,
        )
        return result
