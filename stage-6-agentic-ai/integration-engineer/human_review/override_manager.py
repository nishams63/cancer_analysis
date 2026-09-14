"""Handles explicit human decision overrides."""
from typing import Dict, Any, Optional
from orchestration.session_manager import get_session_manager
from orchestration.lifecycle import RunLifecycleState
from events.event_bus import get_global_event_bus
from events.publisher import EventPublisher
from events.event_types import IntegrationEventType
from agent_engineer.agent.executor import AgentExecutor
from agent_engineer.schemas.agent import AgentState, RunStatus
from agent_engineer.schemas.result import AgentResult
from workflow_engineer.schemas.workflow import Workflow


class OverrideManager:
    """Submits structured overrides to the AgentExecutor and records audit events."""

    def __init__(self, executor: Optional[AgentExecutor] = None, session_manager=None, event_bus=None):
        self.executor = executor or AgentExecutor()
        self.session_manager = session_manager or get_session_manager()
        self.bus = event_bus or get_global_event_bus()

    def submit_override(
        self,
        run_id: str,
        workflow: Workflow,
        decision: str,
        reason: str,
        override_parameters: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        session = self.session_manager.get_session(run_id)
        if not session:
            raise ValueError(f"Run session '{run_id}' not found.")

        publisher = EventPublisher(run_id=run_id, bus=self.bus)
        publisher.emit(
            event_type=IntegrationEventType.HUMAN_OVERRIDE,
            message=f"Human override recorded: {decision} - {reason}",
            task_id=session.current_task,
            payload={"decision": decision, "reason": reason, "params": override_parameters or {}},
        )

        session.human_review.status = "OVERRIDDEN"
        session.update_status(RunLifecycleState.RUNNING)
        self.session_manager.save_session(session)

        # Ensure repo has the run state before resuming/overriding
        if not self.executor.repo.get_run(run_id):
            st = AgentState(run_id=run_id, workflow_id=workflow.workflow_id, status=RunStatus.WAITING_FOR_HUMAN)
            self.executor.repo.save_run(st)

        # Authoritative agent override resumption
        result = self.executor.resume_after_approval(
            run_id=run_id,
            workflow=workflow,
            decision=f"override:{decision}",
            reason=reason,
        )
        return result
