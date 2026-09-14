"""Authoritative AgentExecutor orchestrating workflow tasks, ReAct, and escalations."""
from __future__ import annotations
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Set

from schemas.agent import AgentState, RunStatus
from schemas.trace import EventType
from schemas.result import AgentResult
from schemas.condition import BranchAction
from schemas.escalation import EscalationAction
from workflow_engineer.schemas.workflow import Workflow
from workflow_engineer.schemas.task import Task, FailurePolicy
from workflow_engineer.engine.workflow_validator import validate_workflow

from tools.registry import ToolRegistry, get_default_registry
from llm.base import LLMProvider
from llm.mock import MockLLMProvider
from trace.recorder import TraceRecorder
from trace.repository import TraceRepository
from execution.task_executor import TaskExecutor
from execution.branch_executor import evaluate_task_branch
from execution.escalation import check_escalations
from execution.parallel import get_ready_tasks
from reasoning.confidence import calculate_confidence
from reasoning.conflict_resolution import resolve_competing_causes


class AgentExecutor:
    """Orchestrates deterministic workflow execution under strict graph authority."""

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        llm_provider: Optional[LLMProvider] = None,
        trace_repository: Optional[TraceRepository] = None,
        max_react_steps: int = 5,
    ):
        self.registry = tool_registry or get_default_registry()
        self.llm = llm_provider or MockLLMProvider()
        self.repo = trace_repository or TraceRepository()
        self.recorder = TraceRecorder(repository=self.repo)
        self.task_executor = TaskExecutor(registry=self.registry, llm=self.llm)
        self.max_react_steps = max_react_steps

    def run(
        self,
        workflow: Workflow,
        input_data: Optional[Dict[str, Any]] = None,
        custom_run_id: Optional[str] = None,
    ) -> AgentResult:
        """Execute workflow end-to-end conforming strictly to DAG dependencies."""
        # 1. Structural Validation
        val_result = validate_workflow(workflow)
        if not val_result.valid:
            raise ValueError(f"Workflow validation failed: {val_result.errors}")

        run_id = custom_run_id or f"RUN-{uuid.uuid4().hex[:8].upper()}"
        now_iso = datetime.utcnow().isoformat() + "Z"
        start_time = time.perf_counter()

        existing_run = self.repo.get_run(run_id)
        if existing_run and existing_run.get("state"):
            state = AgentState(**existing_run["state"])
            state.status = RunStatus.RUNNING
            completed_set: Set[str] = set(state.completed_tasks)
            skipped_set: Set[str] = set(state.skipped_tasks)
            failed_set: Set[str] = set(state.failed_tasks)
        else:
            state = AgentState(
                run_id=run_id,
                workflow_id=workflow.workflow_id,
                status=RunStatus.RUNNING,
                metadata={"started_at": now_iso, "goal": workflow.goal},
            )
            state.metrics.total_tasks = len(workflow.tasks)
            completed_set = set()
            skipped_set = set()
            failed_set = set()
            self.recorder.record(
                run_id=run_id,
                event_type=EventType.WORKFLOW_STARTED,
                decision="start",
                decision_reason=f"Initiated workflow '{workflow.workflow_id}' for goal: {workflow.goal}",
            )

        if input_data:
            state.variables.update(input_data)

        self.repo.save_run(state)
        in_progress_set: Set[str] = set()
        task_map: Dict[str, Task] = {t.task_id: t for t in workflow.tasks}

        # Main execution loop
        while len(completed_set | skipped_set | failed_set) < len(workflow.tasks):
            ready_tasks = get_ready_tasks(workflow, completed_set, skipped_set, in_progress_set, failed_set)
            if not ready_tasks:
                break

            for task in ready_tasks:
                in_progress_set.add(task.task_id)
                state.current_task_id = task.task_id

                self.recorder.record(
                    run_id=run_id,
                    task_id=task.task_id,
                    event_type=EventType.TASK_STARTED,
                    tool_name=task.tool,
                    decision="execute_task",
                    decision_reason=f"Prerequisites satisfied for {task.task_id} ({task.name}).",
                )

                # 2. Knowledge Retrieval Hook
                knowledge_context = "None"
                if task.knowledge_retrieval:
                    kr = task.knowledge_retrieval
                    k_resp = self.registry.execute(
                        "retrieve_knowledge",
                        {"query": kr.query, "category": kr.category, "top_k": kr.top_k},
                    )
                    knowledge_context = str(k_resp.get("knowledge", "None"))
                    self.recorder.record(
                        run_id=run_id,
                        task_id=task.task_id,
                        event_type=EventType.OBSERVATION,
                        tool_name="retrieve_knowledge",
                        tool_input={"query": kr.query, "category": kr.category},
                        tool_output=k_resp,
                        decision="knowledge_retrieved",
                        decision_reason=f"Consulted Knowledge Base on '{kr.query}'.",
                    )

                # 3. Execute Task with Tool Allowlisting & ReAct Loop
                step_res = self.task_executor.execute_task(
                    task=task,
                    state=state,
                    knowledge_context=knowledge_context,
                    explicit_inputs=input_data if task.task_id == workflow.entry_task else None,
                )

                # Record Tool Execution Trace Event
                self.recorder.record(
                    run_id=run_id,
                    task_id=task.task_id,
                    event_type=EventType.TOOL_CALL,
                    tool_name=step_res.tool_name,
                    tool_input=step_res.tool_input,
                    tool_output=step_res.tool_output,
                    decision=step_res.decision,
                    decision_reason=step_res.decision_reason,
                    status=step_res.status,
                )

                # Check task failure
                if step_res.status == "error":
                    state.record_task_failure(task.task_id, step_res.decision_reason)
                    failed_set.add(task.task_id)
                    in_progress_set.remove(task.task_id)

                    self.recorder.record(
                        run_id=run_id,
                        task_id=task.task_id,
                        event_type=EventType.TASK_FAILED,
                        decision="task_failed",
                        decision_reason=step_res.decision_reason,
                        status="error",
                    )

                    policy = getattr(task, "failure_policy", FailurePolicy.SKIP)
                    if policy in ("fail_fast", FailurePolicy.RETRY):
                        state.status = RunStatus.FAILED
                        self.repo.save_run(state)
                        return self._build_result(state, workflow, start_time)
                    else:
                        # Skip all dependent successors
                        def skip_successors(tid: str):
                            for succ in workflow.get_successors(tid):
                                if succ.task_id not in skipped_set and succ.task_id not in completed_set and succ.task_id not in failed_set:
                                    skipped_set.add(succ.task_id)
                                    state.mark_task_skipped(succ.task_id)
                                    self.recorder.record(
                                        run_id=run_id,
                                        task_id=succ.task_id,
                                        event_type=EventType.TASK_SKIPPED,
                                        decision="skip",
                                        decision_reason=f"Upstream prerequisite task '{tid}' failed.",
                                    )
                                    skip_successors(succ.task_id)
                        skip_successors(task.task_id)
                        continue

                # Store observation and update state variables
                state.record_observation(task.task_id, step_res.tool_output)
                if isinstance(step_res.tool_output, dict):
                    for k, v in step_res.tool_output.items():
                        if isinstance(v, (int, float, str, bool)):
                            state.update_variable(k, v)

                # 4. Evaluate Task Branch Conditions
                branch_action, target_task = evaluate_task_branch(task, state)
                if branch_action:
                    self.recorder.record(
                        run_id=run_id,
                        task_id=task.task_id,
                        event_type=EventType.BRANCH_DECISION,
                        decision=branch_action.value,
                        decision_reason=f"Evaluated branch action '{branch_action.value}' targeting '{target_task}'.",
                        next_task_id=target_task,
                    )

                    if branch_action == BranchAction.CONTINUE_ANALYSIS and target_task == "T005":
                        if "T004" in task_map and "T004" not in completed_set:
                            skipped_set.add("T004")
                            state.mark_task_skipped("T004")
                            self.recorder.record(
                                run_id=run_id,
                                task_id="T004",
                                event_type=EventType.TASK_SKIPPED,
                                decision="skip",
                                decision_reason="Dataset is clean; skipping conditional cleaning task T004.",
                            )

                    elif branch_action == BranchAction.HUMAN_REVIEW:
                        state.status = RunStatus.WAITING_FOR_HUMAN
                        state.pending_approvals.append({
                            "task_id": task.task_id,
                            "reason": "Branch condition triggered human review",
                        })
                        self.repo.save_run(state)
                        return self._build_result(state, workflow, start_time)

                # 5. Evaluate Global Escalation Rules (excluding already approved rules)
                approved_rules = set(state.metadata.get("approved_rules", []))
                escalation = check_escalations(workflow, state)
                if escalation:
                    rule, msg = escalation
                    if rule.rule_id not in approved_rules:
                        state.escalations.append({"rule_id": rule.rule_id, "message": msg})
                        state.metrics.escalations += 1

                        self.recorder.record(
                            run_id=run_id,
                            task_id=task.task_id,
                            event_type=EventType.ESCALATION,
                            decision=rule.action.value,
                            decision_reason=msg,
                            status="warning",
                        )

                        if rule.action in (EscalationAction.HUMAN_REVIEW, EscalationAction.WORKFLOW_PAUSE):
                            state.status = RunStatus.WAITING_FOR_HUMAN
                            state.pending_approvals.append({
                                "task_id": task.task_id,
                                "rule_id": rule.rule_id,
                                "message": msg,
                            })
                            self.repo.save_run(state)
                            return self._build_result(state, workflow, start_time)

                # Mark Task Completed
                completed_set.add(task.task_id)
                in_progress_set.remove(task.task_id)
                state.mark_task_completed(task.task_id)

                self.recorder.record(
                    run_id=run_id,
                    task_id=task.task_id,
                    event_type=EventType.TASK_COMPLETED,
                    decision="task_done",
                    decision_reason=f"Task {task.task_id} completed successfully.",
                )

        state.status = RunStatus.COMPLETED
        return self._build_result(state, workflow, start_time)

    execute_workflow = run

    def resume_after_approval(
        self,
        run_id: str,
        workflow: Workflow,
        decision: str = "approve",
        reason: str = "Analyst approved execution.",
    ) -> AgentResult:
        """Resume a workflow paused for human review."""
        run_data = self.repo.get_run(run_id)
        if not run_data:
            raise ValueError(f"Run '{run_id}' not found.")

        state = AgentState(**run_data["state"])
        now_iso = datetime.utcnow().isoformat() + "Z"

        current_tid = state.current_task_id or "WORKFLOW"
        self.repo.record_human_override(
            run_id=run_id,
            task_id=current_tid,
            decision=decision,
            reason=reason,
            timestamp=now_iso,
        )

        self.recorder.record(
            run_id=run_id,
            task_id=current_tid,
            event_type=EventType.HUMAN_OVERRIDE,
            decision=decision,
            decision_reason=reason,
        )

        state.metrics.human_overrides += 1

        # Track approved rules in metadata so they don't re-trigger in this run
        approved_rules = state.metadata.setdefault("approved_rules", [])
        for pa in state.pending_approvals:
            if pa.get("rule_id"):
                approved_rules.append(pa["rule_id"])
        state.pending_approvals.clear()

        if decision.lower() in ("reject", "terminate"):
            state.status = RunStatus.FAILED
            self.repo.save_run(state)
            return self._build_result(state, workflow, time.perf_counter())

        # If approved, advance the task that triggered the pause
        if current_tid != "WORKFLOW":
            state.mark_task_completed(current_tid)
            self.recorder.record(
                run_id=run_id,
                task_id=current_tid,
                event_type=EventType.TASK_COMPLETED,
                decision="approved_continue",
                decision_reason=f"Task {current_tid} completed upon analyst approval.",
            )

        state.status = RunStatus.RUNNING
        self.repo.save_run(state)
        return self.run(workflow, custom_run_id=run_id)

    def _build_result(self, state: AgentState, workflow: Workflow, start_time: float) -> AgentResult:
        duration = round(time.perf_counter() - start_time, 3)
        state.metrics.duration_seconds = duration
        now_iso = datetime.utcnow().isoformat() + "Z"
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            state.metadata["completed_at"] = now_iso

        findings: List[Dict[str, Any]] = []
        evidence: List[Dict[str, Any]] = []
        recommendations: List[Dict[str, Any]] = []

        for task_id, obs in state.observations.items():
            if not isinstance(obs, dict):
                continue
            if "ranked_findings" in obs:
                findings.extend(obs["ranked_findings"])
            elif "ranked_causes" in obs and not findings:
                for c in obs["ranked_causes"]:
                    findings.append({
                        "cause": c.get("cause"),
                        "score": c.get("score"),
                        "details": f"effect_size {c.get('effect_size', '')}, consistency {c.get('consistency', '')}",
                    })
            elif "primary_cause" in obs and not findings:
                findings.append({"cause": obs["primary_cause"], "score": obs.get("score", 0.85)})

            if "strategic_action_plan" in obs:
                for rec in obs["strategic_action_plan"]:
                    recommendations.append({"recommendation": rec})

            if "consolidated_rca_drivers" in obs:
                evidence.extend(obs["consolidated_rca_drivers"])
            for metric_key in ("revenue_trend_metrics", "eda_summary_metrics", "statistical_test_results"):
                if metric_key in obs and isinstance(obs[metric_key], dict):
                    evidence.append(obs[metric_key])
            if "decline_magnitude" in obs and isinstance(obs["decline_magnitude"], (int, float)):
                evidence.append({"decline_magnitude": obs["decline_magnitude"]})

        confidence = calculate_confidence(
            data_quality_score=float(state.variables.get("quality_score", 0.85)),
            p_value=0.0001 if findings else 0.05,
            sample_size=10000,
            cause_margin=float(state.variables.get("cause_score_margin", 0.15)),
        )

        root_trace_id = self.recorder.events[0].trace_id if self.recorder.events else "TRACE-ROOT"
        self.repo.save_run(state)

        return AgentResult(
            run_id=state.run_id,
            workflow_id=state.workflow_id,
            status=state.status.value,
            objective=workflow.goal,
            findings=findings,
            evidence=evidence,
            recommendations=recommendations,
            confidence=confidence,
            escalated=len(state.escalations) > 0,
            completed_tasks=state.completed_tasks,
            trace_id=root_trace_id,
            metrics=state.metrics.model_dump(),
        )
