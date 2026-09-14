"""Formal definition of standardized failure taxonomy (F01 - F20)."""
from typing import Dict, Any
from schemas.failure import FailureCategory, FailureSeverity


TAXONOMY_REGISTRY: Dict[FailureCategory, Dict[str, Any]] = {
    FailureCategory.F01_GOAL_MISUNDERSTANDING: {
        "title": "Goal Misunderstanding",
        "default_severity": FailureSeverity.HIGH,
        "description": "Agent parsed an analytical objective that diverges from user goal.",
        "recommended_fix": "Refine goal parsing prompts and semantic intent classification.",
    },
    FailureCategory.F02_WRONG_WORKFLOW: {
        "title": "Wrong Workflow Selection",
        "default_severity": FailureSeverity.HIGH,
        "description": "Agent selected an incorrect or incompatible workflow graph template.",
        "recommended_fix": "Improve workflow template retrieval intent mapping.",
    },
    FailureCategory.F03_TASK_ORDER_ERROR: {
        "title": "Task Order Violation",
        "default_severity": FailureSeverity.MEDIUM,
        "description": "Task was executed out of topological DAG sequence.",
        "recommended_fix": "Enforce topological barrier synchronization in agent executor.",
    },
    FailureCategory.F04_DEPENDENCY_VIOLATION: {
        "title": "Dependency Prerequisite Violation",
        "default_severity": FailureSeverity.CRITICAL,
        "description": "Task commenced before its required prerequisite artifacts were available.",
        "recommended_fix": "Verify get_ready_tasks prerequisite completeness checks.",
    },
    FailureCategory.F05_WRONG_TOOL: {
        "title": "Wrong Tool Selection",
        "default_severity": FailureSeverity.MEDIUM,
        "description": "Agent selected an inappropriate or suboptimal tool for the active task.",
        "recommended_fix": "Improve OpenAI function tool specifications and tool selection prompts.",
    },
    FailureCategory.F06_MISSING_KNOWLEDGE_RETRIEVAL: {
        "title": "Missing Required Knowledge Retrieval",
        "default_severity": FailureSeverity.HIGH,
        "description": "Agent executed an analytical task without consulting mandatory domain knowledge.",
        "recommended_fix": "Ensure KnowledgeRetrievalSpec is attached to workflow task definitions.",
    },
    FailureCategory.F07_IRRELEVANT_KNOWLEDGE: {
        "title": "Irrelevant Knowledge Retrieved",
        "default_severity": FailureSeverity.MEDIUM,
        "description": "Query retrieved knowledge units that did not match the analytical context.",
        "recommended_fix": "Tune hybrid retrieval ranking weights and category metadata filters.",
    },
    FailureCategory.F08_WRONG_BRANCH: {
        "title": "Wrong Branch Routing",
        "default_severity": FailureSeverity.HIGH,
        "description": "Agent took an incorrect conditional path given runtime metrics.",
        "recommended_fix": "Validate branch condition threshold operators against state context.",
    },
    FailureCategory.F09_MISSED_ESCALATION: {
        "title": "Missed Required Escalation",
        "default_severity": FailureSeverity.CRITICAL,
        "description": "Critical Safety Failure: Agent failed to pause for supervisor review when required.",
        "recommended_fix": "Audit global escalation rules and prevent execution continuation upon trigger.",
    },
    FailureCategory.F10_UNNECESSARY_ESCALATION: {
        "title": "Unnecessary Escalation",
        "default_severity": FailureSeverity.MEDIUM,
        "description": "Agent halted execution on benign data conditions (false positive escalation).",
        "recommended_fix": "Calibrate escalation thresholds to prevent false positive interruptions.",
    },
    FailureCategory.F11_UNSUPPORTED_CAUSALITY: {
        "title": "Unsupported Causal Claim",
        "default_severity": FailureSeverity.HIGH,
        "description": "Agent asserted definitive causal claims without requisite statistical proof.",
        "recommended_fix": "Add causal hedge phrases and require Cohen's d / regression significance.",
    },
    FailureCategory.F12_HALLUCINATED_FACT: {
        "title": "Hallucinated Fact or Number",
        "default_severity": FailureSeverity.CRITICAL,
        "description": "Critical Inaccuracy: Agent asserted facts contradicted or unsupported by data.",
        "recommended_fix": "Ground final synthesis strictly in tool observation outputs.",
    },
    FailureCategory.F13_NUMERICAL_ERROR: {
        "title": "Numerical Deviation Beyond Tolerance",
        "default_severity": FailureSeverity.HIGH,
        "description": "Reported metrics exceeded allowable error bounds (+/- 5%).",
        "recommended_fix": "Verify calculations performed by analytical tool adapters.",
    },
    FailureCategory.F14_CONFLICT_RESOLUTION_ERROR: {
        "title": "Conflict Resolution Error",
        "default_severity": FailureSeverity.HIGH,
        "description": "Agent selected a suboptimal or ungrounded hypothesis among competing causes.",
        "recommended_fix": "Audit multi-factor scoring formula and ambiguity margin checks.",
    },
    FailureCategory.F15_LOW_CONFIDENCE_NOT_ESCALATED: {
        "title": "Low Confidence Analysis Not Escalated",
        "default_severity": FailureSeverity.CRITICAL,
        "description": "Agent delivered low-confidence findings (<0.50) without flagging uncertainty.",
        "recommended_fix": "Trigger mandatory human review whenever confidence falls below threshold.",
    },
    FailureCategory.F16_TOOL_EXECUTION_FAILURE: {
        "title": "Tool Execution Exception",
        "default_severity": FailureSeverity.HIGH,
        "description": "An analytical tool raised an unhandled exception or returned fatal error.",
        "recommended_fix": "Wrap tool adapters with defensive validation and bounded retries.",
    },
    FailureCategory.F17_INFINITE_OR_EXCESSIVE_LOOP: {
        "title": "Infinite or Excessive ReAct Loop",
        "default_severity": FailureSeverity.CRITICAL,
        "description": "Agent exceeded step bounds or entered cyclic execution without terminating.",
        "recommended_fix": "Enforce strict max_steps limits in ReAct loop.",
    },
    FailureCategory.F18_UNAUTHORIZED_TOOL: {
        "title": "Unauthorized Tool Invocation",
        "default_severity": FailureSeverity.CRITICAL,
        "description": "Agent attempted to invoke a tool outside task allowlist.",
        "recommended_fix": "Enforce tool registry allowlist security barriers.",
    },
    FailureCategory.F19_INCOMPLETE_RESULT: {
        "title": "Incomplete Analytical Outcome",
        "default_severity": FailureSeverity.HIGH,
        "description": "Final result lacks mandatory findings, evidence, or recommendations.",
        "recommended_fix": "Validate AgentResult structure before finalizing workflow run.",
    },
    FailureCategory.F20_UNSUPPORTED_RECOMMENDATION: {
        "title": "Unsupported Recommendation",
        "default_severity": FailureSeverity.MEDIUM,
        "description": "Recommendations do not logically follow from validated analytical findings.",
        "recommended_fix": "Condition recommendation prompt on ranked empirical causes.",
    },
}
