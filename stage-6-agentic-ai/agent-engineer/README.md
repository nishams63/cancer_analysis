# Stage 06 — Agentic AI: Agent Engineer Layer
**Project: AADA (Autonomous AI Data Analyst)**

---

## 1. Overview & Role Boundaries

The **Agent Engineer** is the execution engine of the AADA system. It is responsible for orchestrating LLM tool-calling, running the ReAct reasoning loop, executing the Directed Acyclic Graph (DAG) created by the **Workflow Engineer**, persisting an immutable execution trace, resolving competing analytical findings, computing deterministic confidence scores, and supporting Human-in-the-Loop escalations.

### Architectural Boundary:
```
Stage 06 Pipeline:
[Knowledge Engineer] ──> [Workflow Engineer] ──> [Agent Engineer] ──> [Evaluation Engineer] ──> [Integration Engineer]
 (Curates Domain KB)     (Compiles Task DAG)     (Executes Tasks &      (Assesses Accuracy      (Final End-to-End
                                                  Tool ReAct Loops)      & Faithfulness)         Deployment)
```

**What the Agent Engineer Does:**
- Traverses and executes analytical workflows according to DAG dependencies.
- Executes analytical tools (profiling, EDA, statistical tests, anomaly detection, root cause analysis, etc.).
- Executes a robust ReAct (Thought -> Action -> Observation) reasoning loop.
- Calculates deterministic confidence scores without LLM hallucinations.
- Resolves conflicting causal hypotheses with a transparent, weighted multi-factor scoring rubric.
- Supports Human-in-the-Loop pause, resume, and decision overrides.
- Records all state transitions, tool invocations, and decisions in an SQLite trace repository.
- Provides a FastAPI server for external orchestration.

**What the Agent Engineer Does NOT Do:**
- Does not modify or re-compile the workflow DAG (that belongs to Workflow Engineer).
- Does not re-index knowledge embeddings or build the KB (that belongs to Knowledge Engineer).
- Does not compute ground-truth evaluation benchmark metrics (that belongs to Evaluation Engineer).
- Does not build a web frontend UI.

---

## 2. Architecture & Execution Flow

```
                      +---------------------------------------+
                      |          Analytical Workflow          |
                      |          (from Workflow Eng)          |
                      +-------------------+-------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              AGENT EXECUTOR                                       |
|                                                                                   |
|  +---------------------+       +---------------------+       +-----------------+  |
|  |     DAG Planner     | ----> |  Ready Task Queue   | ----> |  Task Executor  |  |
|  +---------------------+       +---------------------+       +--------+--------+  |
|                                                                       |           |
|                               +---------------------------------------+           |
|                               v                                                   |
|                +------------------------------+                                   |
|                |         ReAct Loop           |                                   |
|                |  Thought -> Action -> Obs    |                                   |
|                +--------------+---------------+                                   |
|                               |                                                   |
|        +----------------------+----------------------+                            |
|        v                                             v                            |
|  +------------+                              +---------------+                    |
|  | Tool Calls |                              | LLM Provider  |                    |
|  | (Adapters) |                              | (NVIDIA/Mock) |                    |
|  +------------+                              +---------------+                    |
|        |                                             |                            |
|        +----------------------+----------------------+                            |
|                               v                                                   |
|                +------------------------------+                                   |
|                |       Decision Engine        |                                   |
|                |  - Branch Evaluation         |                                   |
|                |  - Escalation Check          |                                   |
|                |  - Conflict Resolution       |                                   |
|                +--------------+---------------+                                   |
+-------------------------------|---------------------------------------------------+
                                |
        +-----------------------+-----------------------+
        v                                               v
+-------------------------------+               +-------------------------------+
|       Trace Repository        |               |      Agent Result Schema      |
|  (SQLite: Runs, Trace Events, |               |  (Status, Confidence, Summary,|
|       Human Overrides)        |               |     Artifacts, Metrics)       |
+-------------------------------+               +-------------------------------+
```

---

## 3. Directory Structure

```
stage-6-agentic-ai/agent-engineer/
├── schemas/                # Pydantic data models
│   ├── agent.py            # RunStatus, AgentMetrics, AgentState
│   ├── trace.py            # EventType, TraceEvent
│   ├── result.py           # AgentResult
│   └── __init__.py
├── llm/                    # LLM integration layer
│   ├── base.py             # LLMProvider ABC, LLMResponse
│   ├── prompts.py          # Structured system & decision prompts
│   ├── mock.py             # Deterministic MockLLMProvider for offline test suites
│   ├── nvidia.py           # Real NVIDIA API provider with retry & backoff
│   └── __init__.py
├── tools/                  # Analytical tool calling subsystem
│   ├── contracts.py        # ToolMetadata, ToolRiskLevel
│   ├── adapters.py         # 15 analytical tool adapters
│   ├── registry.py         # ToolRegistry with execution safety & allowlisting
│   └── __init__.py
├── reasoning/              # Decision-making & reasoning
│   ├── confidence.py       # Deterministic confidence calculation formula
│   ├── conflict_resolution.py # Multi-factor hypothesis scoring & margin validation
│   ├── decision_engine.py  # Branch evaluator & escalation manager
│   ├── react_loop.py       # ReAct reasoning loop
│   └── __init__.py
├── execution/              # Task execution & scheduling
│   ├── retry.py            # Exponential backoff retry handler
│   ├── branch_executor.py  # Conditional branch evaluation
│   ├── escalation.py       # Human review triggers & escalation logic
│   ├── parallel.py         # DAG task readiness resolver
│   ├── task_executor.py    # Per-task execution coordinator
│   └── __init__.py
├── trace/                  # Auditability & persistence
│   ├── models.py           # SQLite database schema
│   ├── repository.py       # SQLite trace & state repository
│   ├── recorder.py         # Thread-safe event recorder
│   └── __init__.py
├── agent/                  # Core Agent runtime
│   ├── context.py          # TaskExecutionContext builder
│   ├── decision.py         # AgentDecision container
│   ├── state.py            # AgentStateManager
│   ├── executor.py         # AgentExecutor (Full DAG orchestration)
│   └── __init__.py
├── api/                    # HTTP REST service
│   ├── server.py           # FastAPI endpoints (/agent/run, /resume, /trace, etc.)
│   └── __init__.py
├── examples/               # Runnable end-to-end examples
│   └── run_revenue_analysis.py
├── tests/                  # 14 pytest modules (30+ test cases)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 4. Key Mechanisms

### A. Deterministic Confidence Score
Rather than having an LLM invent a confidence score, the Agent computes it deterministically:
$$Confidence = 0.35 \times \text{data\_quality} + 0.25 \times \text{stat\_sig} + 0.20 \times \text{consistency} + 0.20 \times \text{evidence\_grounding}$$
- Penalized by $0.15$ if any critical anomalies remain unaddressed.
- Clamped strictly to $[0.0, 1.0]$.

### B. Multi-Factor Conflict Resolution
When competing causes are identified (e.g., Price Increase vs. Competitor Launch):
$$Score = 0.30 \times \text{effect\_size} + 0.25 \times \text{sample\_size} + 0.25 \times \text{evidence} + 0.20 \times \text{consistency}$$
- If the winning hypothesis leads the runner-up by $< 0.10$ (margin check), the conflict is flagged as ambiguous and escalated for human review.

### C. Human-in-the-Loop Escalation
High-risk tools (`train_model`, `human_review`) or conditions triggering escalation pause workflow execution, transitioning state to `WAITING_FOR_HUMAN`. Execution can be resumed via `/agent/resume/{id}` or manual override.

---

## 5. Usage & Testing

### Running Tests
Run all agent unit and integration tests with pytest:
```bash
pytest stage-6-agentic-ai/agent-engineer/tests -v
```

### Running the Example
```bash
python stage-6-agentic-ai/agent-engineer/examples/run_revenue_analysis.py
```

### Launching the API
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```


---

## 9. Deliberative AI Agent Architecture (Oncology Clinical Decision Support)

The Stage 06 Agent Engineer includes a specialized **Deliberative AI Agent** (`deliberation.DeliberativeAgent`) designed for Oncology Clinical Decision Support (CDSS). Unlike purely reactive ReAct loops that immediately call tools upon receiving an instruction, the deliberative agent reasons across an explicit, auditable 12-stage deliberation cycle.

### Core Architectural Principle: Clinical Decision Support (CDSS)
> **CRITICAL CLINICAL BOUNDARY**:  
> The system is strictly a **Clinical Decision Support System**. It does **NOT** autonomously diagnose, prescribe treatment regimens, or replace an oncologist. The treating clinician retains sole clinical responsibility for patient care. Every output includes a mandatory non-autonomous clinical disclaimer.

### 12-Stage Deliberation Reasoning Cycle

```
                    USER
                      │
                      ▼
             Clinical Question
                      │
                      ▼
          ┌─────────────────────┐
          │ DELIBERATIVE AGENT  │
          └──────────┬──────────┘
                     │
                     ▼
          1. UNDERSTAND QUESTION & CONTEXT
                     │
                     ▼
          2. BUILD & VALIDATE CLINICAL PLAN
                     │
                     ▼
          3. IDENTIFY EVIDENCE NEEDS
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 Knowledge Engineer       Trusted Patient Data
          │                     │
          └──────────┬──────────┘
                     ▼
          4. RETRIEVE EVIDENCE
                     │
                     ▼
          5. ANALYZE & SYNTHESIZE EVIDENCE
                     │
                     ▼
          6. FORMULATE COMPETING HYPOTHESES
                     │
                     ▼
          7. COMPARE HYPOTHESES (Score & Rank)
                     │
                     ▼
          8. CHECK CONTRADICTIONS & TOXICITY
                     │
                     ▼
          9. ASSESS UNCERTAINTY (Low/Mod/High/Critical)
                     │
               ┌─────┴─────┐
               ▼           ▼
            SAFE        UNCERTAIN / CONFLICT
               │           │
               │           ▼
               │      10. REQUEST HUMAN REVIEW (WAITING_FOR_HUMAN)
               │           │
               └─────┬─────┘
                     ▼
          11. EXECUTE & VERIFY CONCLUSION (7 Safety Gates)
                     │
                     ▼
          12. PRESENT CLINICAL DECISION SUPPORT SUMMARY
```

### The 7 Clinical Safety Gates

A consequential clinical recommendation cannot bypass these 7 deterministic safety gates:
1. **Gate 1 (Patient Facts Validated)**: Primary cancer type, clinical stage, and key patient identifiers verified against trusted EMR records; missing facts marked `MISSING_INFORMATION`.
2. **Gate 2 (Plan Validated)**: Deliberation plan contains actionable steps, relevance confirmation, and explicit safety checks.
3. **Gate 3 (Evidence Sufficient)**: Non-zero validated clinical guidelines retrieved from Knowledge Engineer.
4. **Gate 4 (Contradictions Checked)**: Audit for fact-vs-fact (e.g., stage I vs distant metastases) and fact-vs-option (e.g., cisplatin in renal failure, osimertinib in active ILD).
5. **Gate 5 (Uncertainty Acceptable)**: Epistemic and aleatoric uncertainty quantified; high/critical uncertainty halts execution.
6. **Gate 6 (Human Review Evaluated)**: Consequential options, conflicting signals, or high uncertainty mandate `WAITING_FOR_HUMAN`.
7. **Gate 7 (Conclusion Verified)**: Claims substantiated by evidence, numerical bounds verified, and non-autonomous disclaimer attached.

### Private Reasoning Safety
The Deliberative Agent strictly prevents the leakage or persistence of raw chain-of-thought, token-level scratchpads, or private internal reasoning. All trace milestones are recorded as auditable structured event summaries:
- `QUESTION_RECEIVED`
- `CONTEXT_VALIDATED`
- `PLAN_CREATED`
- `EVIDENCE_REQUIREMENTS_IDENTIFIED`
- `KNOWLEDGE_RETRIEVED`
- `EVIDENCE_ANALYZED`
- `HYPOTHESES_GENERATED`
- `HYPOTHESES_COMPARED`
- `CONTRADICTION_DETECTED` / `CONTRADICTIONS_AUDITED`
- `UNCERTAINTY_ASSESSED`
- `HUMAN_REVIEW_REQUIRED` (or `HUMAN_APPROVED` / `HUMAN_REJECTED`)
- `SAFETY_CHECK_COMPLETED`
- `CONCLUSION_VERIFIED`
- `RESULT_GENERATED`

### Running the Oncology Deliberation Demo
```powershell
.venv\Scripts\python.exe stage-6-agentic-ai/agent-engineer/examples/run_oncology_deliberation.py
```
