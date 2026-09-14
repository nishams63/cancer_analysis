# Stage 06 — Agentic AI: Integration Engineer Layer

**Autonomous AI Data Analyst (AADA)**  
Production-ready integration, orchestration, streaming, human-in-the-loop, and observability service.

---

## 1. Architecture Overview

```
                     USER / DASHBOARD
                            │
                            ▼
              ┌───────────────────────────┐
              │  AADA Integration API     │
              │  (FastAPI + SSE Stream)   │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │      Workflow Planner     │
              │ (Workflow Engineer Layer) │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │       Agent Executor      │
              │  (Agent Engineer Layer)   │
              └──────┬─────────────┬──────┘
                     │             │
              ┌──────┘             └──────┐
              ▼                           ▼
       Knowledge Tool                 AI Tools
   (Knowledge Engineer)         (Allowlisted Tools)
              │                           │
              └─────────────┬─────────────┘
                            ▼
                     Execution Trace
                     (SQLite Auditing)
                            │
                            ▼
              ┌───────────────────────────┐
              │    Evaluation Engineer    │
              │ (Process, Safety, Outcome)│
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │    Integration Response   │
              │  & Presentation Formatter │
              └───────────────────────────┘
```

---

## 2. Component Integration

The Integration Engineer is strictly an orchestration and presentation layer:
- **Workflow Engineer:** Synthesizes deterministic task DAGs from natural language goals.
- **Agent Engineer:** Executes workflows, enforces tool allowlists, derives confidence, handles retries, and records SQLite audit traces.
- **Knowledge Engineer:** Queried through the agent's `retrieve_knowledge` tool interface.
- **Evaluation Engineer:** Evaluates process fidelity, non-compensatory safety rules, and outcome metrics without LLM-as-a-judge subjectivity.

---

## 3. End-to-End Execution Flow

1. User submits natural language goal via `POST /api/v1/aada/run`.
2. Pipeline calls `WorkflowPlanner().plan(goal)`.
3. Agent executes tasks according to topological order.
4. If ambiguous conditions or low confidence occur, pipeline pauses in `WAITING_FOR_HUMAN` state.
5. Analyst issues `APPROVE`, `REJECT`, or `OVERRIDE` to resume execution.
6. Execution artifacts and traces are evaluated by `ExperimentRunner`.
7. Unified user presentation is produced and broadcast via Server-Sent Events (SSE).
