# Stage 06 — Autonomous AI Data Analyst (AADA): Workflow Engineer Layer

The **Workflow Engineer layer** converts high-level, ambiguous analytical objectives into deterministic, structured, machine-readable Directed Acyclic Graph (DAG) task execution plans.

The Workflow Engineer does **NOT** perform the actual data analysis, run an LLM ReAct loop, train models, or call the NVIDIA API.  
Instead, it creates the **authoritative execution plan** that the future **Agent Engineer** will execute.

---

## 1. System Architecture & Information Flow

```text
User Goal: "Analyze this sales dataset and determine why revenue decreased."
    ↓
Goal Parser (Deterministic Intent, Metric & Time Scope Extraction)
    ↓
Workflow Planner (Coordinates Decomposition, Graph & Rules)
    ↓
Task Decomposer (Decomposes Objective into Typed Analytical Tasks)
    ↓
Dependency Resolver (Validates DAG, Topological Order, Detects Cycles)
    ↓
Branch Conditions & Escalation Rules (Integrates Human Review & Recovery)
    ↓
Knowledge Engineer Integration (References 'retrieve_knowledge' tool contract)
    ↓
Workflow Validator (Enforces Schema & Graph Invariants)
    ↓
Machine-Readable Executable Workflow JSON
    ↓
[Future Agent Engineer Execution Loop]
```

---

## 2. Directory Structure

```text
stage-6-agentic-ai/workflow-engineer/
├── conditions/
│   ├── __init__.py
│   ├── branch_conditions.py       # Presets for missingness, quality, flags
│   ├── escalation_conditions.py   # Presets for low-confidence RCA, retry limits
│   └── validation_conditions.py   # Presets for row counts, schema requirements
├── engine/
│   ├── __init__.py
│   └── workflow_validator.py      # Comprehensive DAG & schema validator
├── examples/
│   ├── customer_churn.json        # Instantiated example workflow
│   └── sales_revenue_decline.json # Instantiated example workflow
├── planner/
│   ├── __init__.py
│   ├── dependency_resolver.py     # DAG coloring cycle detection & Kahn's topological sort
│   ├── goal_parser.py             # Deterministic goal parser & entity extractor
│   ├── task_decomposer.py         # Intent-to-tasks decomposition engine
│   └── workflow_planner.py        # End-to-end plan synthesizer
├── schemas/
│   ├── __init__.py
│   ├── condition.py               # BranchCondition, ComparisonOperator, BranchAction
│   ├── escalation.py              # EscalationRule, EscalationTrigger, EscalationAction
│   ├── task.py                    # Task, TaskType, FailurePolicy, RetryPolicy
│   └── workflow.py                # Workflow, WorkflowInput, WorkflowOutput
├── tests/
│   ├── conftest.py
│   ├── test_conditions.py         # Branch condition evaluation tests
│   ├── test_dependencies.py       # DAG, cycle, and reachability tests
│   ├── test_escalation.py         # Escalation rule trigger tests
│   ├── test_goal_parser.py        # Deterministic parsing & time-scope tests
│   ├── test_schemas.py            # Pydantic schema validation tests
│   ├── test_task_decomposition.py # Goal breakdown & tool assignment tests
│   └── test_workflow_validation.py# Production template & validator tests
├── workflows/
│   ├── __init__.py
│   ├── registry.py                # Template loader & catalog manager
│   └── templates/
│       ├── anomaly_investigation.json # Anomaly detection & explanation template
│       ├── customer_analysis.json     # Segmentation & churn template
│       ├── dataset_analysis.json      # End-to-end general analysis template
│       ├── predictive_analysis.json   # Machine learning pipeline template
│       └── revenue_decline.json       # Multi-branch revenue RCA template
└── README.md
```

---

## 3. Core Pydantic Schemas

### A. Workflow Schema (`schemas/workflow.py`)
```json
{
  "workflow_id": "WF-REVENUE-001",
  "name": "Sales - Revenue Decline",
  "version": "1.0",
  "goal": "Determine the causes of revenue decline and decompose performance drivers.",
  "entry_task": "T001",
  "terminal_tasks": ["T013"],
  "tasks": [...],
  "escalation_rules": [...],
  "metadata": {
    "domain": "sales",
    "intent_type": "revenue_decline",
    "topological_order": ["T001", "T002", "T003", "T004", "T005", "T006", "T007", "T008", "T009", "T010", "T011", "T012", "T013"]
  }
}
```

### B. Task Schema (`schemas/task.py`)
Each analytical task defines contractual boundaries for execution:
- `task_id`: Deterministic ID (`T001`, `T002`, etc.).
- `name`: Human-readable title.
- `task_type`: Permitted `TaskType` enum (`ingestion`, `profiling`, `validation`, `cleaning`, `eda`, `statistical_analysis`, `anomaly_detection`, `segmentation`, `machine_learning`, `root_cause_analysis`, `visualization`, `recommendation`, `reporting`, `human_review`).
- `tool`: Contractual tool name invoked by the Agent (`load_dataset`, `profile_dataset`, `validate_data_quality`, `clean_dataset`, `eda_analysis`, `statistical_analysis`, `detect_anomalies`, `segment_customers`, `train_model`, `root_cause_analysis`, `generate_report`, etc.).
- `dependencies`: List of parent `task_id` prerequisites.
- `required_inputs` & `expected_outputs`: Invariant input and output contracts.
- `conditions`: List of `BranchCondition` evaluated upon completion.
- `failure_policy`: `retry`, `skip`, `fallback`, `pause`, `escalate`, `terminate`.
- `retry_policy`: `max_retries`, `backoff_factor`, `initial_delay_sec`, `fallback_task_id`.
- `knowledge_retrieval`: Optional reference to the Knowledge Engineer `retrieve_knowledge` tool contract.

---

## 4. Multi-Branch Task Graph: Sales Revenue Decline

The revenue decline template decomposes complex investigation into parallel analytical branches before root-cause consolidation:

```text
       T001 (Load Sales Dataset)
             ↓
       T002 (Profile Sales Dataset)
             ↓
       T003 (Validate Data Quality)
        /                     \
       /                       \
[needs_cleaning == True]    [needs_cleaning == False]
     ↓                           ↓
T004 (Clean Dataset)             |
     \                           /
      \                         /
       ↓                        ↓
       T005 (Analyze Revenue Trend)
        ├── T006 (Analyze Product Performance Branch)
        ├── T007 (Analyze Customer Segment Branch)
        ├── T008 (Analyze Regional & Channel Branch)
        └── T009 (Detect Revenue Anomalies Branch)
             \    |    |    /
              \   |    |   /
               ↓   ↓    ↓  ↓
       T010 (Rate vs Volume Mix Analysis)
             ↓
       T011 (Synthesize Evidence-Based Findings)
             ↓
       T012 (Generate Strategic Recommendations)
             ↓
       T013 (Generate Final Executive Report)
```

---

## 5. Branch Conditions & Escalation Policies

### Branch Conditions (`schemas/condition.py`)
Control dynamic runtime routing:
- `missing_rate > 0.30` $ightarrow$ `human_review`
- `missing_rate > 0.05` $ightarrow$ `data_cleaning`
- `quality_score < 0.70` $ightarrow$ `data_cleaning`
- `decline_magnitude <= 0.0` $ightarrow$ `skip_task` (or route to growth analysis)

### Escalation Rules (`schemas/escalation.py`)
Ensure the workflow never fails silently:
1. **Critical Data Quality Failure**: `quality_score < 0.50` $ightarrow$ `human_review`.
2. **Low Confidence Root Cause**: `root_cause_confidence < 0.60` $ightarrow$ `human_review`.
3. **Ambiguous Competing Causes**: `top_cause_score - second_cause_score < 0.05` $ightarrow$ `human_review`.
4. **Missing Required Columns**: `missing_required_columns_count > 0` $ightarrow$ `workflow_pause`.
5. **Task Max Retries Exceeded**: `retry_count >= 3` $ightarrow$ `workflow_escalation`.
6. **High-Impact Recommendation**: `business_impact_level == 'high'` $ightarrow$ `human_review`.

---

## 6. Knowledge Engineer Integration

Tasks that require domain methodology reference the existing `retrieve_knowledge` tool contract:

```json
{
  "task_id": "T003",
  "name": "Validate Data Quality",
  "tool": "validate_data_quality",
  "knowledge_retrieval": {
    "query": "Validation of missing numerical values and hard boundary limits",
    "category": "data_quality",
    "top_k": 2
  }
}
```

The future Agent Engineer invokes `retrieve_knowledge(...)` to obtain authoritative rules (e.g., `DQ-001`, `DC-001`) from Stage 06 Knowledge Engineer before executing the analytical step.

---

## 7. Available Workflow Templates

| Template Key | Workflow ID | Tasks | Intent Domain | Description |
| :--- | :--- | :--- | :--- | :--- |
| `revenue_decline` | `WF-REVENUE-001` | 13 | Sales / Financial | Multi-branch product, customer, channel, and volume mix RCA. |
| `customer_analysis` | `WF-CUSTOMER-001` | 8 | Customer / Marketing | RFM segmentation, behavioral trends, and retention cohort decay. |
| `anomaly_investigation`| `WF-ANOMALY-001` | 6 | Operational / Risk | Multi-method outlier isolation, historical comparison, and cause ranking. |
| `predictive_analysis` | `WF-PREDICTIVE-001`| 7 | Machine Learning | CV protocol selection, feature engineering, model training, and evaluation. |
| `dataset_analysis` | `WF-DATASET-001` | 12 | General EDA | End-to-end dataset profiling, statistical tests, and executive reporting. |

---

## 8. Workflow Validation Engine (`engine/workflow_validator.py`)

Every generated or loaded workflow is verified against strict mathematical and schema invariants:
1. **Uniqueness**: Unique `workflow_id` and unique `task_id` across all tasks.
2. **Taxonomy Conformance**: Every task type belongs to the `TaskType` enum.
3. **Referential Integrity**: All dependencies refer to existing task IDs.
4. **Acyclicity**: DFS graph coloring detects and rejects any circular dependency ($A ightarrow B ightarrow A$).
5. **Reachability**: All tasks must be reachable from the `entry_task` via Kahn's algorithm / BFS.
6. **Terminal Validity**: Terminal tasks must exist and have 0 outgoing dependencies.
7. **Condition & Tool Soundness**: Valid tool names, valid operator evaluations, and valid branch targets.

---

## 9. Usage Guide

### Programmatic Planning Example
```python
from planner.workflow_planner import WorkflowPlanner
from engine.workflow_validator import validate_workflow

planner = WorkflowPlanner()

# Generate deterministic workflow from natural language
workflow = planner.plan(
    "Analyze this quarterly sales dataset and determine why revenue decreased during the last quarter."
)

# Validate graph invariants
result = validate_workflow(workflow)
assert result.valid is True

print(f"Generated Workflow: {workflow.workflow_id} ({len(workflow.tasks)} tasks)")
print(f"Execution Order: {' -> '.join(result.execution_order)}")
```

### Template Registry Discovery
```python
from workflows.registry import WorkflowRegistry

registry = WorkflowRegistry()

# List available templates
catalog = registry.list_templates()
for t in catalog:
    print(f"[{t['workflow_id']}] {t['name']} - {t['task_count']} tasks")

# Fetch template by intent
wf = registry.get_template_for_intent("revenue_decline")
```

---

## 10. How the Future Agent Engineer Consumes the Workflow

1. **Step 1: Receive Plan**: The Agent receives the validated `Workflow` JSON specification.
2. **Step 2: Traverse Execution Order**: The Agent iterates through `workflow.metadata["topological_order"]`.
3. **Step 3: Consult Knowledge Base**: If `task.knowledge_retrieval` is set, the Agent calls `retrieve_knowledge(query, category)` to retrieve authoritative methodology.
4. **Step 4: Execute Analytical Tool**: The Agent invokes `task.tool` with `task.required_inputs`.
5. **Step 5: Evaluate Branch Conditions**: The Agent checks `task.conditions` against runtime metrics (e.g. `needs_cleaning`). If triggered, the Agent branches or routes accordingly.
6. **Step 6: Handle Errors / Escalations**: If errors occur, the Agent applies `task.failure_policy` and checks `workflow.escalation_rules`. If an escalation triggers, execution pauses and alerts the supervisor.
