# Stage 5 GenAI Synthetic Oncology Stress-Test Engine — Integration Layer

Act as the Senior AI Integration Engineer, MLOps Engineer, Backend Systems Engineer, and Evaluation Orchestration Engineer responsible for the end-to-end orchestration, operational, and visualization layer of Stage 5.

```text
DATA ENGINEER
     ↓
EDA / PROMPT ENGINEER
     ↓
GENAI ENGINEER
     ↓
EVALUATION ENGINEER
     ↓
INTEGRATION ENGINEER
     ↓
─────────────────────────────────────────────────────────────
COMPLETE REPRODUCIBLE STAGE 5 STRESS-TEST ENGINE PLATFORM
─────────────────────────────────────────────────────────────
```

---

## 1. Directory Structure

```text
Integration Engineer/
├── configs/
│   ├── integration_config.yaml    # Master batch size, timeout, concurrency, retry rules
│   ├── pipeline_config.yaml       # Step execution toggles (generation, rag, eval, stages, rank)
│   ├── storage_config.yaml        # SQLite database path & multi-format export paths
│   └── dashboard_config.yaml      # Dashboard host/port, CORS, metrics refresh interval
├── runtime/
│   ├── batches/                   # Runtime batch records
│   ├── checkpoints/               # Scenario & step checkpoints for fault recovery
│   ├── cache/                     # Cached models and reference parquets
│   └── locks/                     # Concurrency & file locks
├── results/
│   ├── stage5.db                  # Master SQLite database (10 relational tables)
│   ├── batches/                   # batch_results.jsonl
│   ├── scenarios/                 # scenario_results.jsonl
│   ├── failures/                  # failure_results.jsonl
│   ├── rankings/                  # wildcard_candidates.csv
│   └── exports/                   # Parquet & CSV exports
├── reports/
│   ├── batch_summary.md           # End-to-end batch execution metrics
│   ├── integration_health.md      # Upstream dependency & service audit
│   ├── failure_summary.md         # Cross-stage failure breakdown & cluster analysis
│   └── wildcard_summary.md        # Ranked wildcard candidates with explanations
├── manifests/
│   ├── batch_manifest.json        # Deterministic lineage, seed, and input hashes
│   ├── integration_manifest.json  # Artifact audit trail & sha256 signatures
│   └── system_manifest.json       # Version matrix across all 5 stages
├── src/
│   ├── integration/
│   │   ├── orchestrator.py        # Master batch orchestrator (18-step coordination)
│   │   ├── batch_manager.py       # Batch lifecycle state machine
│   │   ├── scenario_runner.py     # Scenario execution wrapper & retry handling
│   │   ├── pipeline_state.py      # Granular scenario-level state tracker
│   │   ├── checkpoint_manager.py  # Checkpoint persistence & resumption
│   │   ├── health_checks.py       # Pre-flight service health checks
│   │   └── dependency_validator.py# Upstream asset validator
│   ├── adapters/
│   │   ├── base_stage_adapter.py  # BaseStageAdapter contract & normalized schema
│   │   ├── stage1_adapter.py      # Tabular ML hazard & risk adapter
│   │   ├── stage2_adapter.py      # Multimodal DL imaging adapter (SKIPPED_INPUT_UNAVAILABLE)
│   │   ├── stage3_adapter.py      # Clinical NLP entity extraction & triage urgency adapter
│   │   └── stage4_adapter.py      # SLM treatment & resistance bypass adapter
│   ├── storage/
│   │   ├── result_store.py        # Master SQLite repository with WAL mode
│   │   ├── batch_store.py         # Batch metadata & state manager
│   │   ├── scenario_store.py      # Scenario & evaluation query repository
│   │   ├── failure_store.py       # Failure taxonomy records & ranking store
│   │   └── export_service.py      # JSONL, CSV & Parquet exporter
│   ├── ranking/
│   │   ├── wildcard_ranker.py     # Configurable multi-factor candidate ranker
│   │   ├── candidate_filter.py    # Gatekeeper filtering invalid/unfaithful scenarios
│   │   └── ranking_explainer.py   # Clinical rationale generator
│   ├── dashboard/
│   │   ├── data_service.py        # SQLite query service
│   │   ├── metrics_service.py     # Aggregated summary & failure analytics
│   │   └── dashboard_api.py       # FastAPI backend & embedded HTML dashboard
│   └── utils/
│       ├── config.py              # YAML config loader & merger
│       ├── logging.py             # Structured logger with secret redactor
│       ├── ids.py                 # Batch ID & Scenario ID generators
│       ├── timing.py              # Latency tracker & stopwatch
│       └── serialization.py       # JSON/YAML/Parquet safe serializers
├── dashboard/
│   └── index.html                 # Interactive dashboard UI
├── pipelines/
│   ├── run_stage5_pipeline.py     # Direct pipeline execution script
│   └── rerun_failed_scenarios.py  # Targeted rerun script (--batch-id, --failure-code)
├── tests/                         # 14 unit tests (100% pass)
└── verify_done.py                 # Programmatic Definition of Done (Q1–Q10)
```

---

## 2. Master CLI Runner

Run from the repository root:
```bash
# Run standard 20-scenario stress-test batch
python run_stage5.py --n 20 --seed 42

# Run specific prompt scenario
python run_stage5.py --scenario PROMPT-R01 --n 10 --seed 42

# Targeted rerun of failed scenarios
python "stage 5 Gen-AI/Integration Engineer/pipelines/rerun_failed_scenarios.py" --batch-id BATCH-20260911-7100
```

---

## 3. REST API & Dashboard Endpoints

Start the dashboard service:
```bash
uvicorn src.dashboard.dashboard_api:app --host 127.0.0.1 --port 8085
```

Endpoints:
* `GET /`: Interactive web UI
* `GET /api/stage5/health`: Pre-flight dependency & subsystem health
* `GET /api/stage5/summary`: Aggregated realism, fidelity, faithfulness, and failure counts
* `GET /api/stage5/batches`: List all batch runs with lifecycle status
* `GET /api/stage5/scenarios`: List evaluated scenarios
* `GET /api/stage5/scenarios/{scenario_id}`: Full scenario dossier (patient, RAG, narrative, eval, stages)
* `GET /api/stage5/failures`: Mined failure modes filtered by stage or taxonomy code
* `GET /api/stage5/rankings`: Ranked wildcard candidates tagged strictly as `CANDIDATE ONLY`
* `GET /api/stage5/counterfactuals`: Counterfactual sensitivity & decision instability records

---

## 4. Verification & Testing

Execute unit tests:
```bash
pytest "stage 5 Gen-AI/Integration Engineer/tests" -v
```

Execute programmatic Definition of Done:
```bash
python "stage 5 Gen-AI/Integration Engineer/verify_done.py"
```
