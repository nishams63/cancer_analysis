# Stage 06 — Autonomous AI Data Analyst (AADA): Knowledge Engineer Layer

Authoritative, versioned, structured, and queryable Knowledge Base built for the **Autonomous AI Data Analyst (AADA)** Agent runtime.

## Core Architectural Principle

The system does **NOT** dump the entire knowledge base into the LLM prompt. Instead, the Agent interacts with the knowledge layer via controlled tools:

```text
User Goal
   ↓
Agent (LLM Reasoning Engine)
   ↓
retrieve_knowledge(query, category, ...)  ← Controlled Agent Tool
   ↓
AADA Knowledge Base (Local Dense Vector Store + SQLite Metadata)
   ↓
Relevant Methodologies & Invariants
   ↓
Agent Action Formulation & Execution
```

---

## Capabilities & Architecture

- **Domain Coverage**: 60 curated, versioned knowledge units spanning all 8 required analytics domains:
  1. `data_quality` (DQ-001 to DQ-008): Missingness mechanisms, duplicate detection, boundary validation, type coercions.
  2. `data_cleaning` (DC-001 to DC-008): Robust scaling, median/KNN imputation, winsorization, deduplication merging.
  3. `eda` (EDA-001 to EDA-007): Skewness, bivariate correlations, ANOVA effect sizes, cohort segmentation.
  4. `statistics` (STAT-001 to STAT-008): Central tendency selection, Welch's t-test, Mann-Whitney U, Tukey HSD, bootstrap CIs.
  5. `machine_learning` (ML-001 to ML-008): Gradient boosting selection, Lasso regularization, cross-validation protocols.
  6. `anomaly_detection` (ANOM-001 to ANOM-006): 3-Sigma, Tukey IQR fences, Isolation Forest, Local Outlier Factor.
  7. `visualization` (VIS-001 to VIS-007): Freedman-Diaconis binning, box vs violin plots, hexbin overplotting, pie chart anti-patterns.
  8. `root_cause` & `business_analysis` (RCA-001 to RCA-004, BA-001 to BA-004): Factor trees, rate-mix variance, cohort retention, LTV/CAC.

- **Zero External API Dependency for Retrieval**:
  - Semantic vector similarity utilizes an ultra-fast local TF-IDF subword dense n-gram space with L2-normalized cosine distance.
  - Sub-millisecond search latencies (~1.5ms per query).
  - Preserves external LLM rate limits and API budgets solely for Agent reasoning.

- **Storage & Relational Indexing**:
  - SQLite database (`database/knowledge_base.db`) with indexes on `category`, `subcategory`, `version`, `source`, `status`.
  - Ingestion audit history with duplicate tracking, SHA-256 content hashes, and quality scoring.

- **Standard Agent Tool Contract**:
  - `tools/retrieve_knowledge.py`: Callable python tool function and full JSON Schema export (`RETRIEVE_KNOWLEDGE_TOOL_SPEC`).

- **REST API Endpoints**:
  - FastAPI server (`api/server.py`) providing `/knowledge/search`, `/knowledge/items/{id}`, `/knowledge/categories`, `/health`.

---

## Directory Structure

```text
stage-6-agentic-ai/knowledge-engineer/
├── api/
│   ├── __init__.py
│   └── server.py                 # FastAPI service endpoints
├── database/
│   ├── __init__.py
│   ├── db_manager.py             # SQLite persistence, queries, upserts
│   └── models.py                 # DDL schemas and relational tables
├── ingestion/
│   ├── __init__.py
│   ├── chunker.py                # Decomposes multi-item payloads
│   ├── deduplicator.py           # Exact & near-duplicate prevention
│   ├── ingest.py                 # Master ingestion CLI & pipeline
│   ├── loader.py                 # File discovery & JSON loader
│   ├── parser.py                 # Field normalizer & defaults
│   └── validator.py              # Pydantic validation & quality scoring
├── knowledge_base/               # 60 structured domain JSON files
│   ├── anomaly_detection/
│   ├── business_analysis/
│   ├── data_cleaning/
│   ├── data_quality/
│   ├── eda/
│   ├── machine_learning/
│   ├── root_cause/
│   ├── statistics/
│   └── visualization/
├── retrieval/
│   ├── __init__.py
│   ├── ranker.py                 # Hybrid scoring (semantic + keywords + title match)
│   ├── retriever.py              # High-level coordinator
│   └── vector_store.py           # Zero-network local dense vector index
├── schemas/
│   ├── __init__.py
│   └── knowledge.py              # KnowledgeItem, RetrievalQuery, SearchResult
├── tests/
│   ├── conftest.py
│   ├── test_agent_tool.py        # Tool contract & error handling
│   ├── test_api.py               # FastAPI test client validation
│   ├── test_ingestion.py         # Parsing, chunking, deduplication
│   ├── test_retrieval.py         # Semantic search, filtering, ranker
│   └── test_schemas.py           # Pydantic constraints & score calculation
├── README.md
└── requirements.txt
```

---

## Quickstart

### 1. Run Ingestion Pipeline
```bash
python stage-6-agentic-ai/knowledge-engineer/ingestion/ingest.py
```

### 2. Execute Test Suite
```bash
pytest stage-6-agentic-ai/knowledge-engineer/tests -v
```

### 3. Agent Tool Usage Example
```python
from tools.retrieve_knowledge import retrieve_knowledge

result = retrieve_knowledge(
    query="How should I handle missing values in a highly skewed continuous variable?",
    category="data_cleaning",
    top_k=3
)

print(f"Status: {result['status']}, Found: {result['count']}")
for item in result["knowledge"]:
    print(f"- [{item['knowledge_id']}] {item['title']} (Score: {item['relevance_score']})")
    print(f"  Recommended Methods: {', '.join(item['recommended_methods'])}")
```

### 4. Start REST API Server
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```
