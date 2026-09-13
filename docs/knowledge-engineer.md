# Knowledge Engineer Specification: AADA Stage 06

**Autonomous AI Data Analyst (AADA) — Stage 06: Agentic AI**  
*Role*: Senior Knowledge Engineer  
*Version*: 1.0.0  
*Status*: Active & Authoritative  

---

## 1. System Mission & Scope Boundaries

The **Knowledge Engineer** is strictly responsible for:
1. Authoring, organizing, and versioning high-quality data analysis methodologies, statistical rules, machine learning guidelines, and business diagnostics.
2. Building an ingestion engine with validation, normalization, and duplicate detection.
3. Managing persistent metadata storage (SQLite) and local semantic vector indexing.
4. Implementing a hybrid retrieval engine (semantic similarity + keyword matching + metadata filtering).
5. Exposing clean, controlled Agent tools (`retrieve_knowledge()`) and REST APIs.

### Scope Invariants:
- The Knowledge Engineer does **NOT** build the Agent runtime, workflow orchestrator, evaluation engine, or UI.
- Stages 1 through 5 remain untouched (zero diff).
- All retrieval is 100% local, free, deterministic, and independent of external LLM API rate limits.

---

## 2. Structured Knowledge Item Schema

Every atomic knowledge record conforms to the Pydantic specification defined in `schemas/knowledge.py`:

```python
class KnowledgeItem(BaseModel):
    knowledge_id: str                   # e.g., "DQ-001", "STAT-004", "ANOM-003"
    title: str                          # Descriptive, concise title
    category: KnowledgeCategory         # One of 8 core domains
    subcategory: str                    # e.g., "imputation", "hypothesis_testing"
    description: str                    # Clear, authoritative summary
    conditions: List[str]               # Specific prerequisite criteria
    recommended_methods: List[str]      # Concrete, valid algorithms or techniques
    selection_rules: List[str]          # Decision rules (e.g., skewed vs Gaussian)
    limitations: List[str]              # Risks, edge cases, trade-offs
    source: str                         # Standard reference (e.g. "AADA Internal Methodology")
    version: str                        # Semantic version (e.g. "1.0")
    effective_date: str                 # YYYY-MM-DD
    status: str                         # "active", "deprecated", or "draft"
    created_at: Optional[str]           # ISO-8601 timestamp
    updated_at: Optional[str]           # ISO-8601 timestamp
    quality_score: Optional[float]      # Automated completeness metric (0.0 to 1.0)
```

---

## 3. The 8 Knowledge Domains & Master Directory

| Domain Code | Category | File Path | Item Count | Key Topics |
| :--- | :--- | :--- | :--- | :--- |
| **DQ** | `data_quality` | `knowledge_base/data_quality/` | 8 | MCAR/MAR/MNAR missingness, duplicate keys, range bounds, string casing, type coercions, temporal formats, cross-field rules. |
| **DC** | `data_cleaning` | `knowledge_base/data_cleaning/` | 8 | Median/KNN/MICE imputation, winsorization, survivorship merging, NFKD normalization, UTC parsing, RobustScaler, target encoding. |
| **EDA** | `eda` | `knowledge_base/eda/` | 7 | Five-number summary, Shannon entropy, Pearson vs Spearman, ANOVA Eta-squared, Chi-Square Cramer's V, STL decomposition, Simpson's Paradox. |
| **STAT** | `statistics` | `knowledge_base/statistics/` | 8 | Skewness central tendency, IQR/MAD dispersion, empirical percentiles, Welch's t-test, Mann-Whitney U, Tukey HSD, bootstrap CIs, Cohen's d. |
| **ML** | `machine_learning` | `knowledge_base/machine_learning/` | 8 | LightGBM/XGBoost tabular, Ridge/Lasso shrinkage, K-Means vs DBSCAN, PCA/UMAP, permutation importance, TimeSeriesSplit, class weights, PR-AUC. |
| **ANOM** | `anomaly_detection` | `knowledge_base/anomaly_detection/` | 6 | 3-Sigma Gaussian rule, Tukey IQR fences, Isolation Forest, Local Outlier Factor, Mahalanobis MCD distance, STL time-series anomalies. |
| **VIS** | `visualization` | `knowledge_base/visualization/` | 7 | Freedman-Diaconis binning, box vs violin plots, hexbin overplotting, bar vs line chart rules, triangular heatmaps, pie chart anti-patterns, small multiples. |
| **RCA / BA** | `root_cause` & `business_analysis` | `knowledge_base/root_cause/` & `knowledge_base/business_analysis/` | 8 | DuPont factor trees, rate-mix variance decomposition, cohort gap analysis, causal DAG safeguards, cohort retention matrices, BG-NBD LTV, funnel drop-offs, ABC Pareto analysis. |

**Total Curated Units**: 60 Knowledge Items across all 8 domains.

---

## 4. Ingestion & Quality Assurance Pipeline

The ingestion pipeline (`ingestion/ingest.py`) implements a five-stage processing sequence:
1. **Loader (`loader.py`)**: Traverses `knowledge_base/` recursively and loads raw JSON files.
2. **Chunker (`chunker.py`)**: Flattens nested or composite knowledge entries into atomic units.
3. **Parser (`parser.py`)**: Trims whitespace, standardizes case, and injects default metadata.
4. **Validator (`validator.py`)**: Validates against Pydantic schema, enforces ID formatting (`[A-Z]{2,4}-\d{3}`), and computes an automated quality score:
   $$	ext{Quality Score} = 0.15 \cdot 	ext{Title} + 0.25 \cdot 	ext{Description} + 0.15 \cdot 	ext{Conditions} + 0.15 \cdot 	ext{Methods} + 0.15 \cdot 	ext{Rules} + 0.15 \cdot 	ext{Limitations}$$
5. **Deduplicator (`deduplicator.py`)**: Identifies duplicate IDs and identical content SHA-256 hashes.
6. **Persistence & Indexing**: Upserts valid items into SQLite (`database/knowledge_base.db`) and fits the local TF-IDF dense vector store (`retrieval/vector_store.py`).

---

## 5. Hybrid Retrieval & Relevance Ranking

Queries submitted by agents are evaluated through a hybrid scoring pipeline:

1. **Candidate Filtering**: SQLite performs indexed pre-filtering on `category`, `subcategory`, `source`, `version`, and `status`.
2. **Semantic Similarity**: The query is mapped into the vector space, computing cosine similarity ($S_{	ext{sem}} \in [0, 1]$) against all candidate knowledge units.
3. **Keyword Matching**: Token overlap in titles, recommended methods, and selection rules is computed ($S_{	ext{kw}} \in [0, 1]$).
4. **Hybrid Combination**:
   $$	ext{Final Score} = 0.65 	imes S_{	ext{sem}} + 0.35 	imes S_{	ext{kw}} + 	ext{Exact Title Boost (0.15)}$$
5. **Ranking**: Candidates are sorted by Final Score descending and returned with 1-based ranks.

---

## 6. Agent Tool Contract: `retrieve_knowledge`

### Python Signature
```python
def retrieve_knowledge(
    query: str,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    source: Optional[str] = None,
    version: Optional[str] = None,
    top_k: int = 5,
    min_score: float = 0.0
) -> Dict[str, Any]:
    ...
```

### Response Payload Structure
```json
{
  "status": "success",
  "query": "How should I handle missing values in a highly skewed continuous variable?",
  "count": 3,
  "latency_ms": 1.48,
  "filters_applied": {
    "category": "data_cleaning",
    "subcategory": null,
    "source": null,
    "version": null,
    "status": "active"
  },
  "knowledge": [
    {
      "knowledge_id": "DC-001",
      "title": "Numerical Missing Value Imputation Strategy",
      "category": "data_cleaning",
      "subcategory": "imputation",
      "relevance_score": 0.8124,
      "semantic_similarity": 0.7114,
      "keyword_overlap": 1.0,
      "description": "Principled algorithm selection for replacing missing numerical values while preserving distribution shape and variance.",
      "conditions": [
        "Missing numerical values validated as MCAR or MAR",
        "Missingness percentage is between 0.5% and 25%",
        "Downstream models or analytics cannot process NaN values directly"
      ],
      "recommended_methods": [
        "Median imputation for skewed or outlier-heavy distributions",
        "Mean imputation for verified Gaussian / symmetrical distributions",
        "K-Nearest Neighbors (KNN) imputation for multi-correlated features",
        "Iterative chained equations (MICE) for high-stakes clinical or economic datasets"
      ],
      "selection_rules": [
        "If skewness magnitude > 1.0 or heavy outliers exist, select Median imputation over Mean",
        "If feature correlation with other features > 0.60, select KNN or MICE imputation over univariate median",
        "If missingness > 30%, do not impute with mean/median; consider dropping or model-based imputation"
      ],
      "limitations": [
        "Mean and median imputation artificially underestimate feature standard deviation and compress error bars",
        "KNN imputation is computationally expensive O(N_missing * N_train * D) on large datasets"
      ],
      "source": "AADA Internal Methodology",
      "version": "1.0"
    }
  ]
}
```

---

## 7. Verification & Test Suite Results

The comprehensive test suite in `tests/` verifies schema constraints, file parsing, deduplication, semantic ranking, and API endpoints:

- **Command**: `pytest stage-6-agentic-ai/knowledge-engineer/tests -v`
- **Result**: **31 passed in 0.51s (100% pass rate)**.
- **Coverage**:
  - `test_schemas.py`: 7 tests passing (Pydantic validation, ID constraints, score calculations).
  - `test_ingestion.py`: 6 tests passing (loaders, normalizers, deduplication, end-to-end ingestion).
  - `test_retrieval.py`: 7 tests passing (vector fit/search, hybrid ranker, semantic queries, metadata filters).
  - `test_agent_tool.py`: 3 tests passing (tool execution, error handling, function call schema).
  - `test_api.py`: 8 tests passing (FastAPI test client covering all REST endpoints).
