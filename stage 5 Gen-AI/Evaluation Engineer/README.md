# Stage 5: Gen-AI — Evaluation Engineer

Responsible for the comprehensive evaluation layer of the **Stage 5 GenAI Synthetic Oncology Stress-Test Engine**, determining whether synthetic scenarios are **realistic, faithful, difficult, and capable of exposing downstream failures in Stages 1–4**.

---

## 🏛️ Architecture & Evaluation Flow

$$\begin{aligned}
\text{Synthetic Scenario} &\longrightarrow \text{Level 1: Realism \& Plausibility} \\
&\longrightarrow \text{Level 2: RAG Quality \& Provenance} \\
&\longrightarrow \text{Level 3: Scenario Fidelity} \\
&\longrightarrow \text{Level 4: Narrative Faithfulness} \\
&\longrightarrow \text{Counterfactual Validation} \\
&\longrightarrow \text{Level 5: Stages 1–4 Stress Testing} \\
&\longrightarrow \text{Cross-Stage Disagreement Matrix} \\
&\longrightarrow \text{F01–F20 Failure Classification} \\
&\longrightarrow \text{Difficulty Scoring (0–100)} \\
&\longrightarrow \text{Impact Scoring (0–100)} \\
&\longrightarrow \text{Failure Clustering \& Wildcard Evidence Dossiers}
\end{aligned}$$

---

## 📁 Directory Structure

```text
Evaluation Engineer/
├── configs/
│   ├── evaluation_config.yaml    # Master evaluation thresholds, tolerance bounds, paths
│   ├── difficulty_weights.yaml   # Weights for System Stress-Test Difficulty Score
│   ├── impact_weights.yaml       # Weights for Failure Impact Score
│   └── failure_taxonomy.yaml     # F01–F20 categories, severities, descriptions
│
├── src/
│   ├── evaluation/
│   │   ├── realism.py            # Level 1: Population similarity (Wasserstein, KS, TVD)
│   │   ├── scenario_plausibility.py # Level 1: Rare-but-allowed physiological plausibility
│   │   ├── rag_quality.py        # Level 2: Concept coverage, retrieval score, provenance
│   │   ├── fidelity.py           # Level 3: Prompt library compliance & forbidden checks
│   │   ├── narrative_faithfulness.py # Level 4: Fact preservation & hallucination detection
│   │   ├── counterfactual.py     # Sensitivity, immutability, instability
│   │   ├── cross_stage.py        # End-to-end multi-stage execution & discordance
│   │   ├── difficulty.py         # System Stress-Test Difficulty Score (0–100)
│   │   ├── impact.py             # Failure Impact Score (0–100)
│   │   ├── failure_classifier.py # Maps errors to F01–F20 taxonomy
│   │   ├── failure_clustering.py # Groups errors into pattern clusters
│   │   └── wildcard_evidence.py  # Ranked evidence dossiers for Integration Engineer
│   │
│   ├── metrics/
│   │   ├── continuous_metrics.py # Wasserstein, KS, quantiles, mean/median/std diffs
│   │   ├── categorical_metrics.py# TVD, Jensen-Shannon divergence, category coverage
│   │   ├── distribution_metrics.py# Joint co-occurrence and conditional checks
│   │   ├── confidence_metrics.py # Confidence spread, high-confidence errors
│   │   └── disagreement_metrics.py # Pairwise stage agreement & discordance rate
│   │
│   ├── adapters/
│   │   ├── stage1_eval_adapter.py # Stage 1 ML tabular risk evaluation
│   │   ├── stage2_eval_adapter.py # Stage 2 DL multimodal pathology evaluation
│   │   ├── stage3_eval_adapter.py # Stage 3 Clinical NLP entity & urgency evaluation
│   │   └── stage4_eval_adapter.py # Stage 4 SLM targeted therapy recommendation evaluation
│   │
│   └── utils/
│       ├── io.py                 # File serializers (JSON, YAML, CSV, Parquet)
│       ├── logging.py            # Structured logger
│       └── validation.py         # Bounds and schema validator
│
├── pipelines/
│   └── run_evaluation.py         # CLI executable: --batch-id, --scenario-id, --counterfactuals
│
├── results/
│   ├── realism/                  # realism_results.json, scenario_plausibility.json
│   ├── rag/                      # rag_quality_results.json
│   ├── fidelity/                 # fidelity_results.json
│   ├── narrative/                # narrative_faithfulness.json
│   ├── counterfactual/           # counterfactual_results.json
│   ├── stages/                   # stage1_results.json, stage2, stage3, stage4
│   ├── failures/                 # failure_report.json, failure_clusters.csv
│   └── rankings/                 # difficulty_scores.csv, impact_scores.csv, wildcard_candidates.csv
│
├── reports/
│   ├── realism_report.md
│   ├── rag_quality_report.md
│   ├── fidelity_report.md
│   ├── narrative_faithfulness_report.md
│   ├── cross_stage_report.md
│   ├── difficulty_report.md
│   ├── failure_report.md
│   └── wildcard_candidate_report.md
│
├── manifests/
│   └── evaluation_manifest.json  # Cryptographic execution manifest and audit trace
│
├── tests/                        # 24 automated unit tests (100% pass)
└── verify_done.py                # Programmatic verification of Definition of Done Q1–Q10
```

---

## 🧪 Testing & Verification

Run the full evaluation test suite:
```bash
pytest "stage 5 Gen-AI/Evaluation Engineer/tests" -v
```

Execute the master evaluation pipeline:
```bash
python "stage 5 Gen-AI/Evaluation Engineer/pipelines/run_evaluation.py" --batch-id BATCH-001
```

Run Definition of Done verification:
```bash
python "stage 5 Gen-AI/Evaluation Engineer/verify_done.py"
```
