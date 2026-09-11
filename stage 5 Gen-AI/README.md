# Stage 5: Gen-AI — Synthetic Oncology Stress-Test Engine

Comprehensive implementation of **Stage 5** in the clinical AI oncology pipeline, structured across 5 core engineering roles.

---

## 📁 Directory Structure

```text
stage 5 Gen-AI/
├── Data Engineer/
│   ├── data/             # Reference distributions, constraints, evidence corpus
│   ├── src/              # Ingestion, cleaning, normalization, distributions
│   ├── schemas/          # JSON schemas
│   ├── manifests/        # Provenance manifests & source registry
│   ├── tests/            # Data engineering unit tests
│   ├── verify_done.py    # Programmatic DoD verification
│   └── README.md
│
├── Eda Engineer/
│   ├── analysis/         # Blind spots (BS01–BS15), notebooks, priorities
│   ├── prompts/          # Prompt library (PROMPT-R01 to PROMPT-R15), drift rules
│   ├── src/              # Pattern mining & scenario builder
│   ├── reports/          # Blind spot & failure mode reports
│   ├── tests/            # EDA & prompt compliance tests
│   ├── verify_done.py    # Programmatic DoD verification
│   └── README.md
│
├── Gen Ai Engineer/
│   ├── configs/          # GenAI, RAG, and LLM configuration specs
│   ├── generation/       # Factual structured scenarios, verified narratives, counterfactuals
│   ├── retrieval/        # Evidence vector index & retrieval logs
│   ├── src/              # Sampler, validators, RAG retriever, LLM client, provenance
│   ├── pipelines/        # Executable generation & RAG pipelines
│   ├── tests/            # Comprehensive GenAI test suites (7 suites)
│   ├── verify_done.py    # Programmatic DoD verification (Q1–Q10)
│   └── README.md
│
├── Evaluation Engineer/
│   ├── src/              # Stress-test evaluation harness & sensitivity drop metrics
│   ├── benchmarks/       # Vulnerability benchmark definitions
│   ├── reports/          # Evaluation framework specifications
│   ├── tests/            # Evaluation unit tests
│   └── README.md
│
└── Integration Engineer/
    ├── src/              # Microservice & pipeline integration
    ├── deployment/       # Production Dockerfile & configs
    ├── docs/             # API specifications
    ├── tests/            # Integration service tests
    └── README.md
```

---

## 👥 Engineering Roles & Core Deliverables

| Role | Primary Responsibility | Key Deliverables | Status |
| :--- | :--- | :--- | :---: |
| **Data Engineer** | Reference distributions, constraints, approved evidence | Distributions, schemas, evidence chunks, source registry | **VERIFIED (100%)** |
| **Eda Engineer** | Blind-spot discovery & scenario specification | BS01–BS15, PROMPT-R01–R15, drift rules, 7 notebooks | **VERIFIED (100%)** |
| **Gen Ai Engineer** | Structured sampler, constraint validator, RAG, NVIDIA LLM, counterfactuals | 20 scenarios, 20 SOAP notes, 20 counterfactuals, manifests | **VERIFIED (100%)** |
| **Evaluation Engineer** | Downstream model stress testing & vulnerability benchmarking | Sensitivity drop metrics, benchmark suites | **INITIALIZED** |
| **Integration Engineer** | End-to-end service integration & deployment | REST service, Dockerfile, API documentation | **INITIALIZED** |
