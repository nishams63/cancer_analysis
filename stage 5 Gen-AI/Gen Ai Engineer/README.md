# Stage 5: Gen-AI — Gen AI Engineer

Core generation, validation, RAG, and narrative realization engine for the GenAI Synthetic Oncology Stress-Test Engine.

## Architecture
$$\text{Structured Sampler} \to \text{Constraint Validator} \to \text{RAG Retriever} \to \text{LLM Realizer} \to \text{Narrative Validator} \to \text{Counterfactuals} \to \text{Lineage}$$

## Structure
- `configs/`: Master YAML configs and environment specs
- `generation/`: Structured patient scenarios, verified clinical SOAP narratives, counterfactual pairs
- `retrieval/`: TF-IDF vector index and retrieval result logs
- `src/`: Modular components (generation, validation, rag, llm, provenance)
- `pipelines/`: Executable generation and index building pipelines
- `tests/`: 7 comprehensive test suites covering all components
- `verify_done.py`: Programmatic Definition of Done verification (Q1–Q10)
