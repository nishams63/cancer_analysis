# Stage 5: Gen-AI — Data Engineer

Responsible for the versioned, reproducible, evidence-backed reference data layer for the Stage 5 GenAI Synthetic Oncology Stress-Test Engine.

## Structure
- `data/`: Reference distributions, constraint specs, rare combination catalog, and approved oncology evidence corpus
- `src/`: Data cleaning, normalization, distribution estimation, and chunking modules
- `schemas/`: JSON schemas enforcing strict data validation
- `manifests/`: Cryptographic provenance manifests and source registry
- `tests/`: Comprehensive unit test suites (100% pass)
- `verify_done.py`: Programmatic Definition of Done verification script
