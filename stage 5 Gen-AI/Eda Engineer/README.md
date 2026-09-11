# Stage 5: Gen-AI — EDA / Prompt Engineer

Responsible for discovering empirical failure modes, cross-stage discordances, and blind spots (BS01–BS15), and converting them into versioned stress-test scenarios (PROMPT-R01 to PROMPT-R15).

## Structure
- `analysis/`: Empirical pattern mining outputs, blind spot rankings, and 7 Jupyter analysis notebooks
- `prompts/`: Master prompt library, scenario catalog, RAG retrieval intents, and drift rules
- `src/`: Blind spot analyzers, scenario builder, and coverage tracker
- `tests/`: Scenario schema and drift evaluation tests
- `verify_done.py`: Programmatic Definition of Done verification script
