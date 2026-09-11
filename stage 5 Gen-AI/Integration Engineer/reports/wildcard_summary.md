# Stage 5 Integration — Wildcard Candidate Ranking Summary

- **Batch ID**: `RERUN-001`
- **Total Candidates Evaluated**: 2
- **Label**: `CANDIDATE ONLY` (Human clinician and committee review required)

| Rank | Scenario ID | Composite Score | Status | Primary Vulnerability |
| :---: | :--- | :---: | :---: | :--- |
| #1 | **SYN-S001** | 78.2 | `CANDIDATE ONLY` | Dual downstream failure across Stages 1 & 4 |
| #2 | **SYN-S002** | 68.65 | `CANDIDATE ONLY` | Dual downstream failure across Stages 1 & 4 |

## Candidate Explanations

```text
Scenario SYN-S001 ranked #1 (Composite Score: 78.2) [CANDIDATE ONLY]:
- Scenario Plausibility: 90.0% (biologically consistent)
- Prompt Fidelity: 100.0% (all conditions satisfied)
- Narrative Faithfulness: 92.0% (zero critical hallucinations)
- System Stress Difficulty: 66.0 / 100
- Downstream Failure Impact: 76.0 / 100
- Exposed Stages: stage1, stage4
- Failure Codes: F03, F06
- Counterfactual Instability: Preserved
- Reproducibility: Validated deterministic lineage
```

```text
Scenario SYN-S002 ranked #2 (Composite Score: 68.7) [CANDIDATE ONLY]:
- Scenario Plausibility: 90.0% (biologically consistent)
- Prompt Fidelity: 100.0% (all conditions satisfied)
- Narrative Faithfulness: 92.0% (zero critical hallucinations)
- System Stress Difficulty: 67.0 / 100
- Downstream Failure Impact: 77.0 / 100
- Exposed Stages: None
- Failure Codes: None
- Counterfactual Instability: Preserved
- Reproducibility: Validated deterministic lineage
```

