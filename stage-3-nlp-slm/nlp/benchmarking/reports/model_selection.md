# Stage 3 NLP Model Selection & Promotion Decision

## 1. Final Decision

### **DECISION: "IMPROVED SYSTEM" (PROVISIONAL PROMOTION)**

The development benchmarking evaluation demonstrates that the **MiniLM Hybrid Architecture** (paired with the **Train Span Lexicon** for entity extraction) decisively outperforms the validated baseline (**Baseline A**) across all core clinical tasks on the validation cohort, without introducing unacceptable computational overhead or compromising patient safety.

---

## 2. Gate-by-Gate Evaluation Against Promotion Criteria

| Gate / Criterion | Requirement | Baseline A Result | Upgraded System Result | Gate Status |
| :--- | :--- | :---: | :---: | :---: |
| **Gate 1: Triage Urgency Macro F1** | Meaningful improvement over baseline | 0.7557 | **0.8815 (+12.6 pts)** | **PASSED** |
| **Gate 2: Critical Case Safety Recall** | Must not fall below 93.0% (tolerance: 0.01) | 94.57% | **100.00% (92/92 caught)**| **PASSED** |
| **Gate 3: Toxicity Hazard Macro F1** | Substantial improvement on rare hazards | 0.5214 | **0.9551 (+43.4 pts)** | **PASSED** |
| **Gate 4: Exact NER Span F1** | Maintain or improve span boundary extraction | 0.6008 | **0.7186 (+11.8 pts)** | **PASSED** |
| **Gate 5: Patient Split Isolation** | Zero patient, encounter, or text leakage | 0 leaks | **0 leaks verified** | **PASSED** |
| **Gate 6: Whitespace Invariance** | Immune to synthetic formatting shifts | 50.33% error | **0.00% error (100% agreement)**| **PASSED** |
| **Gate 7: Compute Feasibility** | Latency $\le 500$ ms/doc, Size $\le 500$ MB | 12.8 ms, 45 MB | **354 ms, 90.8 MB** | **PASSED** |
| **Gate 8: Regression Test Suite** | 100% pass rate across test suite | 43 passed | **100% passed (56/56)** | **PASSED** |

---

## 3. Specification of the Promoted System Architecture

The promoted Stage 3 NLP system is configured as follows:

```
Raw Clinical Text
       │
       ▼
[Canonicalization Layer] ──► Collapses excessive whitespace, normalizes padding
       │
       ├───────────────────► [Train Span Lexicon & Regex] ──► Exact Clinical Concepts (F1 = 0.7186)
       │                                                      ├─ GENE_MUTATION (F1 = 0.9486)
       │                                                      ├─ DRUG_NAME (F1 = 0.7390)
       │                                                      ├─ DOSAGE (F1 = 0.4207)
       │                                                      └─ ADVERSE_EVENT (F1 = 0.7516)
       │                                                                   │
       ├───────────────────► [Contextual Polarity Scoping] ◄───────────────┘
       │                     (AFFIRMED, NEGATED, HISTORICAL)
       │                                   │
       │                                   ▼
       │                     [12 Structured Concept Counts]
       │                                   │
       ▼                                   ▼
[Pretrained MiniLM Encoder] ──► 384-dim Dense Embeddings
       │                                   │
       └───────────────────────────────────┤
                                           ▼
                            [396-dim Hybrid Representation]
                                           │
                                           ├─► Balanced Logistic Regression ──► Triage Urgency (Macro F1 = 0.8815)
                                           │                                   (Critical Recall = 100.0%)
                                           │
                                           └─► Balanced Logistic Regression ──► Toxicity Hazard (Macro F1 = 0.9551)
```

---

## 4. Preservation of Control Invariants

1. **Baseline A Remains Frozen**: The existing baseline artifacts in `stage-3-nlp-slm/nlp/artifacts/` remain strictly untouched.
2. **Locked Test Partition Protected**: The locked test dataset (`locked_test.parquet`, $N=928$) was never accessed during model tuning or selection.
3. **Independent Evaluation Readiness**: This upgraded candidate is recommended for independent evaluation by the Evaluation Engineer on the sealed locked-test cohort.
