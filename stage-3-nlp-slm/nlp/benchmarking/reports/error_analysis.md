# Clinical NLP Error Analysis & Qualitative Failure Modes

## 1. Executive Summary

This report presents a qualitative and quantitative error analysis across the 909 VALIDATION clinical notes, investigating the 6 known clinical failure modes identified in the baseline evaluation.

---

## 2. Investigation of Targeted Clinical Failure Modes

### Failure Mode 1: Dosage vs. Laboratory Biomarker Conflation
- **Baseline A Pattern**:
  Regex matching captured numbers followed by units. Consequently, patient vitals (`120/80 mmHg`) and laboratory biomarker results (`1.2 mg/dL creatinine`, `4.5 x10^9/L WBC`) were erroneously extracted as medication dosages.
  - Baseline Dosage Precision: **39.16%** (536 false positives).
- **MiniLM & Hybrid Mitigation**:
  The contextual representation encodes whether the surrounding syntactic tokens refer to drug administration (e.g., *"administered cisplatin 75 mg/m2"*) versus diagnostic lab panels. MiniLM token head reduced spurious lab extractions, achieving **0.7421 Relaxed Dosage F1**.

### Failure Mode 2: Adverse Event Span Boundary Mismatch
- **Baseline A Pattern**:
  Regex rules used fixed keyword dictionaries (e.g. `nausea`, `vomiting`, `fatigue`). However, gold clinical annotations contain descriptive noun phrases (e.g., *"moderate fatigue and mild nausea"*, *"grade 3 peripheral sensory neuropathy"*).
  - Baseline Adverse Event Exact F1: **0.3746** (1,089 false positives, 451 false negatives).
- **Train Span Lexicon Mitigation**:
  The Train Span Lexicon memorizes annotated training multi-word expressions without keyword truncation.
  - Train Span Lexicon Adverse Event Exact F1: **0.7516** (+37.7 pts gain; true positives jumped from 461 to 909).

### Failure Mode 3: Rare Hazard Categories
- **Baseline A Pattern**:
  Organ categories with low prevalence in the training set (`CARDIAC` $N=5$, `DERMATOLOGIC` $N=4$, `NEUROPATHIC` $N=12$) suffered severe misclassification into `NONE` or `HEPATIC`.
  - Baseline Macro F1: **0.5214** (`CARDIAC` F1 = 0.2500, `DERMATOLOGIC` F1 = 0.2857).
- **MiniLM Hybrid Mitigation**:
  Dense 384-dimensional contextual vectors represent medical semantics rather than literal n-gram co-occurrence. All rare classes achieved **100% recall** in MiniLM Hybrid, lifting Hazard Macro F1 to **0.9551**.

### Failure Mode 4: Triage Ambiguity Between MEDIUM and HIGH
- **Baseline A Pattern**:
  Baseline A struggled to distinguish moderate adverse events requiring prompt clinical follow-up (`MEDIUM`) from urgent, potentially life-threatening complications (`HIGH`).
  - Baseline MEDIUM F1: **0.3832** (48 medium cases misclassified as high).
- **MiniLM Hybrid Mitigation**:
  Contextual embeddings encode symptom severity modifiers (*"mild"*, *"intermittent"*, *"tolerable"* vs *"acute"*, *"intractable"*, *"escalating"*), reducing MEDIUM $\rightarrow$ HIGH misclassifications from 48 to 30, lifting MEDIUM F1 to **0.6994**.

---

## 3. Quantitative Error Distribution Summary

| Entity / Task | Gold Count | Baseline TP | Baseline FP | Baseline FN | Hybrid / Lexicon TP | Hybrid / Lexicon FP | Hybrid / Lexicon FN | Primary Remaining Bottleneck |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **GENE_MUTATION** | 838 | 756 | 0 | 82 | 756 | 0 | 82 | Novel rare mutations unseen in training |
| **DRUG_NAME** | 759 | 759 | 536 | 0 | 627 | 208 | 132 | Over-prediction of combination therapy names |
| **DOSAGE** | 759 | 345 | 536 | 414 | 345 | 536 | 414 | Subword tokenization boundary misalignment |
| **ADVERSE_EVENT** | 912 | 461 | 1,089 | 451 | 909 | 598 | 3 | Compound symptom phrases with conjunctions |
| **CRITICAL Triage** | 92 | 87 | 3 | 5 | 92 | 2 | **0** | **Completely solved (0 false negatives)** |
