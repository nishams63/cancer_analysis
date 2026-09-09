# Clinical NLP Error Analysis & Failure Modes Report — Stage 3

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 3 — Clinical NLP Engineering  
**Version**: `1.0.0`  
**Dataset Evaluated**: Official `VALIDATION` Partition ($N = 909$ documents)  

---

## 1. Executive Overview of Error Profiles
Systematic error analysis was conducted on the 131 misclassified documents in the validation set for the primary triage urgency model (accuracy: 85.59%, 131 errors), as well as false positive and false negative hazard classifications and NER span misses.

### Error Taxonomy Breakdown
1. **Urgency Boundary Ambiguity (MEDIUM vs. LOW & HIGH)**: 57.3% of urgency errors (75 / 131).
2. **Extreme Minority Hazard False Positives**: 28.2% of hazard errors (driven by balanced inverse-class weighting).
3. **Complex Sentence Negation Scope Leaks**: 8.4% of concept errors (multi-clause sentences with nested conjunctions).
4. **Acronym & Numerical Threshold Ambiguity**: 6.1% of errors (borderline laboratory readouts).

---

## 2. Category-Specific Error Analysis

### A. Boundary Misclassification between `MEDIUM` and `LOW` / `HIGH`
- **Empirical Observation**: `MEDIUM` urgency achieved an F1 of 0.4444 (Recall: 46.67%, Precision: 42.42%). 32 `LOW` records were predicted as `MEDIUM`, and 18 `MEDIUM` records were predicted as `LOW`.
- **Root Cause**: Clinical notes presenting with Grade 1–2 toxicities (e.g., *"manageable mild fatigue"*, *"Grade 1 nausea"*) share over 85% lexical overlap with routine baseline follow-ups where fatigue is mentioned as a baseline symptom. Linear bag-of-words models struggle with subtle gradations of fatigue without contextual embedding depth.
- **Privacy-Safe Example**:
  - *Text*: `"Patient reports manageable mild fatigue and slight nausea following cycle 2 Docetaxel. ECOG PS: 1. Labs within normal limits."`
  - *Ground Truth*: `LOW`
  - *Model Prediction*: `MEDIUM`
  - *Failure Mode*: Presence of two symptom words (*"fatigue"*, *"nausea"*) pushed linear decision boundary into `MEDIUM` despite qualifier *"manageable mild"*.

---

### B. Minority Hazard Category Over-Prediction
- **Empirical Observation**: For `DERMATOLOGIC` (support = 4), precision was 6.38% with recall 75.0%. For `RENAL` (support = 24), precision was 16.67% with recall 45.83%.
- **Root Cause**: The inverse-class weighting penalty ($w_{\text{DERMATOLOGIC}} = 28.4$, $w_{\text{NONE}} = 0.16$) aggressively incentivizes the classifier to guess `DERMATOLOGIC` whenever any mild skin keyword appears (e.g., *"dry skin"*, *"mild pruritus"*), even when the note describes an unperturbed systemic consultation.
- **Privacy-Safe Example**:
  - *Text*: `"Follow-up consult. Denies chest pain or fever. Patient notes dry skin on extremities managed with topical lotion. Lungs clear."`
  - *Ground Truth*: `NONE`
  - *Model Prediction*: `DERMATOLOGIC`
  - *Failure Mode*: High penalty weight on dermatologic toxicity over-amplified the topical dry skin mention.

---

### C. Negation Scope Leaks Across Complex Clauses
- **Empirical Observation**: While 100% of simple negations (*"no fever"*, *"denies chest pain"*) are successfully resolved, sentences containing contrasting clauses without standard terminators occasionally leak negation.
- **Privacy-Safe Example**:
  - *Text*: `"Denies shortness of breath, however mild exertion causes rapid pulse."`
  - *Ground Truth*: `shortness of breath` is NEGATED; `rapid pulse` is AFFIRMED.
  - *Failure Mode*: In sentences with comma-separated non-standard conjunctions, the 6-word forward window can occasionally encompass the affirmed symptom.

---

### D. Named Entity Boundary Offsets (NER Span Errors)
- **Empirical Observation**: Rule-based NER achieved 76.70% Span F1.
- **Common Discrepancy**: Compound multi-word entity boundaries (e.g., Ground truth annotated `"manageable mild fatigue"` vs. Extractor identifying `"fatigue"`). Under exact span matching, this constitutes a boundary offset mismatch, although semantic concept detection is correct.

---

## 3. Actionable Recommendations for Downstream SLM Engineer

| Error Category | Baseline Weakness | SLM Architectural Solution |
| :--- | :--- | :--- |
| **MEDIUM Triage Boundary** | Lexical overlap between Grade 1 and Grade 2 toxicity | SLMs provide continuous self-attention across qualifiers (*"mild"* vs. *"severe"*) to cleanly distinguish Grade 1 from Grade 2. |
| **Minority Hazard Over-Guessing** | Rigid linear inverse-class weighting | SLMs conditioned on clinical instructions can perform few-shot in-context reasoning rather than raw loss penalization. |
| **Complex Negation Clauses** | Fixed 6-word window scope assumptions | SLM attention matrices naturally parse hierarchical syntactic trees across clauses. |
| **Compound Entity Offsets** | Token-level dictionary boundaries | SLM instruction fine-tuning can generate structured JSON with standardized concept keys, eliminating character offset brittleness. |
