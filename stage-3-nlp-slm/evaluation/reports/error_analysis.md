# Systematic Error Analysis Report — Stage 3 Clinical NLP

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 3 — Clinical NLP & Baseline Systems Evaluation  
**Scope**: Granular taxonomy of failure modes across Validation ($N=909$) and Locked Test ($N=928$)  
**Evaluator**: Independent Evaluation Engineer  
**Date**: 2026-09-09  
**Status**: COMPLETE & EVIDENCE-BASED  

---

## 1. Executive Summary
This report analyzes the failure modes of the Stage 3 Clinical NLP baseline models across classification, entity extraction, and negation scoping.
The analysis examines:
- 120 misclassified documents in Locked Test (12.93% error rate) and 131 in Validation (14.41% error rate).
- Class confusion pairings and safety-critical false negatives.
- High-confidence classification errors ($\text{confidence} \ge 0.70$).
- Discrepancies between regex-based clinical concept extraction and gold annotations.
- Scope boundary failures in rule-based negation detection.

---

## 2. Classification Failure Taxonomy (Triage Urgency)

### 2.1 Confusion Pairing Breakdown (Locked Test Set)
Of the 120 total misclassifications on the locked test partition:

| Confusion Pair ($\text{True} \to \text{Predicted}$) | Count | Proportion of Errors | Clinical Consequence | Risk Category |
| :--- | :---: | :---: | :--- | :--- |
| **`LOW -> MEDIUM`** | 31 | 25.83% | Over-triage (benign symptom treated as moderate) | Low (Resource Inefficiency) |
| **`MEDIUM -> HIGH`** | 31 | 25.83% | Over-triage (moderate symptom escalated to acute) | Low (Early Clinical Attention) |
| **`HIGH -> MEDIUM`** | 21 | 17.50% | Under-triage (delayed evaluation of acute toxicity)| **Moderate** (Clinical Delay) |
| **`LOW -> HIGH`** | 16 | 13.33% | Severe over-triage | Low (Unnecessary Escalation) |
| **`HIGH -> LOW`** | 12 | 10.00% | Under-triage (acute symptom classified as routine) | **High** (Missed Complication) |
| **`MEDIUM -> LOW`** | 4 | 3.33% | Under-triage | Low-Moderate |
| **`CRITICAL -> MEDIUM`** | 2 | 1.67% | Severe under-triage of deteriorating patient | **Safety Critical** |
| **`MEDIUM -> CRITICAL`** | 2 | 1.67% | False alarm panic | Low (Clinical Review Overrule) |
| **`CRITICAL -> HIGH`** | 1 | 0.83% | Under-triage (escalated, but missing top code) | Low-Moderate |
| **`CRITICAL -> LOW`** | **0** | **0.00%** | Catastrophic complete miss | **Zero Occurrences (PASSED)** |

### 2.2 Safety-Critical False Negative Audit (`CRITICAL` Misses)
Across 100 true `CRITICAL` documents on the Locked Test set, exactly **3 records were not assigned the `CRITICAL` label**:
1. **Case 1 (`DOC-005612`)**: True `CRITICAL`, Predicted `HIGH` ($\text{Confidence} = 0.5421$). Narrative described severe grade 3 transaminitis with moderate fatigue, but lacked the explicit cue phrase `"acute adverse toxicities"` or `"pneumonitis"`.
2. **Case 2 (`DOC-005874`)**: True `CRITICAL`, Predicted `MEDIUM` ($\text{Confidence} = 0.4812$). Complex narrative describing acute renal decline accompanied by historical chemotherapy mentions that diluted high-risk term weighting.
3. **Case 3 (`DOC-005991`)**: True `CRITICAL`, Predicted `MEDIUM` ($\text{Confidence} = 0.5104$). Sparse note where creatinine elevation was documented as a numerical value rather than named nephrotoxicity.

> [!IMPORTANT]
> In all three critical false negative cases, model confidence was low ($\le 0.54$), indicating that probability thresholding or uncertainty flags can successfully route these borderline cases to clinical review.

### 2.3 High-Confidence Errors ($\text{Confidence} \ge 0.70$)
On Locked Test, 38 errors occurred where the model had $\text{confidence} \ge 0.70$ (31.67% of total errors):
- **Dominant Pattern**: `LOW` records misclassified as `MEDIUM` ($\text{Confidence} \approx 0.74$) when mild non-toxic symptoms (e.g. `"mild fatigue"`, `"manageable nausea"`) contained tokens that heavily activated linear regression coefficients.
- **Root Cause**: Linear Bag-of-Words lacks syntactic compositionality; seeing the word `"nausea"` in a short note strongly pushes the score away from `LOW` even when preceded by the adjective `"mild"`.

---

## 3. Toxicity Hazard Classification Error Patterns

Across the 8-class hazard attribution task ($N=928$, 212 errors, 22.84% error rate):

| Error Mode | Frequency | Example Pattern | Root Cause |
| :--- | :---: | :--- | :--- |
| **`NONE -> DERMATOLOGIC`** | 56 | Routine follow-up note classified as `DERMATOLOGIC` | Term `"rash"` mentioned as a standard screening question |
| **`NONE -> RENAL`** | 57 | Baseline note classified as `RENAL` | Laboratory mention of serum creatinine within normal limits |
| **`NONE -> HEPATIC`** | 26 | Baseline consultation classified as `HEPATIC` | Liver enzyme panels ordered as standard monitoring |
| **`NONE -> HEMATOLOGIC`** | 23 | Non-toxic note classified as `HEMATOLOGIC` | Standard CBC count discussed without active cytopenia |

### Clinical Takeaway for Toxicity Hazard
Because the baseline model uses `class_weight='balanced'`, it penalizes misses on rare classes heavily. As a consequence, it acts as an **overly sensitive screening filter**—it flags mentions of lab tests or screening questions as active toxicities, resulting in low precision on rare classes like `DERMATOLOGIC` (6.67%) and `CARDIAC` (20.0%).

---

## 4. Entity Extraction (NER) Error Taxonomy

Comparing 4,815 predicted spans against 3,333 ground truth spans on Locked Test:

### 4.1 Boundary Offset Discrepancies
- **Manifestation**: 441 instances where the entity was identified correctly, but exact character boundaries differed from gold annotations.
- **Example**:
  - *Gold Standard*: `[{"start": 142, "end": 148, "text": "nausea", "label": "ADVERSE_EVENT"}]`
  - *Regex Extractor*: `[{"start": 137, "end": 148, "text": "mild nausea", "label": "ADVERSE_EVENT"}]`
- **Impact**: Accounts for the gap between **Exact Span F1 (0.6439)** and **Relaxed Overlap F1 (0.7756)**.

### 4.2 False Positive Spurious Captures (Dosage Over-Extraction)
- **Manifestation**: 1,174 false positive `DOSAGE` spans (Precision: 39.83%).
- **Example**: Mentions of blood pressure (`120/80 mmHg`) and laboratory concentrations (`1.2 mg/dL`) matched the generic unit regex `r"\b\d+(?:\.\d+)?\s*(?:mg|mmHg|%)\b"`.
- **Recommendation for SLM**: The SLM must distinguish therapeutic drug administration dosages (e.g. `carboplatin AUC 5` or `cisplatin 75 mg/m2`) from clinical vital signs and physiological lab measures.

### 4.3 False Negatives (Missed Mutations & Regimens)
- **Manifestation**: 160 missed `GENE_MUTATION` mentions (Recall: 81.13%) and 134 missed `DRUG_NAME` mentions (Recall: 82.78%).
- **Root Cause**: Rigid dictionary lookup cannot capture novel, misspelled, or compound drug formulations (e.g. `nab-paclitaxel`, `FOLFOX`, `FOLFIRINOX`) that were omitted from the explicit regular expressions.

---

## 5. Negation & Scope Attribution Errors

From the systematic diagnostic suite:
1. **Long-Range Negation Decay**: Concepts located more than 6 words after a pre-negation cue (e.g. `"No evidence of acute toxicities ... stable neuropathy"`) fall outside the fixed token window and revert to `AFFIRMED`.
2. **Pseudo-Negation Collision**: While phrases like `"no change"` are handled, complex clinical expressions such as `"cannot be ruled out"` or `"equivocal for"` remain unmodeled in the rule dictionary.

---

## 6. Recommendations for Downstream SLM Fine-Tuning
1. **Contextual Span Delimitation**: Use token classification heads with pretrained clinical contextual embeddings to resolve pre-nominal modifier boundaries (`manageable mild fatigue` vs `fatigue`).
2. **Distinguishing Dosages from Lab Values**: Fine-tune the SLM on semantic context so that `mmHg` and `mg/dL` are parsed as vitals/labs, restricting `DOSAGE` to therapeutic antineoplastics.
3. **Intermediate Urgency Disambiguation**: Exploit abstractive reasoning in the SLM to distinguish subtle symptom severities between `MEDIUM` and `HIGH` triage tiers.
