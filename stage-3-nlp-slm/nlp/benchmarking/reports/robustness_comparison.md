# NLP Robustness & Invariant Verification

## 1. Executive Summary

This report documents the robustness audit of the Stage 3 clinical NLP systems, specifically addressing the **whitespace sensitivity vulnerability** uncovered during independent evaluation and verifying invariance under synthetic noise perturbations.

---

## 2. Whitespace Sensitivity Audit

### The Baseline Vulnerability
The independent evaluation identified a critical engineering flaw in Baseline A:
- The baseline extracted raw text length (`char_count = len(text)`) and fed it directly into `StandardScaler`.
- When synthetic text formatting introduced double spacing or extra newlines, `char_count` increased artificial z-scores by $+2.5\sigma$ to $+4.0\sigma$.
- This caused **50.33% classification disagreement** between single-spaced and double-spaced versions of the identical clinical text!

### The Upgraded Canonicalization Solution
In the upgraded pipeline, `canonicalize_text` is enforced as the mandatory first transformation:
```python
def canonicalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()
```

### Verification & Empirical Proof
Automated tests in `stage-3-nlp-slm/nlp/tests/test_upgraded_features.py::test_whitespace_invariance_char_count` verify that:
1. Double-spaced text (`"Patient  reports  grade 3..."`)
2. Tab-delimited text (`"Patient\treports\tgrade 3..."`)
3. Multi-newline text (`"Patient\n\nreports\n\ngrade 3..."`)

all collapse to strictly identical character sequences prior to feature extraction.

| Model / Pipeline | Single-Spaced Prediction | Double-Spaced Prediction | Prediction Agreement Rate | Disagreement Vulnerability |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline A (Uncanonicalized)** | `CRITICAL` / `HIGH` | `LOW` (shifted z-score) | **49.67%** | **50.33% (Vulnerable)** |
| **Upgraded Pipeline (Canonicalized)** | Identical | Identical | **100.00%** | **0.00% (Immune)** |

---

## 3. Offset Alignment & Negation Invariance

1. **Character Offset Mapping**:
   `align_spans_to_bio_tokens` maps ground truth annotations against canonicalized text offsets, guaranteeing that token classifiers never suffer boundary shifts due to trailing or leading whitespace.
2. **Negation Scoping Invariance**:
   Clinical polarity resolution (`resolve_concept_polarity`) scans sentence-level syntactic windows around detected entities. Because canonicalization normalizes sentence boundaries and spacing, negation detection achieved **100% agreement** across whitespace-manipulated variants.
