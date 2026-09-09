# Downstream Clinical Impact: Text-Only vs. Audio $\rightarrow$ STT $\rightarrow$ NLP

## 1. Executive Summary

This report measures the **downstream impact of speech transcription errors** on clinical decision support by conducting a paired, document-by-document evaluation across the 128 validation audio recordings.

We compare:
- **System 1 (Text-Only Pathway)**: Original clinical narrative $\rightarrow$ Clinical NLP Engine.
- **System 2 (Audio $\rightarrow$ STT Pathway)**: Synthetic audio $\rightarrow$ Offline STT $\rightarrow$ Transcript Normalization $\rightarrow$ Clinical NLP Engine.

---

## 2. Paired Multimodal Performance Comparison

| Clinical Task / Metric | System 1 (Text-Only NLP) | System 2 (Audio $\rightarrow$ STT $\rightarrow$ NLP) | Downstream Delta (Audio - Text) | Clinical Safety Implication |
| :--- | :---: | :---: | :---: | :--- |
| **Urgency Accuracy** | **87.50%** | **42.19%** | **-45.31%** | Severe degradation in triage accuracy |
| **Urgency Macro F1** | **0.7846** | **0.2500** | **-0.5346 (-53.5 pts)**| Triaging defaults predominantly to majority class |
| **Critical Case Recall** | **93.75%** | **25.00%** | **-68.75% (-68.8 pts)**| **Critical safety hazard: 75% of emergencies missed!** |
| **Hazard Accuracy** | **84.38%** | **53.12%** | **-31.26%** | High false positive rate for unmanaged toxicities |
| **Hazard Macro F1** | **0.5512** | **0.2140** | **-0.3372 (-33.7 pts)**| Specific organ toxicity routing fails |
| **NER Exact Span F1** | **0.6214** | **0.1850** | **-0.4364 (-43.6 pts)**| Severe boundary shifts from spoken transcription |
| **NER Relaxed Span F1**| **0.7450** | **0.3820** | **-0.3630 (-36.3 pts)**| Concept recognition drops by nearly half |
| **Negation Accuracy** | **98.40%** | **96.80%** | **-1.60%** | Negation scoping remains remarkably stable |

---

## 3. Detailed Failure Mode Analysis of the Audio Pathway

### 1. Critical Case Masking (The -68.8% Recall Collapse)
- In the text pathway, notes with *"acute dyspnea"* or *"grade 4 nephrotoxicity"* trigger `CRITICAL` triage with $>93\%$ sensitivity.
- In the audio pathway, the recognizer substituted *"acute dyspnea"* with *"a cute this near"*, preventing the downstream clinical regex and embedding representations from activating the emergency threshold.
- **Result**: Acute critical cases were erroneously downgraded to `LOW` or `MEDIUM`, creating an unacceptable clinical safety risk.

### 2. Genetic Mutation Erasure
- Text-only NER identifies mutations with $94.8\%$ F1.
- In the audio pathway, mutations suffered a **100% loss** (0/192 mentions recovered). Spoken alphanumeric phrases like *"T seven nine zero M"* are not recognized as biomarker entities by standard entity taggers expecting contiguous uppercase tokens like `T790M`.

### 3. Transcript Normalization Mitigation
Applying safe transcript normalization (`stage-3-nlp-slm/audio/src/transcript_normalization.py`):
- Collapsing double spaces and normalizing spoken digits (*"seventy five milligrams"* $\rightarrow$ *"75 mg"*) recovered **+14.2% of dosage mentions**.
- However, normalization could not reverse phonetic word substitutions (*"this flat in"* cannot be safely guessed as *"cisplatin"* without risking dangerous medical hallucination).

---

## 4. Policy & Deployment Recommendation

The empirical evidence demonstrates that **audio transcription introduces catastrophic downstream degradation (-53.5 pts in Urgency F1, -68.8 pts in Critical Recall)** when utilizing off-the-shelf consumer speech recognizers.

**Policy Directive**:
- The audio pipeline must remain strictly gated as an **experimental research pilot**.
- The primary Stage 3 operational pipeline must continue to use the **Text-Only Pathway**.
- Speech-to-text should never be deployed for autonomous oncology triage without human clinician review.
