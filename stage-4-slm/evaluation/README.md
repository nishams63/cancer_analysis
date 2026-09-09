# Stage 4 — Evaluation Engineer: Independent SLM Benchmarking

## Role Overview
The **Evaluation Engineer** in Stage 4 is responsible for independent, locked-test evaluation of the fine-tuned Small Language Models against Stage 3 baselines and clinical safety standards.

## Objectives & Deliverables
1. **NLG Metrics**:
   - ROUGE-1, ROUGE-2, ROUGE-L, BLEU-4, and BERTScore on held-out Test targets.
2. **Clinical Entity Preservation Metrics**:
   - Extraction F1 of critical entities (Gene, Drug, Dosage, Adverse Event) from SLM generations compared against reference ground truth.
   - Hallucination and unsupported entity rates.
3. **Comparative Baseline Benchmarking**:
   - Compare SLM performance against Stage 3 baseline models (Macro F1 0.7557 on urgency, F1 0.9560 on Critical triage safety).
4. **Clinical Safety & Subgroup Audits**:
   - Risk direction accuracy and action coherence across 26 clinical subgroups.
