# Stage 4: Small Language Model (SLM) Fine-Tuning & Clinical Decision Support

This directory contains the complete end-to-end implementation of **Stage 4** for the Oncology Precision Medicine project.

---

## 📁 Directory Structure

```text
stage-4-slm/
├── data-engineering/    # SLM fine-tuning dataset pipeline, entity gate, provenance, splits
├── eda/                 # Instruction-tuning dataset EDA, token lengths, vocab, distributions
├── slm/                 # Small Language Model fine-tuning (LoRA / QLoRA, checkpointing)
├── evaluation/          # Benchmark evaluation, ROUGE, BLEU, clinical entity preservation metrics
└── integration/         # Serving API / inference pipeline for SLM decision support
```

---

## 👥 Engineering Modules & Responsibilities

| Module | Engineering Role | Key Responsibility & Deliverables |
|:---|:---|:---|
| [`data-engineering/`](data-engineering/) | **Data Engineer** | Stage 3 ingestion, clinical entity normalization, target drafting with provenance logging, entity preservation quality gate, rejection circuit breaker, patient-level 70/15/15 split, multi-dimensional leakage audit (`patient_leakage = 0`), and `slm_finetune_dataset_v1.parquet`. |
| [`eda/`](eda/) | **EDA Engineer** | Prompt/completion length distributions, tokenization profiling, vocabulary coverage, and partition uniformity analysis across splits. |
| [`slm/`](slm/) | **SLM Engineer** | Parameter-efficient fine-tuning (PEFT / LoRA / QLoRA) of domain-adapted clinical SLMs on validated instruction pairs. |
| [`evaluation/`](evaluation/) | **Evaluation Engineer** | Independent locked-test benchmarking, NLG metrics (ROUGE, BLEU, BERTScore), entity span extraction comparison, and subgroup safety analysis. |
| [`integration/`](integration/) | **Integration Engineer** | FastAPI REST inference service, vLLM / ONNX optimization, safety guardrails, and clinical decision support integration. |

---

## 🧪 Testing

To run the full automated test suite for Stage 4 Data Engineering (23 unit tests):
```bash
pytest stage-4-slm/data-engineering/tests/ -v
```

To run the end-to-end dataset pipeline:
```bash
python stage-4-slm/data-engineering/src/pipeline.py
```
