# Additive Stage 3 benchmarks

The validated baseline, its artifacts, upstream data, and old reports are read-only.
New results live here. `src/protocol.py` refuses locked-test loading and checks
train/validation patient, encounter, document, canonical-text and temporal isolation.

Run `python src/run_baseline.py` for fresh validation-only baseline measurements.
Run `python src/run_competitors.py --model minilm` for a cached pretrained encoder
with independently trained document and BIO token heads. `--online` permits checkpoint
retrieval. Available candidate names are in `configs/benchmark.json`.

The compute-controlled protocol freezes pretrained encoders. It is not full encoder
fine-tuning. All document heads use class-weighted logistic regression. NER uses
supervised per-token BIO classification, with validation early stopping. Raw source
text is retained for offset alignment. Hybrid heads combine contextual representations
with the existing structured feature extractor. Unavailable checkpoints produce
NOT RUN records with exact errors, never substitute random encoders.

No promotion is final until robustness, clinical-recall safeguards, efficiency and
reproducibility are measured. A better validation score alone does not authorize
replacing the baseline. Cached frozen models remain intact.
