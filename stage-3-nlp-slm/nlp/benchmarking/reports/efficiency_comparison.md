# NLP Efficiency & Computational Benchmark

## 1. Executive Summary

This report analyzes the trade-offs between model accuracy, parameter scale, disk storage, and runtime latency across all evaluated systems on standard commodity CPU hardware (Windows x64).

---

## 2. Efficiency Comparison Matrix

| Model | Parameter Count | Disk Size (Encoder + Heads) | Training Time (s) | Inference Latency (s/doc) | Throughput (docs/sec) | Peak Memory (RAM) | Compute Device |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A (Control)** | **12,156** | **~45 MB** (joblib) | **< 2.0 s** | **0.0128 s** (12.8 ms) | **78.3 docs/s** | ~280 MB | CPU |
| **Train Span Lexicon** | **1,127** | **< 0.1 MB** (json) | **0.35 s** | **0.0054 s** (5.4 ms) | **185.5 docs/s**| ~260 MB | CPU |
| **MiniLM** | 22,721,301 | 90.8 MB + 81 KB | 1,748.8 s | 0.3533 s (353 ms) | 2.83 docs/s | ~750 MB | CPU (4 threads) |
| **MiniLM Hybrid** | 22,721,301 | 90.8 MB + 81 KB | 1,752.5 s | 0.3541 s (354 ms) | 2.82 docs/s | ~750 MB | CPU (4 threads) |

---

## 3. Key Observations & Trade-Offs

1. **Lightweight Footprint**:
   `all-MiniLM-L6-v2` has only **22.7M parameters** and an on-disk footprint of **90.8 MB**. It easily fits into embedded and edge clinical environments without requiring dedicated GPU infrastructure.
2. **CPU Feasibility**:
   At **354 ms per document** on pure CPU, MiniLM Hybrid processes an oncology clinical narrative in approximately one-third of a second. This easily satisfies clinical turn-around requirements (human clinicians review notes on the order of minutes).
3. **Throughput Scaling**:
   Baseline A and the Train Span Lexicon provide ultra-high throughput (78–185 docs/sec), making them ideal for high-throughput batch historical backfills, while MiniLM Hybrid provides superior clinical precision for real-time triage.
4. **SLM Compute Bound**:
   The Qwen2.5-0.5B model (500M parameters) demanded >4 GB RAM during full-sequence encoding, leading to resource throttling. MiniLM represents the "sweet spot" of performance versus compute efficiency.
