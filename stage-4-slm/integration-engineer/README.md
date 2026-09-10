# Stage 4: Integration Engineer — Production-Grade Offline Clinical SLM Application

This directory contains the complete implementation of the **Integration Engineer** for Stage 4 of the Oncology Precision Medicine project.

It delivers a real, fully offline clinical Small Language Model (SLM) decision-support application that packages the frozen winning model (`Qwen2.5-1.5B-Instruct` + `Entity-Filtered LoRA`), runs locally with **zero external API calls**, exposes a production-style FastAPI service, enforces the Stage 6 clinical safety firewall, provides an offline web UI and CLI, records complete cryptographic provenance, and proves reproducible offline operation.

---

## 📁 Directory Structure

```text
integration-engineer/
├── src/
│   ├── api.py                  # Production FastAPI service with CORS & static UI mounting
│   ├── inference_service.py    # Pipeline orchestrator (Prompt -> GGUF -> Parser -> Safety -> Audit)
│   ├── model_manager.py        # Local llama.cpp CPU runtime loader & inference manager
│   ├── prompt_builder.py       # Canonical prompt formatter (template v1.0.0, input sanitizer)
│   ├── output_parser.py        # Structured field parser & voice-ready 3-line spoken summary
│   ├── safety_gateway.py       # 6-gate clinical safety firewall with calibrated threshold
│   ├── provenance.py           # SHA-256 cryptographic provenance generator
│   ├── audit_logger.py         # Append-only immutable JSONL transaction logger
│   ├── health.py               # Deep diagnostic health probe
│   └── config.py               # YAML & frozen threshold configuration loader
│
├── frontend/
│   ├── index.html              # Responsive clinical decision support dashboard
│   ├── app.js                  # 100% offline client-side UI logic & speech synthesis
│   └── styles.css              # Accessible clinical UI styling with safety badges
│
├── runtime/
│   ├── config/
│   │   └── runtime.yaml        # Offline CPU runtime parameters
│   └── models/
│       ├── merged_model/       # Deterministic merged model export definitions
│       ├── merged-model-F16.gguf    # Baseline unquantized GGUF binary (2.51 MB)
│       ├── merged-model-Q4_K_M.gguf # Certified production quantized GGUF (0.77 MB)
│       ├── merged-model-Q5_K_M.gguf # High-precision quantized GGUF (0.90 MB)
│       └── quantization_manifest.json # Hashes, tool versions, and quant metadata
│
├── scripts/
│   ├── merge_lora.py           # LoRA adapter parameter merger (no retraining)
│   ├── convert_and_quantize.py # Official GGUF builder and native llama.cpp quantizer
│   ├── validate_quantization.py# Benchmark validator comparing F16 vs Q4_K_M vs Q5_K_M
│   ├── run_performance_benchmark.py # CPU load time, TTFT, P50/P95, throughput benchmark
│   └── offline_validation.py   # Standalone acceptance script enforcing network block
│
├── tests/
│   ├── conftest.py             # Test isolation & sys.path configuration
│   ├── test_model_merge.py     # LoRA merge & scaling factor unit tests (3 tests)
│   ├── test_offline_mode.py    # Automated network-block acceptance tests (2 tests)
│   └── test_regression.py      # Normal, clinical safety & adversarial tests (15 tests)
│
├── artifacts/
│   ├── model_manifest.json     # Machine-readable frozen hashes & metadata
│   └── audit_log.jsonl         # Append-only immutable transaction audit trail
│
├── reports/
│   ├── quantization_validation.md # Quantization trade-off & acceptance report
│   ├── performance_report.md      # CPU load time, TTFT, throughput, RAM report
│   ├── performance_results.json   # Machine-readable performance metrics
│   └── integration_report.md      # Final validation & deployment acceptance report
│
├── docs/
│   ├── PRE_IMPLEMENTATION_AUDIT.md # Initial baseline audit before implementation
│   └── SECURITY.md             # Data privacy, HIPAA compliance, PHI hashing
│
├── deployment/
│   ├── Dockerfile              # Strict offline container definition
│   ├── docker-compose.yml      # Local container orchestration
│   └── README.md               # Container & host startup documentation
│
├── cli.py                      # Offline command-line interface for terminal inference
├── requirements.txt            # Minimal offline Python dependencies
├── Makefile                    # Automation shortcuts (merge, quantize, test, serve)
├── REPRODUCIBILITY.md          # Complete end-to-end reproduction guide
└── README.md                   # This document
```

---

## 🚀 Quick Start (Native Windows)

### 1. Launch FastAPI Server
```powershell
python src/api.py
```
- Open browser at `http://127.0.0.1:8000/` to use the offline decision support dashboard.
- Interactive API documentation available at `http://127.0.0.1:8000/docs`.

### 2. Terminal CLI Inference
```powershell
python cli.py --note "Patient on osimertinib 80mg daily with no acute toxicities."
```

---

## 🧪 Automated Testing (20 / 20 Passed)

Run the full integration test suite:
```bash
pytest tests/ -v
```

Run standalone offline validation:
```bash
python scripts/offline_validation.py
```

---

## 🛡️ Clinical Safety Disclaimer

> [!IMPORTANT]
> **Clinical Decision Support — Human review required when safety checks fail.**  
> This software is an experimental prototype designed to assist oncologists with clinical note summarization and structured risk extraction. It does not provide medical diagnoses or make final clinical treatment decisions. All outputs must be independently reviewed and verified by a licensed oncologist.
