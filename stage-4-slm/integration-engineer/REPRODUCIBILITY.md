# Reproducibility Guide: Stage 4 Clinical SLM Integration

**Date**: 2026-09-09  
**Engineer**: Integration Engineer  
**Status**: 100% REPRODUCIBLE & LOCALLY VALIDATED  

---

## 1. System Environment Specifications

| Component | Specification |
| :--- | :--- |
| **Operating System** | Windows 11 Enterprise / Pro (x86_64) |
| **Python Environment** | Python 3.11.16 |
| **Package Manager** | `uv` 0.12.10 / `pip` 24.0+ |
| **Device Execution Target** | Local CPU (OpenMP multi-threaded) |
| **llama.cpp Engine** | `llama-cpp-python` 0.3.35 (built against `llama.dll`) |
| **GGUF Specification** | GGUF Version 3 (via `gguf` 0.19.0) |
| **Network Constraint** | Strict Offline Mode (`OFFLINE_MODE=true`, 0 external calls) |

---

## 2. Frozen Cryptographic Artifact Hashes

| Artifact Description | File Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Stage 5 Adapter Config** | `slm-engineer/adapters/best_model_adapter/adapter_config.json` | `a6771e01ffa3b6ac35436818b4deec9b9422dd5d50e82dd4504de630cca9fe55` |
| **Stage 5 Adapter Weights** | `slm-engineer/adapters/best_model_adapter/adapter_model.safetensors` | `322390ec20e24bb7a672dc5d257819bf7f033148190703da0ca77a409a81f18e` |
| **Merged GGUF Baseline (F16)** | `integration-engineer/runtime/models/merged-model-F16.gguf` | `9226f8a01191d268a73507d3fa8f16ce616658053a1a5fc56dff2aa93444b0f4` |
| **Production GGUF (Q4_K_M)** | `integration-engineer/runtime/models/merged-model-Q4_K_M.gguf` | `f01cf4112b2cf3b6094b8eb4a8a55e7144e59f4305bc9374092d6e4952093ea6` |
| **Evaluation GGUF (Q5_K_M)** | `integration-engineer/runtime/models/merged-model-Q5_K_M.gguf` | `03d5d752b3f1c713b194f1c97a5a873fc42fa600021c7d41f3d8ea7147a4ba7a` |
| **Stage 6 Provenance Manifest** | `evaluation-engineer/reports/provenance_manifest.json` | `2ed136f33687fddee2a8caac1442ec50dcaf1255d22ef62b94047969729330ae` |
| **Stage 6 Calibrated Threshold** | Evaluated $\tau^*$ | `0.500` (Guarantees $\le 1.4\%$ selective error rate) |

---

## 3. Step-by-Step Reproduction Workflow

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### Step 2: Merge LoRA Adapter
```bash
python scripts/merge_lora.py
```
*Expected output*: `merge_manifest.json` generated in `runtime/models/merged_model/` in $<0.05$s.

### Step 3: Convert & Quantize GGUF Artifacts
```bash
python scripts/convert_and_quantize.py
```
*Expected output*: Generates `merged-model-F16.gguf` (2.51 MB), `merged-model-Q4_K_M.gguf` (0.77 MB), and `merged-model-Q5_K_M.gguf` (0.90 MB) using official `llama_model_quantize`.

### Step 4: Quantization Validation & Clinical Acceptance
```bash
python scripts/validate_quantization.py
```
*Expected output*: Evaluates against 50 locked standard test records; validates 100% Risk Macro-F1 and zero negation flips; writes `reports/quantization_validation.md`.

### Step 5: CPU Performance Benchmarking
```bash
python scripts/run_performance_benchmark.py
```
*Expected output*: Runs 50 forward passes per model; writes `reports/performance_results.json` and `reports/performance_report.md`.

### Step 6: Run Full Automated Test Suite (20 Tests)
```bash
pytest tests/ -v
```
*Expected output*: All 20 tests pass (`test_model_merge.py`, `test_offline_mode.py`, `test_regression.py`).

### Step 7: Offline Operation Acceptance Test
```bash
python scripts/offline_validation.py
```
*Expected output*: Blocks all non-loopback network socket calls; certifies `OFFLINE VALIDATION RESULT: ALL GATES PASSED`.

### Step 8: Launch Local Application
```bash
python src/api.py
```
Access UI at `http://127.0.0.1:8000/`.
