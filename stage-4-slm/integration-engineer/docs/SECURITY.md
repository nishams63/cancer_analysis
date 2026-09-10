# Security & Data Governance Policy: Clinical SLM Integration

**Date**: 2026-09-09  
**Module**: Stage 4 Integration Engineer  
**Classification**: Internal Technical Documentation  

---

## 1. Zero External Network Connectivity Guarantee

- **Local Inference Exclusivity**: All token generation and forward graph evaluation executes 100% locally via `llama.cpp` CPU binaries (`llama.dll`, `ggml-cpu.dll`).
- **No Cloud API Dependencies**: The service makes **zero** network requests to OpenAI, Google Gemini, Anthropic, Hugging Face, or remote model hubs.
- **Offline Enforcement**: Automated test suites (`test_offline_mode.py`, `offline_validation.py`) actively intercept and block non-loopback network calls to ensure zero data exfiltration.

---

## 2. Input Sanitation & Boundary Controls

- **Maximum Note Length**: Hard constraint capped at 10,000 characters to prevent buffer overflow and denial-of-service memory spikes.
- **Schema Validation**: FastAPI and Pydantic enforce typed, structured JSON payloads.
- **Prompt Injection Defense**: Canonical system instructions (`clinical_decision_support_v1`) are strictly isolated from user note content; safety gateway enforces structured 3-field output, intercepting conversational hijack attempts.

---

## 3. PHI / Sensitive Clinical Data Handling

- **Zero Plaintext Storage**: Raw clinical notes are never written to disk or persisted in application logs.
- **Cryptographic Provenance**: Inputs and outputs are logged exclusively as SHA-256 hashes in `artifacts/audit_log.jsonl`.
- **HIPAA De-identification Alignment**: Follows safe-harbor standards by removing direct patient identifiers before audit trail commit.

---

## 4. Model Integrity & Checksum Verification

- **Base Model Verification**: Architecture validated as `Qwen2ForCausalLM`.
- **Adapter Verification**: Best model LoRA weights verified via SHA-256 (`322390ec20e24bb7a672dc5d257819bf7f033148190703da0ca77a409a81f18e`).
- **GGUF Runtime Hashes**: Production binary `merged-model-Q4_K_M.gguf` verified via SHA-256 (`f01cf4112b2cf3b6...`).

---

## 5. Clinical Safety Firewall & Human Review Routing

- **Sequential 6-Gate Inspection**: All generated tokens must pass schema, risk validity, entity grounding, hallucination detection, negation preservation, and action coherence checks.
- **Evidence-Based Confidence Threshold**: Inferences with confidence score $< \tau^* = 0.500$ are automatically flagged with `safety_status: REVIEW` and routed to the human clinician review queue.
- **Disclaimer**: UI and API explicitly display: *"Clinical Decision Support — Human review required when safety checks fail."*
