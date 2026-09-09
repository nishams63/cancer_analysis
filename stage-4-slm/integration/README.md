# Stage 4 — Integration Engineer: SLM Clinical Decision Support Service

## Role Overview
The **Integration Engineer** in Stage 4 is responsible for packaging and deploying the fine-tuned Small Language Model into a high-performance clinical decision support service.

## Objectives & Deliverables
1. **Inference Engine**:
   - vLLM / ONNX / TensorRT-LLM optimized serving for low-latency batch and streaming inference.
2. **REST API Service**:
   - Production-ready FastAPI endpoints (`/health`, `/generate/risk`, `/generate/action`, `/batch`).
   - Pydantic request and response schemas with strict input sanitization.
3. **Safety Guardrails & Confidence Scoring**:
   - Output validation filters preventing toxic or unsafe recommendations.
   - Uncertainty estimation and fallbacks to Stage 3 classical baseline classifiers.
