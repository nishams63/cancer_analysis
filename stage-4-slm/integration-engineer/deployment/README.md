# Deployment Guide: Clinical SLM Decision Support Service

This directory provides production deployment configurations for both containerized Docker environments and native Windows host startup.

---

## 1. Native Windows Host Startup (Recommended for Local Dev)

Ensure virtual environment is active and dependencies are installed:
```powershell
# 1. Navigate to integration-engineer root
cd stage-4-slm/integration-engineer

# 2. Run local FastAPI server
python src/api.py
```
Access the application:
- Web User Interface: `http://127.0.0.1:8000/`
- Interactive API Docs: `http://127.0.0.1:8000/docs`
- Health Probe: `http://127.0.0.1:8000/health`

CLI Execution:
```powershell
python cli.py --note "Patient on osimertinib 80mg daily with no acute toxicities."
```

---

## 2. Containerized Deployment (Docker Compose)

The container operates in strict offline mode with pre-baked dependencies and local volume mounting.

```bash
# Navigate to deployment directory
cd stage-4-slm/integration-engineer/deployment

# Start containerized service
docker compose up -d
```

### Verification
```bash
curl http://localhost:8000/health
```
