#!/usr/bin/env python3
"""
Launcher for Stage 5 GenAI Synthetic Oncology Stress-Test Dashboard.
Starts the FastAPI server with embedded interactive UI at http://127.0.0.1:8085
"""
import sys
from pathlib import Path

# Add Stage 5 Integration Engineer directory to python path
base_dir = Path(__file__).resolve().parent / "stage 5 Gen-AI" / "Integration Engineer"
sys.path.insert(0, str(base_dir))

import uvicorn
from src.dashboard.dashboard_api import app

if __name__ == "__main__":
    port = 8085
    host = "127.0.0.1"
    print(f"\n========================================================")
    print(f" Starting Stage 5 GenAI Dashboard at http://{host}:{port}")
    print(f"========================================================\n")
    uvicorn.run(app, host=host, port=port, log_level="info")
