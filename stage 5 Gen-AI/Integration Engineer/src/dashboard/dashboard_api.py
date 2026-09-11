import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ..storage.result_store import ResultStore
from .data_service import DashboardDataService
from .metrics_service import DashboardMetricsService
from ..integration.health_checks import HealthChecker

def create_app(db_path: Optional[str] = None) -> FastAPI:
    app = FastAPI(
        title="Stage 5 GenAI Synthetic Oncology Stress-Test Engine",
        version="1.0.0",
        description="Unified Dashboard & REST API for Stage 5 Stress-Test Orchestration"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    data_service = DashboardDataService(ResultStore(db_path) if db_path else None)
    metrics_service = DashboardMetricsService(data_service)
    health_checker = HealthChecker()

    @app.get("/api/stage5/health")
    def get_health():
        status, checks = health_checker.check_health()
        return {"status": status.value, "checks": checks}

    @app.get("/api/stage5/summary")
    def get_summary(batch_id: Optional[str] = None):
        return metrics_service.get_summary(batch_id)

    @app.get("/api/stage5/batches")
    def get_batches():
        return data_service.get_batches()

    @app.get("/api/stage5/scenarios")
    def get_scenarios(batch_id: Optional[str] = None):
        return data_service.get_scenarios(batch_id)

    @app.get("/api/stage5/scenarios/{scenario_id}")
    def get_scenario_detail(scenario_id: str):
        res = data_service.get_scenario_detail(scenario_id)
        if not res:
            raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")
        return res

    @app.get("/api/stage5/failures")
    def get_failures(batch_id: Optional[str] = None, failure_code: Optional[str] = None):
        return data_service.get_failures(batch_id, failure_code)

    @app.get("/api/stage5/rankings")
    def get_rankings(batch_id: Optional[str] = None):
        return data_service.get_rankings(batch_id)

    @app.get("/api/stage5/counterfactuals")
    def get_counterfactuals(batch_id: Optional[str] = None):
        return data_service.get_counterfactuals(batch_id)

    @app.get("/", response_class=HTMLResponse)
    def index():
        html_path = Path(__file__).resolve().parent.parent.parent / "dashboard" / "index.html"
        if html_path.exists():
            return html_path.read_text(encoding="utf-8")
        return "<h1>Stage 5 GenAI Dashboard API Online</h1><p>Visit /api/stage5/summary for metrics.</p>"

    return app
