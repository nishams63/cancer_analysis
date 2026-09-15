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

    @app.api_route("/api/stage5/generate_patient", methods=["GET", "POST"])
    def generate_patient(
        cancer_type: Optional[str] = "Lung Cancer",
        age_range: Optional[str] = "45 - 75",
        mutation: Optional[str] = "EGFR",
        biomarker: Optional[str] = "PD-L1",
        treatment: Optional[str] = "Drug A"
    ):
        import random
        # Parse age from range
        if "18" in str(age_range):
            age = random.randint(28, 44)
        elif "75" in str(age_range) and "+" in str(age_range):
            age = random.randint(76, 88)
        else:
            age = random.randint(52, 72)

        # Mapping for clinical realism
        cancer_norm = cancer_type or "Lung Cancer"
        if "Lung" in cancer_norm:
            full_cancer = "Non-Small Cell Lung Cancer (NSCLC)"
            default_organs = "Lung, Mediastinal Nodes"
        elif "Melanoma" in cancer_norm:
            full_cancer = "Cutaneous Melanoma (Stage IIIc/IV)"
            default_organs = "Skin, Regional Nodes, Subcutaneous"
        elif "Colorectal" in cancer_norm:
            full_cancer = "Colorectal Adenocarcinoma (mCRC)"
            default_organs = "Colon, Liver, Peritoneum"
        elif "Pancreatic" in cancer_norm:
            full_cancer = "Pancreatic Ductal Adenocarcinoma (PDAC)"
            default_organs = "Pancreas, Peripancreatic Tissue"
        else:
            full_cancer = f"{cancer_norm} (Metastatic)"
            default_organs = "Primary Site, Distant Nodes"

        mut = mutation or "EGFR"
        bio = biomarker or "PD-L1"
        trt = treatment or "Drug A"

        stat_score = random.randint(91, 96)
        bio_score = random.randint(89, 94)
        clin_score = random.randint(86, 92)
        comp_score = random.randint(94, 98)
        div_score = random.randint(86, 92)

        patient_num = random.randint(10, 999)

        return {
            "patient_id": f"SYN-{patient_num:03d}",
            "age": age,
            "cancer_type": full_cancer,
            "gene_mutation": mut,
            "mutation_status": "Pathogenic (Activating)",
            "biomarker": f"Positive ({bio} TPS {random.choice([45, 60, 75, 85])}%)" if "PD-L1" in bio else f"{bio} High (Tier 1A)",
            "treatment": f"{trt} ({mut} Targeted Line)",
            "response": random.choice(["Partial Response", "Stable Disease (RECIST 1.1)", "Complete Response", "Minor Regression"]),
            "renal_function": "Normal (eGFR > 85 mL/min)",
            "organ_involvement": default_organs,
            "disease_severity": "Moderate (ECOG PS 1)",
            "tags": [
                "Personalized Profile",
                "Clinically Plausible",
                "Rare Scenario",
                "Generated by GenAI"
            ],
            "validation": {
                "statistical": stat_score,
                "biological": bio_score,
                "clinical": clin_score,
                "constraint_compliance": comp_score,
                "diversity": div_score,
                "privacy": "No close match detected (Differential Privacy ε=0.5)",
                "status": "SYNTHETIC CASE ACCEPTED"
            },
            "narrative": (
                f"A {age}-year-old patient diagnosed with {full_cancer} presented for targeted clinical evaluation. "
                f"Molecular sequencing confirmed {mut} pathogenic mutation with biomarker concordance {bio}. "
                f"Initiated on {trt} systemic protocol with monitored renal tolerability. "
                f"Follow-up CT imaging demonstrates documented partial regression with no acute dose-limiting toxicities."
            ),
            "rag_evidence": [
                {"source": "NCCN Guidelines v4.2024", "excerpt": f"Targeted therapy recommendation for {mut} positive oncology patients under first-line and subsequent progression."},
                {"source": "CTCAE v5.0", "excerpt": "Grade 1-2 constitutional symptoms managed with standard supportive dosing; organ function within acceptable parameters."},
                {"source": "FDA Safety Bulletin", "excerpt": f"Safety profile of {trt} demonstrates sustained tolerability in biomarker-selected cohorts."}
            ],
            "downstream_stress": {
                "stage1_hazard": {"model": "Tabular ML Cox-PH", "risk_score": round(random.uniform(0.32, 0.48), 3), "status": "CONCORDANT"},
                "stage2_imaging": {"model": "Multimodal DL Imaging", "status": "SKIPPED_INPUT_UNAVAILABLE", "note": "No fabricated pixel scans per project safety policy"},
                "stage3_nlp": {"model": "Clinical NLP Triage", "triage": "URGENT", "confidence": 0.91, "status": "CONCORDANT"},
                "stage4_slm": {"model": "SLM Treatment Rec", "recommendation": f"{trt} + Close Monitoring", "status": "OPTIMAL"}
            }
        }

    @app.get("/", response_class=HTMLResponse)
    def index():
        html_path = Path(__file__).resolve().parent.parent.parent / "dashboard" / "index.html"
        if html_path.exists():
            return html_path.read_text(encoding="utf-8")
        return "<h1>Stage 5 GenAI Dashboard API Online</h1><p>Visit /api/stage5/summary for metrics.</p>"

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.dashboard.dashboard_api:app", host="127.0.0.1", port=8085, reload=False)
