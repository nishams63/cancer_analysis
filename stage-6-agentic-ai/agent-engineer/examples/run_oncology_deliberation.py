"""End-to-End Deliberative Agent Oncology Clinical Decision Support Example."""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure paths
CURRENT_DIR = Path(__file__).resolve().parent
AGENT_DIR = CURRENT_DIR.parent
STAGE6_DIR = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(1, str(STAGE6_DIR / "knowledge-engineer"))

import schemas
schemas_path = str(AGENT_DIR / "schemas")
if hasattr(schemas, "__path__") and schemas_path not in schemas.__path__:
    schemas.__path__.insert(0, schemas_path)

from schemas.deliberation import (
    PatientContext,
    PatientFact,
    FactProvenance,
    DeliberationStage,
)
from deliberation.agent import DeliberativeAgent
from deliberation.presenter import ClinicalPresenter


def main():
    print("=" * 60)
    print("ONCOLOGY CLINICAL DECISION SUPPORT SYSTEM (DELIBERATIVE AGENT)")
    print("=" * 60)

    # 1. Patient Clinical Context from EMR
    patient = PatientContext(
        patient_id="PT-ONC-2026-089",
        age=63,
        sex="F",
        cancer_type="NSCLC",
        stage="IV",
        biomarkers={
            "EGFR": "Exon 19 deletion (c.2235_2249del)",
            "ALK": "Negative",
            "ROS1": "Negative",
            "PD-L1": "TPS 15%",
        },
        pathology_findings=["Invasive lung adenocarcinoma", "Acinar pattern predominant"],
        imaging_findings=["4.2 cm primary in left lower lobe", "Multiple bilateral lung and pleural metastases"],
        laboratory_results={"creatinine": 0.95, "egfr": 78.0, "alt": 28, "ast": 24},
        comorbidities=["Controlled hypertension"],
        medications=["Lisinopril 10mg daily"],
    )

    question = "What is the recommended evidence-grounded first-line targeted therapy options to discuss for this patient?"
    print(f"Clinical Question:\n{question}\n")
    print(f"Patient Profile: {patient.patient_id}, {patient.age}y/{patient.sex}, {patient.cancer_type} Stage {patient.stage}")
    print(f"Biomarkers: {patient.biomarkers}\n")

    # 2. Instantiate Deliberative Agent
    agent = DeliberativeAgent()

    # 3. Deliberate across 12 structured stages
    print("-" * 60)
    print("DELIBERATION CYCLE PROGRESSION")
    print("-" * 60)
    state = agent.deliberate(clinical_question=question, patient_context=patient)

    for event in state.events:
        print(f"[{event.step_number:02d}] {event.event_name:<30} -> {event.summary}")

    # 4. Render Clinical Summary
    print("\n" + "-" * 60)
    presenter = ClinicalPresenter()
    rendered = presenter.render_markdown(state.final_decision_support)
    print(rendered)


if __name__ == "__main__":
    main()
