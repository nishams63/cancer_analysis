"""
Stage 2 Deep Learning - Standalone Patient Multimodal Viewer & HTML Generator

Generates a self-contained HTML report for any patient using PatientInferenceService
without requiring an active web server.

MANDATORY REGULATORY NOTICE:
SYNTHETIC RESEARCH PROTOTYPE — NOT CLINICALLY VALIDATED
Synthetic data != Real patient evidence != Clinical validation
"""
import os
import sys
import argparse
from pathlib import Path
import json
import pandas as pd

DASHBOARD_DIR = Path(__file__).resolve().parent
INTEGRATION_DIR = DASHBOARD_DIR.parent
SRC_DIR = INTEGRATION_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(INTEGRATION_DIR.parent))

try:
    import config, schemas
    from inference_service import PatientInferenceService
    from integration_pipeline import _load_sample_patient_data
except (ImportError, ValueError):
    from . import config, schemas
    from .inference_service import PatientInferenceService
    from .integration_pipeline import _load_sample_patient_data


def generate_patient_html_report(patient_id: str, output_path: str = None) -> str:
    """Generates a standalone self-contained HTML report for a patient."""
    tiles, bio_df = _load_sample_patient_data(patient_id)

    obs_list = []
    if bio_df is not None and len(bio_df) > 0:
        for _, row in bio_df.iterrows():
            obs_list.append(schemas.BiomarkerObservation(
                days_from_baseline=int(row['days_from_baseline']),
                ctDNA_vaf_percent=float(row['ctDNA_vaf_percent']) if pd.notna(row.get('ctDNA_vaf_percent')) else None,
                cea_ng_ml=float(row['cea_ng_ml']) if pd.notna(row.get('cea_ng_ml')) else None,
                ca125_u_ml=float(row['ca125_u_ml']) if pd.notna(row.get('ca125_u_ml')) else None,
                ldh_u_l=float(row['ldh_u_l']) if pd.notna(row.get('ldh_u_l')) else None,
                crp_mg_l=float(row['crp_mg_l']) if pd.notna(row.get('crp_mg_l')) else None,
                delta_days=float(row['delta_days']) if pd.notna(row.get('delta_days')) else None,
            ))

    req = schemas.PatientInferenceRequest(
        patient_id=patient_id,
        tile_paths=tiles if tiles else None,
        temporal_observations=obs_list if obs_list else None,
        model_configuration="auto",
        fusion_mode="gated",
        aggregation_method="attention"
    )

    service = PatientInferenceService.get_shared()
    res = service.predict(req)

    badge_color = "#c62828" if res.risk.level == "HIGH" else ("#ef6c00" if res.risk.level == "MODERATE" else "#2e7d32")
    ood_color = "#2e7d32" if res.ood.status == "IN_DISTRIBUTION" else "#c62828"

    prob_val = f"{res.progression_probability:.4f}" if res.progression_probability is not None else "N/A"
    ctdna_val = f"{res.ctdna_forecast_30d:.2f}%" if res.ctdna_forecast_30d is not None else "N/A"

    p_summary = res.pathology
    t_summary = res.temporal
    f_summary = res.fusion

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Multimodal Patient Report - {patient_id} (Prototype)</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 30px; background-color: #f8f9fa; color: #212529; }}
        .banner {{ background-color: #ffebee; border: 2px solid #c62828; border-radius: 8px; padding: 15px; margin-bottom: 25px; }}
        .banner h3 {{ color: #c62828; margin: 0 0 6px 0; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; }}
        .metric-box {{ background: #eef2f5; padding: 12px; border-radius: 6px; text-align: center; }}
        .metric-val {{ font-size: 1.4rem; font-weight: bold; margin-top: 4px; color: #1a365d; }}
        .badge {{ background-color: {badge_color}; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; display: inline-block; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 8px 12px; border: 1px solid #dee2e6; text-align: left; }}
        th {{ background-color: #f1f3f5; }}
    </style>
</head>
<body>
    <div class="banner">
        <h3>⚠️ SYNTHETIC RESEARCH PROTOTYPE — NOT CLINICALLY VALIDATED</h3>
        <strong>{config.EQUIVALENCE_DISCLAIMER}</strong><br>
        {config.MANDATORY_DISCLAIMER}
    </div>

    <div class="card">
        <h2>Patient Multimodal Summary: {patient_id}</h2>
        <div class="grid">
            <div class="metric-box"><div>Inference Mode</div><div class="metric-val">{res.inference_mode}</div></div>
            <div class="metric-box"><div>Progression Probability</div><div class="metric-val">{prob_val}</div></div>
            <div class="metric-box"><div>Research Risk Level</div><div class="metric-val"><span class="badge">{res.risk.level}</span></div></div>
            <div class="metric-box"><div>Forecasted 30d ctDNA</div><div class="metric-val">{ctdna_val}</div></div>
        </div>
        <p style="margin-top: 15px; font-size: 0.95rem; color: #495057;">
            <strong>Status:</strong> {res.risk.joint_confidence_status} | 
            <strong>Model:</strong> {res.model_configuration} |
            <strong>Latency:</strong> {res.latency.total_ms:.1f} ms
        </p>
    </div>

    <div class="card">
        <h3>🛡️ Reliability, Uncertainty & OOD Detection</h3>
        <div class="grid">
            <div class="metric-box"><div>Confidence</div><div class="metric-val">{res.uncertainty.confidence or 'N/A'}</div></div>
            <div class="metric-box"><div>Epistemic Uncertainty</div><div class="metric-val">{res.uncertainty.uncertainty_score or 'N/A'}</div></div>
            <div class="metric-box"><div>Calibration Status</div><div class="metric-val" style="font-size: 1rem;">{res.uncertainty.calibration_status}</div></div>
            <div class="metric-box"><div>OOD Status</div><div class="metric-val" style="color: {ood_color};">{res.ood.status}</div></div>
        </div>
    </div>

    <div class="card">
        <h3>🔬 Modality A: Histopathology Biopsy Analysis</h3>
        <p><strong>Tiles Analyzed:</strong> {p_summary.num_tiles_analyzed if p_summary else 0} (Method: {p_summary.aggregation_method if p_summary else 'N/A'})</p>
        <div class="grid">
            <div class="metric-box"><div>P(Malignant)</div><div class="metric-val">{f"{p_summary.malignant_probability:.4f}" if p_summary else 'N/A'}</div></div>
            <div class="metric-box"><div>P(Benign)</div><div class="metric-val">{f"{p_summary.benign_probability:.4f}" if p_summary else 'N/A'}</div></div>
            <div class="metric-box"><div>P(Inflammation)</div><div class="metric-val">{f"{p_summary.inflammation_probability:.4f}" if p_summary else 'N/A'}</div></div>
        </div>
    </div>

    <div class="card">
        <h3>📈 Modality B: Longitudinal Biomarkers (Historical Window &le; 90d)</h3>
        <div class="grid">
            <div class="metric-box"><div>Visits Analyzed</div><div class="metric-val">{t_summary.sequence_length if t_summary else 0} visits</div></div>
            <div class="metric-box"><div>Last Historical Day</div><div class="metric-val">Day {t_summary.max_historical_day if t_summary else 0}</div></div>
            <div class="metric-box"><div>Progression Risk</div><div class="metric-val">{f"{t_summary.progression_probability:.4f}" if t_summary else 'N/A'}</div></div>
            <div class="metric-box"><div>Predicted 30d ctDNA</div><div class="metric-val">{f"{t_summary.ctdna_30d_forecast:.2f}%" if t_summary else 'N/A'}</div></div>
        </div>
    </div>
</body>
</html>
"""
    save_path = output_path or str(config.REPORTS_DIR / f"patient_report_{patient_id}.html")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone HTML Report Generator")
    parser.add_argument("--patient-id", type=str, default="PAT-0001", help="Patient ID")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    args = parser.parse_args()

    out = generate_patient_html_report(args.patient_id, args.output)
    print(f"Generated standalone report at: {out}")
