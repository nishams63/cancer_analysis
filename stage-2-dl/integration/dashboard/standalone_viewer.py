"""
Stage 2 Deep Learning - Standalone Patient Multimodal Viewer & HTML Generator

Generates a standalone HTML report or console summary for any patient without
requiring an active web server.

MANDATORY NOTICE:
SYNTHETIC RESEARCH PROTOTYPE — NOT CLINICALLY VALIDATED
Synthetic data != Real patient evidence != Clinical validation
"""
import os
import sys
import argparse
from pathlib import Path
import json

DASHBOARD_DIR = Path(__file__).resolve().parent
INTEGRATION_DIR = DASHBOARD_DIR.parent
SRC_DIR = INTEGRATION_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

import config, integration_pipeline


def generate_patient_html_report(patient_id: str, output_path: str = None) -> str:
    """Generates a standalone self-contained HTML report for a patient."""
    tiles, bio_df = integration_pipeline._load_sample_patient_data(patient_id)
    result = integration_pipeline.run_patient_inference(
        patient_id=patient_id,
        pathology_tiles=tiles,
        temporal_history=bio_df
    )

    p_res = result['pathology_summary']
    t_res = result['temporal_summary']
    f_res = result['multimodal_fusion']
    prov = result['provenance']

    score_val = f_res['prototype_multimodal_risk_score']
    alert_lbl = f_res['prototype_alert_level']
    badge_color = "#c62828" if "HIGH" in alert_lbl else ("#ef6c00" if "MODERATE" in alert_lbl else "#2e7d32")

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
            <div class="metric-box"><div>Modality Status</div><div class="metric-val">{result['modality_status']}</div></div>
            <div class="metric-box"><div>Prototype Multimodal Risk</div><div class="metric-val">{score_val:.4f}</div></div>
            <div class="metric-box"><div>Prototype Alert Level</div><div class="metric-val"><span class="badge">{alert_lbl}</span></div></div>
            <div class="metric-box"><div>Forecasted ctDNA 30d</div><div class="metric-val">{t_res['predicted_ctDNA_30d_vaf']:.2f}%</div></div>
        </div>
    </div>

    <div class="card">
        <h3>🔬 Modality A: Histopathology Biopsy Analysis</h3>
        <p><strong>Tiles Analyzed:</strong> {p_res['num_tiles_analyzed']} (Aggregation: {p_res['aggregation_method']})</p>
        <div class="grid">
            <div class="metric-box"><div>P(Malignant)</div><div class="metric-val">{p_res['malignant_probability']:.4f}</div></div>
            <div class="metric-box"><div>P(Benign)</div><div class="metric-val">{p_res['benign_probability']:.4f}</div></div>
            <div class="metric-box"><div>P(Inflammation)</div><div class="metric-val">{p_res['inflammation_probability']:.4f}</div></div>
        </div>
        <p>Tile Classification Breakdown: {json.dumps(p_res['tile_predictions_breakdown'])}</p>
    </div>

    <div class="card">
        <h3>📈 Modality B: Longitudinal Biomarkers (Historical Window &le; 90d)</h3>
        <div class="grid">
            <div class="metric-box"><div>Historical Visits Analyzed</div><div class="metric-val">{t_res['input_sequence_length']} visits</div></div>
            <div class="metric-box"><div>Last Historical Day</div><div class="metric-val">Day {t_res['max_days_from_baseline']}</div></div>
            <div class="metric-box"><div>Progression Risk</div><div class="metric-val">{t_res['progression_probability']:.4f}</div></div>
            <div class="metric-box"><div>Predicted ctDNA 30d VAF</div><div class="metric-val">{t_res['predicted_ctDNA_30d_vaf']:.2f}%</div></div>
        </div>
    </div>

    <div class="card">
        <h3>⚖️ Engineering Explanation & Provenance</h3>
        <p><strong>Engineering Narrative:</strong> {result['engineering_explanation']}</p>
        <p><strong>Active Weights:</strong> {json.dumps(f_res['weights_used'])}</p>
        <p><strong>Pathology Checkpoint:</strong> <code>{prov['pathology_model_checkpoint']}</code></p>
        <p><strong>Temporal Checkpoint:</strong> <code>{prov['temporal_model_checkpoint']}</code></p>
        <p><strong>Timestamp:</strong> {prov['timestamp']} (UTC)</p>
    </div>
</body>
</html>
"""
    out_file = output_path or str(config.INTEGRATION_DIR / 'reports' / f"patient_{patient_id}_report.html")
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Standalone HTML report written to: {out_file}")
    return out_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient-id', type=str, default='PAT-0001')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    generate_patient_html_report(args.patient_id, args.output)
