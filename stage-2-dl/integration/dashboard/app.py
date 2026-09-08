"""
Stage 2 Deep Learning - Multimodal Oncology Research Prototype Dashboard

Streamlit research interface for inspecting multimodal patient predictions.
Unites:
  - ResNet-50 / ResNet-18 Pathology with Gated Attention-MIL
  - Continuous Temporal Transformer / BiLSTM Forecaster
  - Learned Multimodal Fusion (Gated, Concat-MLP, Cross-Attention)
  - Epistemic Uncertainty (MC Dropout) & Validation Temperature Calibration
  - Latent Out-of-Distribution (OOD) Detection

MANDATORY REGULATORY BANNER:
SYNTHETIC RESEARCH PROTOTYPE — NOT CLINICALLY VALIDATED
Synthetic data != Real patient evidence != Clinical validation
"""
import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Path resolution
DASHBOARD_DIR = Path(__file__).resolve().parent
INTEGRATION_DIR = DASHBOARD_DIR.parent
SRC_DIR = INTEGRATION_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(INTEGRATION_DIR.parent))

try:
    import config, schemas, validation
    from inference_service import PatientInferenceService
    from integration_pipeline import _load_sample_patient_data
except (ImportError, ValueError):
    from . import config, schemas, validation
    from .inference_service import PatientInferenceService
    from .integration_pipeline import _load_sample_patient_data

# Streamlit Page Configuration
st.set_page_config(
    page_title="Multimodal Oncology Prototype Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. MANDATORY TOP BANNER
st.markdown(
    f"""
    <div style="background-color: #ffebee; border: 2px solid #c62828; border-radius: 8px; padding: 14px; margin-bottom: 20px;">
        <h3 style="color: #c62828; margin: 0; font-size: 1.15rem; font-weight: bold;">
            ⚠️ SYNTHETIC RESEARCH PROTOTYPE — NOT CLINICALLY VALIDATED
        </h3>
        <p style="color: #b71c1c; margin: 6px 0 0 0; font-size: 0.90rem;">
            <strong>{config.EQUIVALENCE_DISCLAIMER}</strong><br>
            {config.MANDATORY_DISCLAIMER}
            <br><em>This prototype does NOT prescribe treatment and must NEVER be used for clinical decision-making.</em>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("🔬 Multimodal Oncology Patient Inference Explorer")
st.caption("Stage 2 Deep Learning: Pathology Attention-MIL + Longitudinal Temporal Transformer + Learned Fusion")

# 2. SIDEBAR CONFIGURATION
st.sidebar.header("⚙️ Patient & Model Setup")

# Patient selection
sample_patients = [f"PAT-{i:04d}" for i in range(1, 21)]
selected_patient_id = st.sidebar.selectbox("Select Patient Case", sample_patients, index=0)

architecture_choice = st.sidebar.radio(
    "Model Architecture",
    ["Upgraded (ResNet-50 + Transformer + Gated)", "Baseline (ResNet-18 + BiLSTM + Fixed)"],
    index=0
)
is_upgraded = "Upgraded" in architecture_choice
model_config = "upgraded" if is_upgraded else "baseline"

if is_upgraded:
    fusion_mode = st.sidebar.selectbox("Learned Fusion Strategy", ["gated", "concat_mlp", "cross_attention"], index=0)
    aggregation_method = st.sidebar.selectbox("Pathology Aggregation", ["attention", "mean", "median", "max"], index=0)
else:
    fusion_mode = "fixed_weighted"
    aggregation_method = st.sidebar.selectbox("Pathology Aggregation", ["mean", "median", "max"], index=0)

# Modality availability toggles
st.sidebar.subheader("Simulate Modality Availability")
include_pathology = st.sidebar.checkbox("Include Pathology Biopsy Tiles", value=True)
include_temporal = st.sidebar.checkbox("Include Temporal Biomarkers", value=True)

# 3. DATA LOADING & INFERENCE EXECUTION
tiles, bio_df = _load_sample_patient_data(selected_patient_id)

obs_list = []
if include_temporal and bio_df is not None and len(bio_df) > 0:
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
    patient_id=selected_patient_id,
    tile_paths=tiles if include_pathology else None,
    temporal_observations=obs_list if include_temporal else None,
    model_configuration=model_config,
    fusion_mode=fusion_mode,
    aggregation_method=aggregation_method
)

service = PatientInferenceService.get_shared()
with st.spinner("Running unified patient inference..."):
    result = service.predict(req)

# 4. DASHBOARD DISPLAY
# Top Level Summary Cards
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Inference Mode", result.inference_mode)
with c2:
    prob_str = f"{result.progression_probability:.4f}" if result.progression_probability is not None else "N/A"
    st.metric("Progression Probability", prob_str)
with c3:
    forecast_str = f"{result.ctdna_forecast_30d:.2f}%" if result.ctdna_forecast_30d is not None else "N/A"
    st.metric("Forecasted 30d ctDNA VAF", forecast_str)
with c4:
    badge_color = "#c62828" if result.risk.level == "HIGH" else ("#ef6c00" if result.risk.level == "MODERATE" else "#2e7d32")
    st.markdown(f"**Research Risk Level**<br><span style='background-color:{badge_color}; color:white; padding:4px 10px; border-radius:4px; font-weight:bold;'>{result.risk.level}</span>", unsafe_allow_html=True)

st.markdown("---")

# Safety & Reliability Row
st.subheader("🛡️ Safety, Calibration & Out-of-Distribution Status")
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.metric("Confidence", f"{result.uncertainty.confidence:.4f}" if result.uncertainty.confidence else "N/A")
with s2:
    st.metric("Epistemic Uncertainty", f"{result.uncertainty.uncertainty_score:.4f}" if result.uncertainty.uncertainty_score else "N/A")
with s3:
    st.metric("Calibration Status", result.uncertainty.calibration_status)
with s4:
    ood_color = "#2e7d32" if result.ood.status == "IN_DISTRIBUTION" else "#c62828"
    st.markdown(f"**OOD Status**<br><span style='color:{ood_color}; font-weight:bold;'>{result.ood.status}</span>", unsafe_allow_html=True)
    if result.ood.mahalanobis_distance:
        st.caption(f"Mahalanobis Dist: {result.ood.mahalanobis_distance:.2f} (Threshold: {result.ood.threshold:.2f})")

st.markdown("---")

# Modality Tabs
tab_path, tab_temp, tab_fusion, tab_audit = st.tabs(["🔬 Pathology & Attention-MIL", "📈 Longitudinal Biomarkers", "🔗 Multimodal Fusion", "📋 Latency & Audit"])

with tab_path:
    if result.pathology:
        p_res = result.pathology
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("P(Malignant)", f"{p_res.malignant_probability:.4f}")
        col_p2.metric("P(Benign)", f"{p_res.benign_probability:.4f}")
        col_p3.metric("P(Inflammation)", f"{p_res.inflammation_probability:.4f}")

        if p_res.attention_tiles:
            st.markdown("#### Attention-MIL Ranked Biopsy Tiles")
            tile_cols = st.columns(min(len(p_res.attention_tiles), 6))
            for i, tile_info in enumerate(p_res.attention_tiles[:6]):
                with tile_cols[i]:
                    st.caption(f"Rank #{tile_info.rank} (Weight: {tile_info.attention_weight:.3f})")
                    if tile_info.tile_path and os.path.exists(tile_info.tile_path):
                        with Image.open(tile_info.tile_path) as im:
                            st.image(im, use_container_width=True)
                    else:
                        st.write(f"Tile {tile_info.tile_index}")
    else:
        st.info("Pathology modality was not provided in this inference run.")

with tab_temp:
    if result.temporal and bio_df is not None and len(bio_df) > 0:
        st.markdown("#### Longitudinal Biomarker Trajectory (Days 0–90)")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(bio_df['days_from_baseline'], bio_df['ctDNA_vaf_percent'], marker='o', label='ctDNA VAF (%)', color='#1976d2')
        if 'cea_ng_ml' in bio_df.columns:
            ax.plot(bio_df['days_from_baseline'], bio_df['cea_ng_ml'], marker='s', label='CEA (ng/mL)', color='#388e3c', linestyle='--')
        ax.axvline(90, color='red', linestyle=':', label='Historical Cutoff (Day 90)')
        ax.set_xlabel("Days from Baseline")
        ax.set_ylabel("Measurement")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Temporal modality was not provided in this inference run.")

with tab_fusion:
    if result.fusion:
        f_res = result.fusion
        st.markdown(f"**Fusion Architecture:** `{f_res.model_name}`")
        st.metric("Multimodal Risk Score", f"{f_res.multimodal_risk_score:.4f}" if f_res.multimodal_risk_score else "N/A")
        st.metric("Fusion Latent Dimension", f_res.fusion_dimension)
    else:
        st.info("Multimodal fusion was not executed (single modality or insufficient data).")

with tab_audit:
    st.markdown("#### Execution Latency Breakdown (ms)")
    lat = result.latency
    st.json(lat.model_dump())
    st.markdown("#### System Provenance")
    st.json(result.system)
