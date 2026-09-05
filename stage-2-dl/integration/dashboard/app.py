"""
Stage 2 Deep Learning - Multimodal Oncology Research Prototype Dashboard

Streamlit research interface for inspecting multimodal patient predictions.

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

import config, integration_pipeline, fusion

# Streamlit Page Configuration
st.set_page_config(
    page_title="Multimodal Oncology Prototype",
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
st.caption("Stage 2 Deep Learning Prototype: Pathology CNN (ResNet-18) + Longitudinal Temporal BiLSTM")

# 2. SIDEBAR CONFIGURATION
st.sidebar.header("⚙️ Patient & Pipeline Setup")

# Patient selection
sample_patients = [f"PAT-{i:04d}" for i in range(1, 21)]
selected_patient_id = st.sidebar.selectbox("Select Patient Cohort Case", sample_patients, index=0)

st.sidebar.subheader("Multimodal Parameters")
aggregation_method = st.sidebar.selectbox(
    "Pathology Tile Aggregation",
    config.ALLOWED_TILE_AGGREGATIONS,
    index=0,
    help="Strategy to aggregate individual biopsy tile predictions into a patient-level signal."
)

fusion_method = st.sidebar.selectbox(
    "Fusion Strategy",
    ['weighted_linear', 'rule_based'],
    index=0,
    help="Multimodal fusion strategy uniting pathology and temporal predictions."
)

# Configurable weights (Prototype Engineering Choices)
st.sidebar.markdown("**Prototype Fusion Weights** *(Heuristic baseline)*")
w_mal = st.sidebar.slider("Pathology Malignant Weight", 0.0, 1.0, config.DEFAULT_FUSION_WEIGHTS['pathology_malignant'], 0.05)
w_prog = st.sidebar.slider("Temporal Progression Weight", 0.0, 1.0, config.DEFAULT_FUSION_WEIGHTS['temporal_progression'], 0.05)
w_ctdna = st.sidebar.slider("ctDNA VAF Risk Weight", 0.0, 1.0, config.DEFAULT_FUSION_WEIGHTS['ctdna_vaf_risk'], 0.05)

# Normalize weights
sum_w = w_mal + w_prog + w_ctdna
if sum_w > 0:
    custom_weights = {
        'pathology_malignant': round(w_mal / sum_w, 4),
        'temporal_progression': round(w_prog / sum_w, 4),
        'ctdna_vaf_risk': round(w_ctdna / sum_w, 4)
    }
else:
    custom_weights = config.DEFAULT_FUSION_WEIGHTS

# Modality availability toggles (for missing modality simulation)
st.sidebar.subheader("Simulate Modality Availability")
include_pathology = st.sidebar.checkbox("Include Pathology Tiles", value=True)
include_temporal = st.sidebar.checkbox("Include Temporal Biomarkers", value=True)


# 3. DATA LOADING & INFERENCE EXECUTION
tiles_raw, bio_raw = integration_pipeline._load_sample_patient_data(selected_patient_id)

tiles_input = tiles_raw if include_pathology else None
bio_input = bio_raw if include_temporal else None

with st.spinner("Executing multimodal pipeline inference..."):
    result = integration_pipeline.run_patient_inference(
        patient_id=selected_patient_id,
        pathology_tiles=tiles_input,
        temporal_history=bio_input,
        aggregation_method=aggregation_method,
        fusion_method=fusion_method,
        custom_weights=custom_weights
    )

modality_status = result['modality_status']
fusion_res = result['multimodal_fusion']
pathology_res = result['pathology_summary']
temporal_res = result['temporal_summary']


# 4. MAIN DASHBOARD PANELS

# Top Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Patient ID", value=selected_patient_id)

with col2:
    st.metric(
        label="Modality Status",
        value=modality_status,
        delta="Complete" if modality_status == "FULL_MULTIMODAL" else "Partial"
    )

with col3:
    score_val = fusion_res['prototype_multimodal_risk_score']
    st.metric(
        label="Prototype Multimodal Risk",
        value=f"{score_val:.4f}" if score_val is not None else "N/A",
        help="Weighted engineering score [0.0 - 1.0]. Prototype parameter only."
    )

with col4:
    alert_lbl = fusion_res['prototype_alert_level']
    badge_color = "#c62828" if "HIGH" in alert_lbl else ("#ef6c00" if "MODERATE" in alert_lbl else "#2e7d32")
    st.markdown(
        f"""
        <div style="background-color: {badge_color}; color: white; text-align: center; border-radius: 6px; padding: 10px; font-weight: bold; margin-top: 5px;">
            {alert_lbl}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Layout: 2 Columns (Left: Pathology & Temporal Details, Right: Fusion & Explainability)
left_col, right_col = st.columns([1.1, 0.9])

with left_col:
    # --- PATHOLOGY MODALITY SECTION ---
    st.subheader("🔬 Modality A: Histopathology Biopsy")
    if pathology_res['available']:
        p_mal = pathology_res['malignant_probability']
        p_ben = pathology_res['benign_probability']
        p_inf = pathology_res['inflammation_probability']

        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        p_col1.metric("Tiles Analyzed", pathology_res['num_tiles_analyzed'])
        p_col2.metric("P(Malignant)", f"{p_mal:.4f}")
        p_col3.metric("P(Benign)", f"{p_ben:.4f}")
        p_col4.metric("P(Inflammation)", f"{p_inf:.4f}")

        # Tile prediction breakdown
        b_down = pathology_res['tile_predictions_breakdown']
        st.caption(f"Tile Breakdown: {b_down['malignant']} Malignant | {b_down['benign']} Benign | {b_down['inflammation']} Inflammation")

        # Tile Gallery expander
        with st.expander(f"View {pathology_res['num_tiles_analyzed']} Biopsy Tile Gallery", expanded=False):
            tile_grid = st.columns(min(4, len(pathology_res['tile_details'])))
            for idx, t_info in enumerate(pathology_res['tile_details'][:8]):
                col_idx = idx % len(tile_grid)
                with tile_grid[col_idx]:
                    if os.path.exists(t_info['tile_identifier']):
                        img = Image.open(t_info['tile_identifier'])
                        st.image(img, caption=f"Tile {t_info['tile_index']}: {t_info['prediction']} ({t_info['confidence']:.2f})", use_container_width=True)
    else:
        st.info("Pathology modality not provided or disabled.")

    st.markdown("---")

    # --- TEMPORAL MODALITY SECTION ---
    st.subheader("📈 Modality B: Longitudinal Biomarkers (t <= 90d)")
    if temporal_res['available']:
        p_prog = temporal_res['progression_probability']
        vaf_30d = temporal_res['predicted_ctDNA_30d_vaf']
        norm_risk = temporal_res['normalized_ctdna_risk']

        t_col1, t_col2, t_col3 = st.columns(3)
        t_col1.metric("P(Progression Risk)", f"{p_prog:.4f}", help="Temporal BiLSTM binary progression probability")
        t_col2.metric("Forecasted ctDNA 30d", f"{vaf_30d:.2f}% VAF", help="Predicted ctDNA VAF 30 days forward")
        t_col3.metric("Normalized ctDNA Risk", f"{norm_risk:.4f}", help="Scaled [0, 1] risk index")

        # Historical Trajectory Plot
        if bio_input is not None and len(bio_input) > 0:
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.plot(bio_input['days_from_baseline'], bio_input['ctDNA_vaf_percent'], marker='o', color='teal', label='Historical ctDNA VAF (%)')

            # Add forecast point at max_day + 30
            max_day = bio_input['days_from_baseline'].max()
            forecast_day = max_day + 30
            ax.plot([max_day, forecast_day], [bio_input['ctDNA_vaf_percent'].iloc[-1], vaf_30d], 'r--', marker='s', label=f'30d Forecast ({vaf_30d:.2f}%)')

            ax.axvline(90, color='gray', linestyle=':', label='Forecast Boundary (Day 90)')
            ax.set_xlabel('Days from Baseline', fontsize=9)
            ax.set_ylabel('ctDNA VAF (%)', fontsize=9)
            ax.set_title(f'Historical Biomarker Trajectory & 30-Day Forecast ({selected_patient_id})', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Temporal biomarker modality not provided or disabled.")


with right_col:
    # --- FUSION & EXPLAINABILITY SECTION ---
    st.subheader("⚖️ Multimodal Fusion & Contribution")

    st.write(f"**Fusion Method:** `{fusion_res['fusion_method']}`")
    st.write(f"**Modality Status:** `{modality_status}`")

    # Score components bar chart
    if fusion_res['score_components']:
        comps = fusion_res['score_components']
        fig_c, ax_c = plt.subplots(figsize=(7, 3))
        keys = list(comps.keys())
        vals = [comps[k] for k in keys]
        clean_keys = [k.replace('_contribution', '').replace('_', ' ').title() for k in keys]

        colors = ['#3498db', '#e67e22', '#2ecc71']
        ax_c.barh(clean_keys, vals, color=colors[:len(keys)], edgecolor='black')
        ax_c.set_xlabel('Score Contribution to Total Risk', fontsize=9)
        ax_c.set_title('Engineering Feature Contribution Breakdown', fontsize=10, fontweight='bold')
        ax_c.set_xlim([0.0, 1.0])
        ax_c.grid(True, axis='x', alpha=0.3)
        st.pyplot(fig_c)
        plt.close()

    # Engineering Explanation Box
    st.markdown("**Engineering Explanation:**")
    st.info(result['engineering_explanation'])

    # Weight settings
    with st.expander("View Active Fusion Weights & Thresholds", expanded=False):
        st.json(fusion_res['weights_used'])
        st.caption("Thresholds: Low < 0.35, Moderate 0.35 - 0.70, High >= 0.70")

    # Provenance & Disclaimer Drawer
    with st.expander("Data Provenance & Checkpoint Verification", expanded=False):
        prov = result['provenance']
        st.write(f"**Timestamp:** `{prov['timestamp']}`")
        st.write(f"**Data Source:** `{prov['data_source']}`")
        st.write(f"**Validation Status:** `{prov['clinical_validation_status']}`")
        st.write(f"**Pathology Checkpoint:** `{prov['pathology_model_checkpoint']}`")
        st.write(f"**Temporal Checkpoint:** `{prov['temporal_model_checkpoint']}`")
        st.warning(prov['mandatory_disclaimer'])
