"""
Stage 2 Deep Learning - Multimodal Fusion Engine Module

Implements transparent, configurable multimodal fusion uniting:
  - Aggregated Pathology Malignant Probability (P_malignant)
  - Longitudinal Progression Probability (P_progression)
  - Forecasted 30-Day ctDNA VAF Risk Index (normalized ctDNA risk)

IMPORTANT FUSION VALIDATION NOTICE:
The default fusion weights and alert thresholds in this module are prototype
engineering parameters only. They are NOT clinically validated.
They were NOT optimized or tuned using the locked test set.
LOW, MODERATE, and HIGH represent prototype engineering alert categories, NOT clinically
established risk strata.
"""
from typing import Dict, Any, Optional
import numpy as np

try:
    from . import config, validation
except (ImportError, ValueError):
    import config, validation


def normalize_ctdna_risk(
    ctdna_vaf: Optional[float],
    vaf_min: float = config.CTDNA_VAF_MIN,
    vaf_max: float = config.CTDNA_VAF_MAX
) -> Optional[float]:
    """
    Normalizes continuous ctDNA VAF (%) into a [0, 1] risk index.
    Engineering scaling based on synthetic distribution dynamic range.
    """
    if ctdna_vaf is None or np.isnan(ctdna_vaf):
        return None
    clamped = max(vaf_min, min(float(ctdna_vaf), vaf_max))
    norm_risk = (clamped - vaf_min) / (vaf_max - vaf_min)
    return round(float(norm_risk), 4)


def map_score_to_prototype_alert(score: Optional[float]) -> str:
    """
    Maps a prototype multimodal risk score to an engineering alert category.
    Documented as prototype categories, NOT clinically calibrated strata.
    """
    if score is None or np.isnan(score):
        return config.PROTOTYPE_ALERT_LABELS['INSUFFICIENT_DATA']

    low_th = config.PROTOTYPE_ALERT_THRESHOLDS['low_upper']
    mod_th = config.PROTOTYPE_ALERT_THRESHOLDS['moderate_upper']

    if score < low_th:
        return config.PROTOTYPE_ALERT_LABELS['LOW']
    elif score < mod_th:
        return config.PROTOTYPE_ALERT_LABELS['MODERATE']
    else:
        return config.PROTOTYPE_ALERT_LABELS['HIGH']


def execute_multimodal_fusion(
    pathology_summary: Dict[str, Any],
    temporal_summary: Dict[str, Any],
    fusion_method: str = 'weighted_linear',
    custom_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Executes multimodal fusion with explicit missing-modality handling.

    Args:
        pathology_summary: Output from PatientPathologyAggregator.
        temporal_summary: Output from TemporalBiomarkerPredictor.
        fusion_method: 'weighted_linear' or 'rule_based'.
        custom_weights: Optional custom fusion weights.

    Returns:
        Structured multimodal fusion dictionary with engineering explanations.
    """
    pathology_avail = pathology_summary.get('available', False)
    temporal_avail = temporal_summary.get('available', False)

    # 1. Determine Modality Status
    if pathology_avail and temporal_avail:
        modality_status = "FULL_MULTIMODAL"
    elif pathology_avail and not temporal_avail:
        modality_status = "PATHOLOGY_ONLY"
    elif not pathology_avail and temporal_avail:
        modality_status = "TEMPORAL_ONLY"
    else:
        modality_status = "INSUFFICIENT_DATA"

    # Extract raw model signals
    p_mal = pathology_summary.get('malignant_probability')
    p_prog = temporal_summary.get('progression_probability')
    vaf_raw = temporal_summary.get('predicted_ctDNA_30d_vaf')
    vaf_risk = normalize_ctdna_risk(vaf_raw)

    weights_used = {}
    contributions = {}

    # 2. Case Handling by Modality Availability
    if modality_status == "FULL_MULTIMODAL":
        weights = custom_weights or config.DEFAULT_FUSION_WEIGHTS
        validation.validate_fusion_weights(weights)
        w_mal = weights['pathology_malignant']
        w_prog = weights['temporal_progression']
        w_vaf = weights['ctdna_vaf_risk']
        weights_used = weights

        if fusion_method == 'weighted_linear':
            score = (w_mal * p_mal) + (w_prog * p_prog) + (w_vaf * vaf_risk)
            score = round(float(np.clip(score, 0.0, 1.0)), 4)
            alert = map_score_to_prototype_alert(score)

            contributions = {
                'pathology_contribution': round(float(w_mal * p_mal), 4),
                'progression_contribution': round(float(w_prog * p_prog), 4),
                'ctdna_contribution': round(float(w_vaf * vaf_risk), 4)
            }
            explanation = (
                f"Multimodal score ({score:.4f}) computed via prototype weighted sum: "
                f"Pathology P_malignant={p_mal:.4f} (weight {w_mal:.2f}), "
                f"Temporal P_progression={p_prog:.4f} (weight {w_prog:.2f}), "
                f"ctDNA Risk Index={vaf_risk:.4f} (weight {w_vaf:.2f}, raw forecasted VAF {vaf_raw:.2f}%). "
                f"Mapped to prototype alert level: {alert}."
            )
        elif fusion_method == 'rule_based':
            # Conservative heuristic logic
            if p_mal >= 0.70 and p_prog >= 0.70:
                alert = config.PROTOTYPE_ALERT_LABELS['HIGH']
                score = round(max(p_mal, p_prog), 4)
            elif p_mal < 0.35 and p_prog < 0.35 and vaf_risk < 0.35:
                alert = config.PROTOTYPE_ALERT_LABELS['LOW']
                score = round(min(p_mal, p_prog), 4)
            else:
                alert = config.PROTOTYPE_ALERT_LABELS['MODERATE']
                score = round(float(np.mean([p_mal, p_prog, vaf_risk])), 4)

            explanation = (
                f"Rule-based heuristic prototype decision: P_mal={p_mal:.4f}, P_prog={p_prog:.4f}, "
                f"ctDNA_risk={vaf_risk:.4f} -> Resulting category: {alert}."
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

    elif modality_status == "PATHOLOGY_ONLY":
        weights_used = {'pathology_malignant': 1.0}
        score = p_mal
        alert = map_score_to_prototype_alert(score)
        contributions = {'pathology_contribution': p_mal}
        explanation = (
            f"Temporal modality unavailable. Single-modality pathology prototype score is {score:.4f} "
            f"(P_malignant across tiles). Mapped to prototype alert level: {alert}. "
            "Note: Partial-modality score is not equivalent to full multimodal evaluation."
        )

    elif modality_status == "TEMPORAL_ONLY":
        w_prog = config.TEMPORAL_ONLY_WEIGHTS['temporal_progression']
        w_vaf = config.TEMPORAL_ONLY_WEIGHTS['ctdna_vaf_risk']
        weights_used = config.TEMPORAL_ONLY_WEIGHTS
        score = (w_prog * p_prog) + (w_vaf * vaf_risk)
        score = round(float(np.clip(score, 0.0, 1.0)), 4)
        alert = map_score_to_prototype_alert(score)
        contributions = {
            'progression_contribution': round(float(w_prog * p_prog), 4),
            'ctdna_contribution': round(float(w_vaf * vaf_risk), 4)
        }
        explanation = (
            f"Pathology modality unavailable. Single-modality temporal prototype score is {score:.4f} "
            f"(P_progression={p_prog:.4f} [60%], ctDNA Risk={vaf_risk:.4f} [40%]). "
            f"Mapped to prototype alert level: {alert}. "
            "Note: Partial-modality score is not equivalent to full multimodal evaluation."
        )

    else:  # INSUFFICIENT_DATA
        score = None
        alert = config.PROTOTYPE_ALERT_LABELS['INSUFFICIENT_DATA']
        explanation = "Neither pathology nor temporal observations were provided. Cannot compute multimodal risk."

    return {
        'modality_status': modality_status,
        'fusion_method': fusion_method,
        'weights_used': weights_used,
        'multimodal_risk_score': score,
        'prototype_alert_level': alert,
        'normalized_ctdna_risk': vaf_risk,
        'score_components': contributions,
        'engineering_explanation': explanation,
        'clinical_disclaimer': config.MANDATORY_DISCLAIMER
    }
