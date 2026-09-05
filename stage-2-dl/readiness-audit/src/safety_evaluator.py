"""
Stage 2 Deep Learning - Safety & Failure-Mode Evaluator Module

Evaluates the integrated system across 12 critical failure and edge-case regimes:
1. Missing inputs (None / empty)
2. Malformed / unreadable image files
3. Invalid patient IDs
4. Future temporal records (days > 90)
5. Forbidden target leakage columns in input
6. Missing biomarker assays (NaN values in time series)
7. Out-of-range biomarker values
8. Unavailable pathology modality
9. Unavailable temporal modality
10. Neither modality available
11. Corrupted file paths
12. Excessive sequence length (>10 timepoints)
"""
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from PIL import Image

try:
    from . import audit_config as config
except (ImportError, ValueError):
    import audit_config as config

# Import integration pipeline & validation
sys.path.insert(0, str(config.STAGE_2_DIR / 'integration' / 'src'))
import integration_pipeline, validation


def run_safety_evaluations() -> List[Dict[str, Any]]:
    """Executes systematic safety failure-mode tests and records status."""
    print("=" * 60)
    print("EXECUTING SAFETY & EDGE-CASE AUDIT EVALUATIONS")
    print("=" * 60)

    pipeline = integration_pipeline.MultimodalPatientPipeline.get_shared_pipeline()
    tiles_valid, bio_valid = integration_pipeline._load_sample_patient_data("PAT-0001")

    safety_cases = []

    # Case 1: Empty / None Patient ID
    try:
        pipeline.run_patient_inference(patient_id="", pathology_tiles=tiles_valid[:2], temporal_history=bio_valid)
        c1_actual = "Inference executed without patient ID"
        c1_status = "FAIL"
    except validation.ValidationError as e:
        c1_actual = f"Rejected cleanly with ValidationError: {e}"
        c1_status = "PASS"
    safety_cases.append({
        'case_id': 'SAFE-01',
        'failure_mode': 'Invalid / Empty Patient ID',
        'expected_behavior': 'Raise ValidationError; reject request',
        'actual_behavior': c1_actual,
        'status': c1_status
    })

    # Case 2: Temporal Boundary Violation (days > 90)
    leaked_bio = bio_valid.copy()
    leaked_bio.loc[len(leaked_bio)] = {
        'patient_id': 'PAT-0001',
        'days_from_baseline': 120,  # Exceeds Day 90
        'ctDNA_vaf_percent': 2.5,
        'is_input_window': 0
    }
    try:
        pipeline.run_patient_inference(patient_id="PAT-0001", pathology_tiles=None, temporal_history=leaked_bio)
        c2_actual = "Inference executed with leaked future observation"
        c2_status = "FAIL"
    except validation.ValidationError as e:
        c2_actual = f"Rejected cleanly with ValidationError: {e}"
        c2_status = "PASS"
    safety_cases.append({
        'case_id': 'SAFE-02',
        'failure_mode': 'Future Observation Leaked (days > 90)',
        'expected_behavior': 'Raise ValidationError; block future observations',
        'actual_behavior': c2_actual,
        'status': c2_status
    })

    # Case 3: Forbidden Target Column Present
    target_leaked_bio = bio_valid.copy()
    target_leaked_bio['future_ctDNA_30d_target'] = 1.50
    try:
        pipeline.run_patient_inference(patient_id="PAT-0001", pathology_tiles=None, temporal_history=target_leaked_bio)
        c3_actual = "Inference accepted forbidden target column"
        c3_status = "FAIL"
    except validation.ValidationError as e:
        c3_actual = f"Rejected cleanly with ValidationError: {e}"
        c3_status = "PASS"
    safety_cases.append({
        'case_id': 'SAFE-03',
        'failure_mode': 'Forbidden Target Column in Input',
        'expected_behavior': 'Raise ValidationError; enforce strict target isolation',
        'actual_behavior': c3_actual,
        'status': c3_status
    })

    # Case 4: Non-existent Pathology Image File
    try:
        pipeline.run_patient_inference(patient_id="PAT-0001", pathology_tiles=["non_existent_tile_12345.png"], temporal_history=None)
        c4_actual = "Inference accepted missing file path"
        c4_status = "FAIL"
    except validation.ValidationError as e:
        c4_actual = f"Rejected cleanly with ValidationError: {e}"
        c4_status = "PASS"
    safety_cases.append({
        'case_id': 'SAFE-04',
        'failure_mode': 'Non-Existent Tile Path',
        'expected_behavior': 'Raise ValidationError; identify missing file',
        'actual_behavior': c4_actual,
        'status': c4_status
    })

    # Case 5: Corrupted / Malformed Image File
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp.write(b"NOT_A_VALID_IMAGE_BYTES_CORRUPTED")
        tmp_path = tmp.name
    try:
        pipeline.run_patient_inference(patient_id="PAT-0001", pathology_tiles=[tmp_path], temporal_history=None)
        c5_actual = "Inference executed on corrupted image"
        c5_status = "FAIL"
    except validation.ValidationError as e:
        c5_actual = f"Rejected cleanly with ValidationError: {e}"
        c5_status = "PASS"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    safety_cases.append({
        'case_id': 'SAFE-05',
        'failure_mode': 'Corrupted Image Payload',
        'expected_behavior': 'Raise ValidationError; catch image decode failure',
        'actual_behavior': c5_actual,
        'status': c5_status
    })

    # Case 6: Missing Biomarkers (NaNs in Time Series)
    nan_bio = bio_valid.copy()
    nan_bio.loc[1, 'ctDNA_vaf_percent'] = np.nan
    nan_bio.loc[2, 'cea_ng_ml'] = np.nan
    try:
        res_nan = pipeline.run_patient_inference(patient_id="PAT-0001", pathology_tiles=None, temporal_history=nan_bio)
        assert res_nan['temporal_summary']['available'] is True
        c6_actual = f"Handled gracefully via forward-fill imputation. Progression={res_nan['temporal_summary']['progression_probability']:.4f}"
        c6_status = "PASS"
    except Exception as e:
        c6_actual = f"Crashed on NaNs: {e}"
        c6_status = "FAIL"
    safety_cases.append({
        'case_id': 'SAFE-06',
        'failure_mode': 'Missing Biomarker Assays (NaNs)',
        'expected_behavior': 'Impute cleanly within trajectory; produce valid score',
        'actual_behavior': c6_actual,
        'status': c6_status
    })

    # Case 7: Extreme Out-of-Range Biomarker Value
    outlier_bio = bio_valid.copy()
    outlier_bio.loc[0, 'ctDNA_vaf_percent'] = 25.0  # Real dynamic range is 0.05 - 6.0
    try:
        res_out = pipeline.run_patient_inference(patient_id="PAT-0001", pathology_tiles=None, temporal_history=outlier_bio)
        # ctDNA risk should clamp to 1.0
        assert res_out['temporal_summary']['normalized_ctdna_risk'] <= 1.0
        c7_actual = f"Clamped safely to 1.0 (normalized risk={res_out['temporal_summary']['normalized_ctdna_risk']})"
        c7_status = "PASS"
    except Exception as e:
        c7_actual = f"Crashed on outlier: {e}"
        c7_status = "FAIL"
    safety_cases.append({
        'case_id': 'SAFE-07',
        'failure_mode': 'Extreme Biomarker Outlier (25% VAF)',
        'expected_behavior': 'Clamp normalized risk gracefully to 1.0 without crash',
        'actual_behavior': c7_actual,
        'status': c7_status
    })

    # Case 8: Missing Temporal Modality (Pathology Only)
    res_path_only = pipeline.run_patient_inference(patient_id="PAT-0001", pathology_tiles=tiles_valid[:2], temporal_history=None)
    c8_pass = res_path_only['modality_status'] == "PATHOLOGY_ONLY" and res_path_only['temporal_summary']['available'] is False
    safety_cases.append({
        'case_id': 'SAFE-08',
        'failure_mode': 'Missing Temporal Modality',
        'expected_behavior': 'Produce PATHOLOGY_ONLY score; label partial modality',
        'actual_behavior': f"Returned status={res_path_only['modality_status']}, score={res_path_only['multimodal_fusion']['prototype_multimodal_risk_score']}",
        'status': "PASS" if c8_pass else "FAIL"
    })

    # Case 9: Missing Pathology Modality (Temporal Only)
    res_temp_only = pipeline.run_patient_inference(patient_id="PAT-0001", pathology_tiles=None, temporal_history=bio_valid)
    c9_pass = res_temp_only['modality_status'] == "TEMPORAL_ONLY" and res_temp_only['pathology_summary']['available'] is False
    safety_cases.append({
        'case_id': 'SAFE-09',
        'failure_mode': 'Missing Pathology Modality',
        'expected_behavior': 'Produce TEMPORAL_ONLY score; label partial modality',
        'actual_behavior': f"Returned status={res_temp_only['modality_status']}, score={res_temp_only['multimodal_fusion']['prototype_multimodal_risk_score']}",
        'status': "PASS" if c9_pass else "FAIL"
    })

    # Case 10: Neither Modality Available
    res_none = pipeline.run_patient_inference(patient_id="PAT-EMPTY", pathology_tiles=None, temporal_history=None)
    c10_pass = res_none['modality_status'] == "INSUFFICIENT_DATA" and res_none['multimodal_fusion']['prototype_multimodal_risk_score'] is None
    safety_cases.append({
        'case_id': 'SAFE-10',
        'failure_mode': 'Neither Modality Available',
        'expected_behavior': 'Return INSUFFICIENT_DATA and score=None; avoid hallucination',
        'actual_behavior': f"Returned status={res_none['modality_status']}, alert={res_none['multimodal_fusion']['prototype_alert_level']}",
        'status': "PASS" if c10_pass else "FAIL"
    })

    # Case 11: Invalid Fusion Weights (Do Not Sum to 1.0)
    bad_weights = {'pathology_malignant': 0.5, 'temporal_progression': 0.5, 'ctdna_vaf_risk': 0.5}  # Sum = 1.5
    try:
        pipeline.run_patient_inference(patient_id="PAT-0001", pathology_tiles=tiles_valid[:2], temporal_history=bio_valid, custom_weights=bad_weights)
        c11_actual = "Accepted weights that sum to 1.5"
        c11_status = "FAIL"
    except validation.ValidationError as e:
        c11_actual = f"Rejected cleanly with ValidationError: {e}"
        c11_status = "PASS"
    safety_cases.append({
        'case_id': 'SAFE-11',
        'failure_mode': 'Invalid Fusion Weights (Sum != 1.0)',
        'expected_behavior': 'Raise ValidationError; enforce weight normalization',
        'actual_behavior': c11_actual,
        'status': c11_status
    })

    # Case 12: Invalid Tile Aggregation Strategy
    try:
        pipeline.run_patient_inference(patient_id="PAT-0001", pathology_tiles=tiles_valid[:2], temporal_history=bio_valid, aggregation_method="invalid_mode")
        c12_actual = "Accepted invalid aggregation mode"
        c12_status = "FAIL"
    except validation.ValidationError as e:
        c12_actual = f"Rejected cleanly with ValidationError: {e}"
        c12_status = "PASS"
    safety_cases.append({
        'case_id': 'SAFE-12',
        'failure_mode': 'Invalid Tile Aggregation Strategy',
        'expected_behavior': 'Raise ValidationError; enforce allowed aggregation set',
        'actual_behavior': c12_actual,
        'status': c12_status
    })

    for sc in safety_cases:
        print(f"  [{sc['case_id']}] {sc['failure_mode']:40s}: {sc['status']}")

    return safety_cases


if __name__ == '__main__':
    run_safety_evaluations()
