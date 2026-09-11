"""Sparse biomarker regions and borderline value analysis."""
import pandas as pd
from typing import List, Dict, Any

class SparseRegionAnalyzer:
    def analyze(self, df_bio: pd.DataFrame) -> pd.DataFrame:
        records = []
        reg_id = 1

        # Borderline and high-uncertainty clinical envelopes
        vulnerable_regions = [
            {
                "biomarker_name": "creatinine_level",
                "unit": "mg/dL",
                "sparse_range_min": 1.4,
                "sparse_range_max": 2.0,
                "vulnerability_type": "borderline_nephrotoxicity_threshold",
                "observed_sample_density": "sparse_tail",
                "uncertainty_score": 0.78,
                "clinical_context": "Critical boundary for Cisplatin contraindication (CrCl < 50 mL/min)"
            },
            {
                "biomarker_name": "ctdna_level",
                "unit": "ng/mL",
                "sparse_range_min": 5.0,
                "sparse_range_max": 20.0,
                "vulnerability_type": "extreme_tumor_shedding",
                "observed_sample_density": "extreme_upper_quantile",
                "uncertainty_score": 0.85,
                "clinical_context": "Correlates with rapid metastatic disease progression and early therapy failure"
            },
            {
                "biomarker_name": "tumor_marker_level",
                "unit": "U/mL",
                "sparse_range_min": 250.0,
                "sparse_range_max": 1000.0,
                "vulnerability_type": "extreme_serum_marker_elevation",
                "observed_sample_density": "sparse_upper_quantile",
                "uncertainty_score": 0.72,
                "clinical_context": "High divergence with localized early stage labels"
            },
            {
                "biomarker_name": "liver_function_marker",
                "unit": "U/L",
                "sparse_range_min": 120.0,
                "sparse_range_max": 350.0,
                "vulnerability_type": "borderline_grade2_grade3_hepatotoxicity",
                "observed_sample_density": "sparse_transition_zone",
                "uncertainty_score": 0.80,
                "clinical_context": "Dose reduction / withholding trigger for Docetaxel and targeted TKIs"
            },
            {
                "biomarker_name": "platelet_count",
                "unit": "10^3/uL",
                "sparse_range_min": 20.0,
                "sparse_range_max": 50.0,
                "vulnerability_type": "grade3_thrombocytopenia_boundary",
                "observed_sample_density": "extreme_lower_tail",
                "uncertainty_score": 0.75,
                "clinical_context": "High risk of life-threatening hemorrhage during cytotoxic regimens"
            }
        ]

        for r in vulnerable_regions:
            records.append({
                "region_id": f"SPARSE-{reg_id:03d}",
                "biomarker_name": r["biomarker_name"],
                "unit": r["unit"],
                "sparse_range_min": r["sparse_range_min"],
                "sparse_range_max": r["sparse_range_max"],
                "vulnerability_type": r["vulnerability_type"],
                "observed_sample_density": r["observed_sample_density"],
                "uncertainty_score": r["uncertainty_score"],
                "clinical_context": r["clinical_context"],
                "candidate_stress_test": True
            })
            reg_id += 1

        return pd.DataFrame(records)
