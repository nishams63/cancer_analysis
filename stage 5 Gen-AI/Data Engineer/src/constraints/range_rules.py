"""Numerical and categorical range constraints for patient sampling."""
from typing import Dict, Any

class RangeRulesBuilder:
    def build(self) -> Dict[str, Any]:
        return {
            "schema_constraints": {
                "patient_id": {"type": "string", "nullable": False, "pattern": "^PT-[0-9]{6}$"},
                "age": {"type": "integer", "nullable": False, "min": 18, "max": 100},
                "sex": {"type": "string", "nullable": False, "allowed": ["Male", "Female", "Unknown"]},
                "cancer_type": {
                    "type": "string", "nullable": False,
                    "allowed": ["Non-Small Cell Lung Cancer", "Breast Cancer", "Colorectal Cancer", "Pancreatic Cancer", "Unknown"]
                },
                "cancer_stage": {
                    "type": "string", "nullable": False,
                    "allowed": ["Stage I", "Stage II", "Stage III", "Stage IV", "Unknown"]
                }
            },
            "biomarker_range_constraints": {
                "ctdna_level": {"unit": "ng/mL", "min": 0.0, "max": 25.0, "distribution_quantile_p95": 8.5},
                "tumor_marker_level": {"unit": "U/mL", "min": 0.0, "max": 500.0, "distribution_quantile_p95": 120.0},
                "inflammation_marker": {"unit": "mg/L", "min": 0.0, "max": 150.0, "distribution_quantile_p95": 45.0},
                "gene_expression_score": {"unit": "score", "min": 0.0, "max": 100.0, "distribution_quantile_p95": 85.0},
                "hemoglobin": {"unit": "g/dL", "min": 5.0, "max": 19.0, "critical_threshold_low": 7.0},
                "white_blood_cell_count": {"unit": "10^3/uL", "min": 0.5, "max": 50.0, "critical_threshold_low": 1.0},
                "platelet_count": {"unit": "10^3/uL", "min": 10.0, "max": 1000.0, "critical_threshold_low": 25.0},
                "creatinine_level": {"unit": "mg/dL", "min": 0.3, "max": 10.0, "critical_threshold_high": 3.0},
                "liver_function_marker": {"unit": "U/L", "min": 5.0, "max": 500.0, "critical_threshold_high": 150.0}
            },
            "vital_sign_constraints": {
                "systolic_bp": {"unit": "mmHg", "min": 75.0, "max": 210.0},
                "diastolic_bp": {"unit": "mmHg", "min": 45.0, "max": 125.0},
                "heart_rate": {"unit": "bpm", "min": 45.0, "max": 180.0},
                "oxygen_saturation": {"unit": "%", "min": 75.0, "max": 100.0}
            }
        }
