"""Treatment and dosage constraint rules."""
from typing import Dict, Any

class TreatmentRulesBuilder:
    def build(self) -> Dict[str, Any]:
        return {
            "treatments": {
                "Docetaxel": {
                    "unit": "mg",
                    "approved_project_range": {"min": 60.0, "max": 300.0, "median": 140.0},
                    "allowed_modalities": ["Chemotherapy", "Adjuvant", "Neoadjuvant"],
                    "contraindicated_conditions": ["Severe hepatic impairment (ALT > 150 U/L)", "Bilirubin > 3x ULN"]
                },
                "Osimertinib": {
                    "unit": "mg",
                    "approved_project_range": {"min": 40.0, "max": 80.0, "median": 80.0},
                    "allowed_modalities": ["Targeted Therapy"],
                    "contraindicated_conditions": ["Pre-existing severe QTc prolongation (> 500 ms)"]
                },
                "Cisplatin": {
                    "unit": "mg",
                    "approved_project_range": {"min": 40.0, "max": 150.0, "median": 75.0},
                    "allowed_modalities": ["Chemotherapy"],
                    "contraindicated_conditions": ["Creatinine clearance < 50 mL/min", "Pre-existing severe renal failure"]
                },
                "Pembrolizumab": {
                    "unit": "mg",
                    "approved_project_range": {"min": 100.0, "max": 400.0, "median": 200.0},
                    "allowed_modalities": ["Immunotherapy"],
                    "contraindicated_conditions": ["Active life-threatening autoimmune disorder"]
                },
                "Trastuzumab": {
                    "unit": "mg",
                    "approved_project_range": {"min": 120.0, "max": 600.0, "median": 420.0},
                    "allowed_modalities": ["Targeted Therapy"],
                    "contraindicated_conditions": ["Severe congestive heart failure (LVEF < 40%)"]
                },
                "radiotherapy-standard": {
                    "unit": "Gy",
                    "approved_project_range": {"min": 20.0, "max": 70.0, "median": 60.0},
                    "allowed_modalities": ["Radiation", "Stereotactic"],
                    "contraindicated_conditions": ["Prior radiation toxicity in same anatomical field"]
                }
            },
            "general_dosage_policy": {
                "negative_doses_allowed": False,
                "zero_dose_allowed_for_withheld": True,
                "disclaimer": "Observed project ranges represent empirical bounds from research cohorts and must not be used as autonomous medical prescription advice."
            }
        }
