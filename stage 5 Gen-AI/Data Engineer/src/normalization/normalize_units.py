"""Unit harmonization for clinical laboratory measurements."""
import pandas as pd

class UnitNormalizer:
    STANDARD_UNITS = {
        "drug_dose": "mg",
        "hemoglobin": "g/dL",
        "white_blood_cell_count": "10^3/uL",
        "platelet_count": "10^3/uL",
        "creatinine_level": "mg/dL",
        "liver_function_marker": "U/L",
        "systolic_bp": "mmHg",
        "diastolic_bp": "mmHg",
        "heart_rate": "bpm",
        "oxygen_saturation": "%",
        "ctdna_vaf_percent": "%",
        "ctdna_ng_ml": "ng/mL",
        "cea_ng_ml": "ng/mL",
        "ca125_u_ml": "U/mL",
        "ldh_u_l": "U/L",
        "crp_mg_l": "mg/L"
    }

    def get_standard_unit(self, field_name: str) -> str:
        return self.STANDARD_UNITS.get(field_name, "dimensionless")
