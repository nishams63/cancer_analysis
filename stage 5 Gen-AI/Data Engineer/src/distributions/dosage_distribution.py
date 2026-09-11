"""Observed project dosage ranges and reference boundaries."""
import pandas as pd

class DosageDistributionBuilder:
    def compute(self, df: pd.DataFrame, source: str = "PROJECT_STAGE1", version: str = "v1.0") -> pd.DataFrame:
        records = []
        if "drug_name" in df.columns and "drug_dose" in df.columns:
            for drug, grp in df.groupby("drug_name"):
                doses = grp["drug_dose"].dropna()
                if len(doses) == 0:
                    continue
                records.append({
                    "treatment": str(drug),
                    "minimum_observed": round(float(doses.min()), 2),
                    "maximum_observed": round(float(doses.max()), 2),
                    "median": round(float(doses.median()), 2),
                    "unit": "mg",
                    "source_type": "observed_project_range",
                    "source": source,
                    "version": version
                })
        return pd.DataFrame(records)
