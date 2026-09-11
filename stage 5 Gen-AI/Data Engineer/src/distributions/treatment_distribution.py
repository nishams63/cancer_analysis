"""Treatment frequency distributions."""
import pandas as pd

class TreatmentDistributionBuilder:
    def compute(self, df: pd.DataFrame, source: str = "PROJECT_STAGE1", version: str = "v1.0") -> pd.DataFrame:
        records = []
        if "drug_name" in df.columns:
            counts = df["drug_name"].value_counts()
            total = len(df)
            for drug, cnt in counts.items():
                cat = df[df["drug_name"] == drug]["treatment_type"].iloc[0] if "treatment_type" in df.columns else "Systemic"
                records.append({
                    "treatment": str(drug),
                    "frequency": round(float(cnt / total), 6),
                    "count": int(cnt),
                    "associated_scenario": str(cat),
                    "source": source,
                    "version": version
                })
        return pd.DataFrame(records).sort_values("count", ascending=False).reset_index(drop=True)
