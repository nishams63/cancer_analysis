"""Adverse event and toxicity hazard distributions."""
import pandas as pd

class AdverseEventDistributionBuilder:
    def compute(self, df: pd.DataFrame, source: str = "PROJECT_STAGE1", version: str = "v1.0") -> pd.DataFrame:
        records = []
        if "toxicity_risk" in df.columns:
            counts = df["toxicity_risk"].value_counts()
            total = len(df)
            for risk, cnt in counts.items():
                records.append({
                    "adverse_event_category": str(risk),
                    "frequency": round(float(cnt / total), 6),
                    "count": int(cnt),
                    "severity_category": str(risk),
                    "treatment_relationship": "associated_toxicity_risk",
                    "source": source,
                    "version": version
                })
        return pd.DataFrame(records)
