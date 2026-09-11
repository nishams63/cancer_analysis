"""Missingness pattern analysis for stress-testing missing clinical features."""
import pandas as pd

class MissingnessDistributionBuilder:
    def compute(self, df: pd.DataFrame, source: str = "PROJECT_STAGE1", version: str = "v1.0") -> pd.DataFrame:
        cols_to_check = [
            "ctdna_level", "tumor_marker_level", "inflammation_marker",
            "mutation_secondary", "smoking_history", "comorbidity_count"
        ]
        available_cols = [c for c in cols_to_check if c in df.columns]
        missing_matrix = df[available_cols].isna()
        
        # Pattern identification
        patterns = missing_matrix.apply(lambda row: "+".join([col for col, val in row.items() if val]) or "COMPLETE", axis=1)
        pattern_counts = patterns.value_counts()
        total = len(df)

        records = []
        for pid, (pat, cnt) in enumerate(pattern_counts.items(), 1):
            records.append({
                "pattern_id": f"MISS-{pid:03d}",
                "missing_fields": str(pat),
                "count": int(cnt),
                "frequency": round(float(cnt / total), 6),
                "percentage": round(float(cnt / total * 100), 2),
                "source": source,
                "version": version
            })
        return pd.DataFrame(records)
