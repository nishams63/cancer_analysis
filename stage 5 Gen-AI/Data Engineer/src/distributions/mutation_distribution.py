"""Mutation frequency distribution calculation."""
import pandas as pd
from typing import List

class MutationDistributionBuilder:
    def compute(self, df: pd.DataFrame, source: str = "PROJECT_STAGE1", version: str = "v1.0") -> pd.DataFrame:
        mut_list = []
        for col in ["mutation_primary", "mutation_secondary"]:
            if col in df.columns:
                mut_list.extend(df[col].dropna().tolist())

        s = pd.Series(mut_list)
        counts = s.value_counts()
        total = len(df)
        records = []
        for mut, cnt in counts.items():
            records.append({
                "mutation": str(mut),
                "count": int(cnt),
                "frequency": round(float(cnt / total), 6),
                "source": source,
                "distribution_version": version
            })
        return pd.DataFrame(records).sort_values("count", ascending=False).reset_index(drop=True)
