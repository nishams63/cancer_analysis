"""Mutation co-occurrence and joint rarity calculation."""
import pandas as pd
from itertools import combinations
from typing import List, Dict, Any

class MutationCooccurrenceBuilder:
    def compute(self, df: pd.DataFrame, source: str = "PROJECT_STAGE1", version: str = "v1.0") -> pd.DataFrame:
        total_patients = len(df)
        # Compute individual frequencies
        primary_counts = df["mutation_primary"].value_counts().to_dict() if "mutation_primary" in df.columns else {}
        secondary_counts = df["mutation_secondary"].value_counts().to_dict() if "mutation_secondary" in df.columns else {}
        
        all_genes = sorted(list(set(list(primary_counts.keys()) + list(secondary_counts.keys()))))
        if "None/Unknown" in all_genes:
            all_genes.remove("None/Unknown")

        # Map patient pairs
        cooc_counts: Dict[tuple, int] = {}
        for _, row in df.iterrows():
            m1 = str(row.get("mutation_primary", "None/Unknown"))
            m2 = str(row.get("mutation_secondary", "None/Unknown"))
            active = set()
            if m1 != "None/Unknown":
                active.add(m1)
            if m2 != "None/Unknown":
                active.add(m2)
            if len(active) >= 2:
                for pair in combinations(sorted(list(active)), 2):
                    cooc_counts[pair] = cooc_counts.get(pair, 0) + 1

        records = []
        for gene_a in all_genes:
            freq_a = float((df["mutation_primary"] == gene_a).sum() + (df["mutation_secondary"] == gene_a).sum()) / total_patients
            for gene_b in all_genes:
                if gene_a >= gene_b:
                    continue
                freq_b = float((df["mutation_primary"] == gene_b).sum() + (df["mutation_secondary"] == gene_b).sum()) / total_patients
                cnt = cooc_counts.get((gene_a, gene_b), 0)
                cooc_freq = float(cnt) / total_patients
                # Rarity score: product of marginals vs joint
                joint_rarity = "common" if cooc_freq >= 0.10 else ("uncommon" if cooc_freq >= 0.03 else ("rare" if cooc_freq >= 0.005 else "very_rare"))
                records.append({
                    "mutation_A": gene_a,
                    "mutation_B": gene_b,
                    "cooccurrence_count": int(cnt),
                    "cooccurrence_frequency": round(cooc_freq, 6),
                    "individual_frequency_A": round(freq_a, 6),
                    "individual_frequency_B": round(freq_b, 6),
                    "joint_rarity": joint_rarity,
                    "source": source,
                    "distribution_version": version
                })

        return pd.DataFrame(records).sort_values("cooccurrence_count", ascending=False).reset_index(drop=True)
