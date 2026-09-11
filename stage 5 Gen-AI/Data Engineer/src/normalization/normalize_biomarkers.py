"""Biomarker name and representation harmonization."""
import pandas as pd

class BiomarkerNormalizer:
    CANONICAL_NAMES = {
        "ctdna_level": "ctdna_ng_ml",
        "ctDNA_vaf_percent": "ctdna_vaf_percent",
        "cea_ng_ml": "cea_ng_ml",
        "ca125_u_ml": "ca125_u_ml",
        "ldh_u_l": "ldh_u_l",
        "crp_mg_l": "crp_mg_l",
        "tumor_marker_level": "tumor_marker_u_ml",
        "inflammation_marker": "inflammation_marker_mg_l",
        "gene_expression_score": "gene_expression_score"
    }

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        rename_dict = {}
        for col in out.columns:
            if col in self.CANONICAL_NAMES:
                rename_dict[col] = self.CANONICAL_NAMES[col]
        return out.rename(columns=rename_dict)
