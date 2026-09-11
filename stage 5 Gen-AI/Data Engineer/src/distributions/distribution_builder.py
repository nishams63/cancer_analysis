"""Master distribution builder compiling all reference distributions."""
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from .demographic_distribution import DemographicDistributionBuilder
from .mutation_distribution import MutationDistributionBuilder
from .mutation_cooccurrence import MutationCooccurrenceBuilder
from .biomarker_distribution import BiomarkerDistributionBuilder
from .treatment_distribution import TreatmentDistributionBuilder
from .dosage_distribution import DosageDistributionBuilder
from .adverse_event_distribution import AdverseEventDistributionBuilder
from .missingness_distribution import MissingnessDistributionBuilder
from .temporal_distribution import TemporalDistributionBuilder
from ..utils.io import write_parquet
from ..utils.logging import get_logger

logger = get_logger("distribution_builder")

class MasterDistributionBuilder:
    def __init__(self, processed_dir: str | Path = "C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/processed"):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def build_all(self, df_stage1: pd.DataFrame, df_stage2: pd.DataFrame, version: str = "v1.0") -> Dict[str, pd.DataFrame]:
        logger.info("Computing demographic distributions...")
        df_demo = DemographicDistributionBuilder().compute(df_stage1, version=version)
        
        logger.info("Computing mutation frequencies...")
        df_mut = MutationDistributionBuilder().compute(df_stage1, version=version)
        
        logger.info("Computing mutation co-occurrences...")
        df_cooc = MutationCooccurrenceBuilder().compute(df_stage1, version=version)
        
        logger.info("Computing biomarker distributions...")
        df_bio = BiomarkerDistributionBuilder().compute(df_stage1, version=version)
        
        logger.info("Computing treatment distributions...")
        df_tx = TreatmentDistributionBuilder().compute(df_stage1, version=version)
        
        logger.info("Computing dosage ranges...")
        df_dose = DosageDistributionBuilder().compute(df_stage1, version=version)
        
        logger.info("Computing adverse event distributions...")
        df_ae = AdverseEventDistributionBuilder().compute(df_stage1, version=version)
        
        logger.info("Computing missingness patterns...")
        df_miss = MissingnessDistributionBuilder().compute(df_stage1, version=version)
        
        logger.info("Computing temporal patterns...")
        df_temp = TemporalDistributionBuilder().compute(df_stage2, version=version)

        # Write individual Parquets
        write_parquet(df_mut, self.processed_dir / "mutation_frequencies.parquet")
        write_parquet(df_cooc, self.processed_dir / "mutation_cooccurrence.parquet")
        write_parquet(df_bio, self.processed_dir / "biomarker_distributions.parquet")
        write_parquet(df_tx, self.processed_dir / "treatment_distributions.parquet")
        write_parquet(df_dose, self.processed_dir / "dosage_ranges.parquet")
        write_parquet(df_ae, self.processed_dir / "adverse_event_distributions.parquet")
        write_parquet(df_miss, self.processed_dir / "missingness_patterns.parquet")
        write_parquet(df_temp, self.processed_dir / "temporal_patterns.parquet")

        # Compile consolidated reference_distributions.parquet
        consolidated = []
        for df, dist_type in [
            (df_demo, "demographic"), (df_bio, "biomarker"),
            (df_tx, "treatment"), (df_dose, "dosage"),
            (df_ae, "adverse_event"), (df_temp, "temporal")
        ]:
            c_df = df.copy()
            c_df["distribution_category"] = dist_type
            consolidated.append(c_df)
        df_ref = pd.concat(consolidated, ignore_index=True, sort=False)
        write_parquet(df_ref, self.processed_dir / "reference_distributions.parquet")

        logger.info(f"All 9 reference distribution parquets generated in {self.processed_dir}")
        return {
            "reference": df_ref,
            "mutation_frequencies": df_mut,
            "mutation_cooccurrence": df_cooc,
            "biomarker_distributions": df_bio,
            "treatment_distributions": df_tx,
            "dosage_ranges": df_dose,
            "adverse_event_distributions": df_ae,
            "missingness_patterns": df_miss,
            "temporal_patterns": df_temp
        }
