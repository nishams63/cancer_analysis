"""Pipeline to build reference distributions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.source_registry import SourceRegistry
from src.ingestion.load_project_data import ProjectDataLoader
from src.cleaning.clean_demographics import DemographicsCleaner
from src.cleaning.clean_mutations import MutationCleaner
from src.cleaning.clean_biomarkers import BiomarkersCleaner
from src.cleaning.clean_treatments import TreatmentCleaner
from src.cleaning.clean_adverse_events import AdverseEventCleaner
from src.cleaning.clean_timestamps import TimestampCleaner
from src.normalization.normalize_mutations import MutationNormalizer
from src.normalization.normalize_biomarkers import BiomarkerNormalizer
from src.normalization.normalize_treatments import TreatmentNormalizer
from src.normalization.normalize_categories import CategoryNormalizer
from src.distributions.distribution_builder import MasterDistributionBuilder
from src.utils.io import write_parquet, save_json
from src.utils.logging import get_logger

logger = get_logger("build_reference_data")

def run_build_reference_data() -> dict:
    logger.info("Starting Reference Data Build Pipeline...")
    reg = SourceRegistry()
    loader = ProjectDataLoader(reg)
    ingested = loader.ingest_all_project_data()
    df_stage1 = ingested["stage1"]["dataframe"]
    df_stage2 = ingested["stage2"]["dataframe"]

    # Cleaning
    all_issues = []
    df_clean, issues = DemographicsCleaner().clean(df_stage1)
    all_issues.extend(issues)
    df_clean, issues = MutationCleaner().clean(df_clean)
    all_issues.extend(issues)
    df_clean, issues = BiomarkersCleaner().clean(df_clean)
    all_issues.extend(issues)
    df_clean, issues = TreatmentCleaner().clean(df_clean)
    all_issues.extend(issues)
    df_clean, issues = AdverseEventCleaner().clean(df_clean)
    all_issues.extend(issues)
    df_clean, issues = TimestampCleaner().clean(df_clean)
    all_issues.extend(issues)

    # Save cleaned interim
    cleaned_dir = Path("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/interim/cleaned")
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(df_clean, cleaned_dir / "cleaned_master_patient.parquet")
    save_json(all_issues, cleaned_dir / "cleaning_issues_log.json")
    logger.info(f"Cleaned Stage 1 data. Logged {len(all_issues)} cleaning/quarantine issues.")

    # Normalization
    df_norm = MutationNormalizer().normalize(df_clean)
    df_norm = BiomarkerNormalizer().normalize(df_norm)
    df_norm = TreatmentNormalizer().normalize(df_norm)
    df_norm = CategoryNormalizer().normalize(df_norm)

    norm_dir = Path("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/interim/normalized")
    norm_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(df_norm, norm_dir / "normalized_master_patient.parquet")

    # Clean Stage 2 timestamps
    df_s2_clean, s2_issues = TimestampCleaner().clean(df_stage2)
    write_parquet(df_s2_clean, cleaned_dir / "cleaned_longitudinal.parquet")

    # Build Distributions
    dist_builder = MasterDistributionBuilder()
    distributions = dist_builder.build_all(df_norm, df_s2_clean)

    return {
        "ingested": ingested,
        "cleaning_issues": all_issues,
        "distributions": distributions
    }

if __name__ == "__main__":
    run_build_reference_data()
