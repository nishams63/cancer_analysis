"""
Stage 4 End-to-End Orchestrator: SLM Fine-Tuning Dataset Pipeline.
Academic Project: Personalized Precision Medicine for Oncology Treatment Optimization.

Executes:
Stage 3 Ingestion -> Data Validation -> Draft Generation -> Entity Quality Gate ->
Circuit Breaker Check -> Human Review Audit -> Instruction Formatting ->
Patient-Level Split -> Leakage Audit -> Parquet Export -> Quality Reports.
"""

import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Ensure project paths are resolved
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(CURRENT_DIR))

from data_loader import Stage4DataLoader
from validator import Stage4DataValidator
from summary_generator import DraftTargetGenerator
from entity_validator import EntityPreservationQualityGate, EntityNormalizer
from rejection_monitor import RejectionMonitor
from formatter import InstructionTuningFormatter
from splitter import PatientLevelSplitter
from leakage_audit import LeakageAuditor
from quality_report import Stage4QualityReporter

logger = logging.getLogger("stage4.pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_stage4_pipeline(config_path: str = None) -> Dict[str, Any]:
    """Executes the complete Stage 4 SLM fine-tuning data pipeline."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    logger.info("=" * 75)
    logger.info("STAGE 4 — SLM FINE-TUNING DATASET PIPELINE (v2)")
    logger.info("Config file: %s", config_path)
    logger.info("=" * 75)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve paths relative to scratch repo root
    repo_root = PROJECT_ROOT.parent.parent
    stage3_path = repo_root / cfg["paths"]["stage3_parquet_path"]
    data_dir = repo_root / cfg["paths"]["output_data_dir"]
    reports_dir = repo_root / cfg["paths"]["reports_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    synonyms_path = PROJECT_ROOT / "config" / "drug_synonyms.yaml"
    gen_config_path = PROJECT_ROOT / "config" / "generation_config.yaml"

    final_dataset_path = repo_root / cfg["paths"]["final_dataset_path"]
    rejected_dataset_path = repo_root / cfg["paths"]["rejected_dataset_path"]
    generation_log_path = repo_root / cfg["paths"]["generation_log_path"]
    report_json_path = repo_root / cfg["paths"]["quality_report_json"]
    report_md_path = repo_root / cfg["paths"]["quality_report_md"]

    # ---------------------------------------------------------
    # STEP 1: INGEST & JOIN STAGE 3 DATA
    # ---------------------------------------------------------
    logger.info("\n[STEP 1] Ingesting Stage 3 notes and NER entities...")
    loader = Stage4DataLoader(stage3_parquet_path=str(stage3_path))
    df_raw, loader_audit = loader.load_and_join()

    # ---------------------------------------------------------
    # STEP 2: PRE-GENERATION DATA VALIDATION
    # ---------------------------------------------------------
    logger.info("\n[STEP 2] Pre-generation validation of clinical notes and identifiers...")
    validator = Stage4DataValidator()
    df_clean, df_invalid_input, validator_audit = validator.validate_dataset(df_raw)

    # ---------------------------------------------------------
    # STEP 3: DRAFT TARGET GENERATION & PROVENANCE LOGGING
    # ---------------------------------------------------------
    logger.info("\n[STEP 3] Generating draft targets with provenance tracking...")
    generator = DraftTargetGenerator(generation_config_path=str(gen_config_path))
    df_drafted, df_gen_log = generator.generate_draft_targets(df_clean)

    # Export generation log
    df_gen_log.to_parquet(generation_log_path, index=False)
    logger.info("  --> Saved generation log with provenance to: %s", generation_log_path)

    # ---------------------------------------------------------
    # STEP 4: ENTITY-PRESERVATION QUALITY GATE & NORMALIZATION
    # ---------------------------------------------------------
    logger.info("\n[STEP 4] Evaluating entity-preservation quality gate...")
    normalizer = EntityNormalizer(drug_synonyms_path=str(synonyms_path))
    quality_gate = EntityPreservationQualityGate(normalizer=normalizer)
    df_pass, df_gate_reject, gate_metrics = quality_gate.evaluate_dataset(df_drafted)

    # ---------------------------------------------------------
    # STEP 5: REJECTION-RATE CIRCUIT BREAKER
    # ---------------------------------------------------------
    logger.info("\n[STEP 5] Checking rejection-rate circuit breaker...")
    circuit_monitor = RejectionMonitor(
        max_rejection_rate=cfg["circuit_breaker"]["max_rejection_rate"],
        warn_rejection_rate=cfg["circuit_breaker"]["warn_rejection_rate"]
    )
    cb_result = circuit_monitor.check_circuit_breaker(
        total_count=gate_metrics["total_evaluated"],
        reject_count=gate_metrics["reject_count"],
        batch_id="GLOBAL_DATASET"
    )

    # ---------------------------------------------------------
    # STEP 6: FORMAT FOR SLM INSTRUCTION TUNING
    # ---------------------------------------------------------
    logger.info("\n[STEP 6] Formatting validated PASS records...")
    formatter = InstructionTuningFormatter()
    df_formatted = formatter.format_dataset(df_pass)

    # Combine all rejected records (invalid inputs + quality gate rejections)
    all_rejects = []
    if not df_invalid_input.empty:
        all_rejects.append(df_invalid_input)
    if not df_gate_reject.empty:
        all_rejects.append(df_gate_reject)
    
    if all_rejects:
        df_all_rejects = pd.concat(all_rejects, ignore_index=True)
    else:
        df_all_rejects = pd.DataFrame()

    df_all_rejects.to_parquet(rejected_dataset_path, index=False)
    logger.info("  --> Saved %d rejected/flagged records to: %s", len(df_all_rejects), rejected_dataset_path)

    # ---------------------------------------------------------
    # STEP 7: PATIENT-LEVEL SPLIT
    # ---------------------------------------------------------
    logger.info("\n[STEP 7] Performing patient-level Train/Validation/Test split...")
    splitter = PatientLevelSplitter(
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
        random_seed=cfg["project"]["random_seed"],
        align_with_stage3=True
    )
    df_final, split_summary = splitter.split_dataset(df_formatted)

    # ---------------------------------------------------------
    # STEP 8: LEAKAGE & INTEGRITY AUDIT
    # ---------------------------------------------------------
    logger.info("\n[STEP 8] Conducting rigorous multi-dimensional leakage audit...")
    auditor = LeakageAuditor(
        similarity_threshold=cfg["leakage_audit"]["similarity_threshold"],
        max_tfidf_features=cfg["leakage_audit"]["max_features"],
        forbidden_outcome_terms=cfg["leakage_audit"]["forbidden_outcome_terms"]
    )
    leakage_result = auditor.run_full_audit(df_final)

    # Export final dataset
    df_final.to_parquet(final_dataset_path, index=False)
    logger.info("  --> Successfully saved final fine-tuning dataset (%d records) to: %s", len(df_final), final_dataset_path)

    # ---------------------------------------------------------
    # STEP 9: HUMAN REVIEW AUDIT & DATA QUALITY REPORT
    # ---------------------------------------------------------
    logger.info("\n[STEP 9] Conducting human-review spot-check and compiling report...")
    reporter = Stage4QualityReporter(reports_dir=str(reports_dir))
    human_audit_summary, human_audit_md = reporter.conduct_human_review_audit(
        df_pass=df_final,
        df_reject=df_all_rejects,
        sample_pass=cfg["human_audit"]["sample_size_pass"],
        sample_reject=cfg["human_audit"]["sample_size_reject"],
        random_seed=cfg["human_audit"]["random_seed"]
    )

    final_report = reporter.generate_data_quality_report(
        loader_audit=loader_audit,
        validator_audit=validator_audit,
        quality_gate_metrics=gate_metrics,
        circuit_breaker_result=cb_result,
        split_summary=split_summary,
        leakage_audit_result=leakage_result,
        human_audit_summary=human_audit_summary,
        output_json_path=report_json_path,
        output_md_path=report_md_path
    )

    logger.info("=" * 75)
    logger.info("STAGE 4 PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("Final Accepted Records: %d", len(df_final))
    logger.info("Final Rejected Records: %d", len(df_all_rejects))
    logger.info("Rejection Rate: %.2f%%", cb_result["rejection_rate"] * 100)
    logger.info("Patient Leakage: %d", leakage_result["patient_leakage"])
    logger.info("=" * 75)

    return {
        "final_dataset": df_final,
        "rejected_dataset": df_all_rejects,
        "generation_log": df_gen_log,
        "quality_report": final_report
    }


if __name__ == "__main__":
    run_stage4_pipeline()
