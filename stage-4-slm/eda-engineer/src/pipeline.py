"""
End-to-End Orchestrator and CLI for Stage 4 EDA Clinical Data Readiness Audit.
Executes complete profiling, tokenization, vocabulary, entity, negation, split,
leakage, and drift audits, generating publication figures and structured reports.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import pandas as pd

# Ensure src directory is in sys.path for direct execution
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Internal module imports
try:
    from .data_loader import Stage4DataLoader, load_stage3_reference_data
    from .profiling import profile_dataset_overview, analyze_missingness, analyze_duplicates
    from .token_analysis import TokenAnalyzer
    from .vocabulary_analysis import VocabularyAnalyzer
    from .entity_analysis import EntityAnalyzer
    from .negation_analysis import NegationAnalyzer
    from .risk_analysis import map_clinical_risk_tier, RiskAnalyzer
    from .split_analysis import SplitAndLeakageAuditor
    from .drift_analysis import DriftAnalyzer
    from .quality_flags import QualityFlagEvaluator
    from .visualization import FigureGenerator
    from .report_generator import ReportGenerator
except ImportError:
    # Direct script execution fallback
    from data_loader import Stage4DataLoader, load_stage3_reference_data
    from profiling import profile_dataset_overview, analyze_missingness, analyze_duplicates
    from token_analysis import TokenAnalyzer
    from vocabulary_analysis import VocabularyAnalyzer
    from entity_analysis import EntityAnalyzer
    from negation_analysis import NegationAnalyzer
    from risk_analysis import map_clinical_risk_tier, RiskAnalyzer
    from split_analysis import SplitAndLeakageAuditor
    from drift_analysis import DriftAnalyzer
    from quality_flags import QualityFlagEvaluator
    from visualization import FigureGenerator
    from report_generator import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("stage4_eda.pipeline")


def run_eda_pipeline(
    config_path: str,
    input_override: Optional[str] = None,
    output_override: Optional[str] = None,
    figures_override: Optional[str] = None,
    generate_figures: bool = True,
    strict_mode: bool = False
) -> Dict[str, Any]:
    """Runs the full EDA audit pipeline and returns the complete result payload."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve paths
    base_dir = cfg_file.parent
    input_path = Path(input_override) if input_override else Path(config["paths"]["input_dataset"])
    stage3_path = Path(config["paths"].get("stage3_dataset", ""))
    critical_terms_path = Path(config["paths"].get("critical_terms", "config/critical_terms.yaml"))
    reports_dir = Path(output_override) if output_override else Path(config["paths"]["reports_dir"])
    figures_dir = Path(figures_override) if figures_override else Path(config["paths"]["figures_dir"])

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Ingesting Stage 4 dataset: {input_path}")
    expected_sha = config.get("expected_metadata", {}).get("sha256")
    loader = Stage4DataLoader(str(input_path), expected_sha256=expected_sha)
    df, dataset_meta = loader.load_data()
    logger.info(f"Loaded {len(df):,} records across {len(df.columns)} columns.")

    # Upstream Stage 3 reference
    stage3_df = load_stage3_reference_data(str(stage3_path)) if stage3_path.exists() else None

    # 1. Profiling
    logger.info("Running dataset profiling & duplicate analysis...")
    overview = profile_dataset_overview(df)
    missingness = analyze_missingness(df)
    duplicates = analyze_duplicates(df)

    # 2. Token Analysis
    logger.info("Running BPE token-length & truncation risk analysis...")
    t_cfg = config.get("token_analysis", {})
    tok_analyzer = TokenAnalyzer(
        tokenizer_model=t_cfg.get("tokenizer_model", "cl100k_base"),
        context_limits=t_cfg.get("context_limits", [512, 1024, 2048, 4096])
    )
    token_results = tok_analyzer.analyze_token_distributions(df)
    trunc_risk = tok_analyzer.analyze_critical_entity_truncation_risk(
        df,
        target_context_limit=t_cfg.get("target_context_limit", 4096),
        tail_fraction=t_cfg.get("truncation_tail_fraction", 0.10)
    )
    token_results["critical_truncation_risk"] = trunc_risk

    # 3. Vocabulary & Tokenizer Fragmentation
    logger.info("Auditing clinical vocabulary & tokenizer fragmentation...")
    vocab_analyzer = VocabularyAnalyzer(
        critical_terms_yaml=str(critical_terms_path),
        tokenizer_model=t_cfg.get("tokenizer_model", "cl100k_base")
    )
    vocab_profile = vocab_analyzer.profile_vocabulary(df)
    frag_audit = vocab_analyzer.audit_tokenizer_fragmentation(df)

    # 4. Entity Analysis
    logger.info("Analyzing entity density & target preservation...")
    entity_analyzer = EntityAnalyzer()
    entity_dens = entity_analyzer.analyze_entity_density(df)
    entity_ret = entity_analyzer.analyze_entity_retention(df)

    # 5. Negation Scope Analysis
    logger.info("Auditing clinical negation scope & polarity preservation...")
    neg_analyzer = NegationAnalyzer()
    negation_metrics, negation_review_df = neg_analyzer.audit_negation_in_dataset(df)
    neg_parquet_path = reports_dir / "negation_review_cases.parquet"
    negation_review_df.to_parquet(neg_parquet_path, index=False)
    logger.info(f"Exported {len(negation_review_df)} negation audit cases to {neg_parquet_path}")

    # 6. Risk Distribution & Stratified Information Loss
    logger.info("Auditing target risk distribution & stratified information loss...")
    risk_series = map_clinical_risk_tier(df, stage3_df=stage3_df)
    risk_analyzer = RiskAnalyzer()
    risk_dist = risk_analyzer.analyze_risk_distribution(risk_series)
    strat_loss = risk_analyzer.analyze_stratified_information_loss(
        df,
        risk_series=risk_series,
        source_tokens=token_results["token_arrays"]["source"],
        target_tokens=token_results["token_arrays"]["target"]
    )

    # 7. Split Verification & Data Leakage Audit
    logger.info("Verifying train/val/test splits & running leakage audit...")
    sim_cfg = config.get("similarity", {})
    leakage_auditor = SplitAndLeakageAuditor(
        near_dup_threshold=sim_cfg.get("near_duplicate_threshold", 0.85),
        max_tfidf_features=sim_cfg.get("max_features", 5000)
    )
    leakage_json_path = reports_dir / "leakage_eda.json"
    leakage_report = leakage_auditor.run_full_leakage_audit(df, output_json_path=leakage_json_path)

    # 8. Drift Analysis
    logger.info("Analyzing temporal distribution & statistical cross-split drift...")
    drift_analyzer = DriftAnalyzer()
    temporal_report = drift_analyzer.analyze_temporal_distribution(df, stage3_df=stage3_df)
    stat_drift = drift_analyzer.analyze_distribution_drift(
        df,
        source_tokens=token_results["token_arrays"]["source"],
        risk_series=risk_series,
        entity_densities=entity_dens["overall_array"]
    )
    drift_report = {
        "temporal": temporal_report,
        "statistical_drift": stat_drift
    }

    # 9. Quality Flags Evaluation
    logger.info("Evaluating quality thresholds & computing final readiness...")
    evaluator = QualityFlagEvaluator(config.get("quality_thresholds", {}))
    # Context overflow rate at standard limit (4096)
    lim_4096 = token_results["metrics"]["context_limits_evaluation"].get("limit_4096", {})
    overflow_rate = lim_4096.get("overflow_percentage", 0.0)

    quality_payload = {
        "token_overflow_rate": overflow_rate,
        "negation_flip_rate": negation_metrics["negation_flip_rate"],
        "entity_retention_rate": entity_ret["overall_retention_rate"],
        "patient_leakage": leakage_report["patient_isolation"]["total_patient_leakage"],
        "exact_duplicate_rate": duplicates["exact_duplicate_rate"],
        "minority_risk_class_percentage": risk_dist["minority_class_percentage"]
    }
    quality_decision = evaluator.evaluate_all_flags(quality_payload)

    # Compile Full Audit Payload
    audit_results = {
        "dataset": dataset_meta,
        "overview": overview,
        "missingness": missingness,
        "duplicates": duplicates,
        "token_analysis": token_results,
        "vocabulary": vocab_profile,
        "tokenizer_fragmentation": frag_audit,
        "entity_analysis": entity_dens,
        "entity_retention": entity_ret,
        "negation_analysis": negation_metrics,
        "risk_analysis": risk_dist,
        "stratified_risk_loss": strat_loss,
        "split_analysis": leakage_report["patient_isolation"],
        "leakage": leakage_report,
        "drift": drift_report,
        "quality_flags": quality_decision,
        "final_readiness": quality_decision
    }

    # 10. Visualizations
    if generate_figures:
        logger.info("Generating 12 publication-quality PNG charts...")
        fig_gen = FigureGenerator(str(figures_dir))
        figs = fig_gen.generate_all_figures(
            df=df,
            token_arrays=token_results["token_arrays"],
            risk_data=risk_dist,
            entity_density_data=entity_dens,
            entity_retention_data=entity_ret,
            vocab_data=frag_audit,
            negation_data=negation_metrics,
            split_data=leakage_report["patient_isolation"],
            stratified_risk_loss=strat_loss
        )
        logger.info(f"Successfully generated {len(figs)} figures in {figures_dir}.")

    # 11. Reports
    logger.info("Generating structured reports (JSON & Markdown)...")
    rep_gen = ReportGenerator(str(reports_dir))
    json_path = rep_gen.generate_json_results(audit_results)
    md_path = rep_gen.generate_markdown_report(audit_results)
    logger.info(f"Saved: {json_path} and {md_path}")

    # 12. Final Console Summary Box per Section 24
    print("\n" + "=" * 40)
    print("EDA ENGINEER — DATA READINESS AUDIT")
    print("=" * 40)
    print(f"Records:              {overview['total_records']}")
    print(f"Patients:             {overview['unique_patients']}")
    print(f"Context overflow:     {overflow_rate:.2f}%")
    print(f"Entity retention:     {entity_ret['overall_retention_rate']*100:.2f}%")
    print(f"Negation flips:       {negation_metrics['negation_flip_rate']*100:.2f}%")
    print(f"Patient leakage:      {leakage_report['patient_isolation']['total_patient_leakage']}")
    print(f"Critical issues:      {quality_decision['critical_count']}")
    print(f"Warnings:             {quality_decision['warning_count']}")
    print("\nFINAL STATUS:")
    print(quality_decision["final_status"])
    print("=" * 40 + "\n")

    if strict_mode and quality_decision["final_status"] != "READY":
        logger.error(f"Strict mode failure: status is {quality_decision['final_status']}")
        sys.exit(1)

    return audit_results


def main():
    parser = argparse.ArgumentParser(description="Stage 4 Clinical SLM Training Data Readiness Audit")
    parser.add_argument("--config", type=str, default="stage-4-slm/eda-engineer/config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", type=str, default=None, help="Path to input parquet file")
    parser.add_argument("--output", type=str, default=None, help="Directory to save reports")
    parser.add_argument("--figures-dir", type=str, default=None, help="Directory to save figures")
    parser.add_argument("--generate-figures", action="store_true", default=True, help="Generate figures")
    parser.add_argument("--strict", action="store_true", default=False, help="Strict exit on warnings")

    args = parser.parse_args()
    run_eda_pipeline(
        config_path=args.config,
        input_override=args.input,
        output_override=args.output,
        figures_override=args.figures_dir,
        generate_figures=args.generate_figures,
        strict_mode=args.strict
    )


if __name__ == "__main__":
    main()
