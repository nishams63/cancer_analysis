"""
Data Ingestion and EDA Gate Enforcement Module for Stage 5 SLM.
Enforces the strict EDA readiness check, validates patient-level splits,
and verifies zero leakage across Train, Validation, and Test partitions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd

logger = logging.getLogger("stage5_slm.data_loader")


class SLMReadinessBlockError(Exception):
    """Raised when the upstream EDA readiness gate blocks SLM fine-tuning."""
    pass


class SLMDataLoader:
    """Manages dataset ingestion, EDA gate enforcement, and split isolation."""

    def __init__(
        self,
        dataset_path: str,
        eda_results_path: str,
        raw_dataset_path: Optional[str] = None
    ):
        self.dataset_path = Path(dataset_path)
        self.eda_results_path = Path(eda_results_path)
        self.raw_dataset_path = Path(raw_dataset_path) if raw_dataset_path else None

    def check_eda_readiness(self) -> Dict[str, Any]:
        """
        Inspects eda_results.json and extracts readiness determination.
        Raises SLMReadinessBlockError if status is 'NOT READY'.
        """
        if not self.eda_results_path.exists():
            raise FileNotFoundError(
                f"EDA results not found at: {self.eda_results_path}. Cannot verify data readiness."
            )

        with open(self.eda_results_path, "r", encoding="utf-8") as f:
            eda_results = json.load(f)

        final_readiness = eda_results.get("final_readiness", {})
        status = final_readiness.get("final_status", "UNKNOWN")
        reasons = final_readiness.get("decision_reasons", [])

        gate_report = {
            "status": status,
            "critical_count": final_readiness.get("critical_count", 0),
            "warning_count": final_readiness.get("warning_count", 0),
            "reasons": reasons,
            "is_blocked": status == "NOT READY"
        }

        if gate_report["is_blocked"]:
            block_msg = (
                "\n========================================\n"
                "SLM TRAINING BLOCKED\n"
                "========================================\n\n"
                "Reason:\n"
                f"EDA dataset readiness status = {status}\n"
                + "\n".join([f"- {r}" for r in reasons])
                + "\n\nAction:\n"
                "Resolve Stage 4 / EDA blocking findings and regenerate/re-audit the dataset.\n"
                "========================================\n"
            )
            logger.error(block_msg)
            return gate_report

        logger.info(f"EDA readiness gate PASSED: status is '{status}'.")
        return gate_report

    def load_dataset(self, enforce_gate: bool = True) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Loads the Stage 4 dataset, checks the EDA gate, and partitions into Train, Val, Test.
        """
        if enforce_gate:
            gate_info = self.check_eda_readiness()
            if gate_info["is_blocked"]:
                raise SLMReadinessBlockError(
                    f"SLM training blocked by EDA gate: status is '{gate_info['status']}'."
                )

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Stage 4 dataset not found at: {self.dataset_path}")

        df = pd.read_parquet(self.dataset_path)

        # Enforce required columns
        required_cols = ["patient_id", "note_id", "clinical_note", "instruction",
                         "target_risk", "target_key_finding", "target_action", "split"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Stage 4 dataset is missing required columns: {missing}")

        # Verify patient isolation
        train_df = df[df["split"] == "TRAIN"].copy()
        val_df = df[df["split"] == "VALIDATION"].copy()
        test_df = df[df["split"] == "TEST"].copy()

        train_pts = set(train_df["patient_id"])
        val_pts = set(val_df["patient_id"])
        test_pts = set(test_df["patient_id"])

        leakage_val = len(train_pts.intersection(val_pts))
        leakage_test = len(train_pts.intersection(test_pts))
        leakage_vt = len(val_pts.intersection(test_pts))
        total_leakage = leakage_val + leakage_test + leakage_vt

        if total_leakage > 0:
            raise ValueError(
                f"CRITICAL: Patient leakage detected across splits! "
                f"Train/Val: {leakage_val}, Train/Test: {leakage_test}, Val/Test: {leakage_vt}"
            )

        splits = {
            "train": train_df,
            "val": val_df,
            "test": test_df
        }
        return df, splits

    def load_raw_dataset(self) -> Optional[pd.DataFrame]:
        """Loads generation_log.parquet for Experiment B (Raw LoRA baseline)."""
        if not self.raw_dataset_path or not self.raw_dataset_path.exists():
            logger.warning("Raw generation log dataset not found.")
            return None
        return pd.read_parquet(self.raw_dataset_path)
