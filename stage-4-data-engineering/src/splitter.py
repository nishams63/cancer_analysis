"""
Patient-Level Dataset Splitting Module for Stage 4.
Guarantees strict patient isolation across Train, Validation, and Test splits
per Section 9.
"""

import random
import logging
from typing import Dict, List, Any, Tuple, Set
import numpy as np
import pandas as pd

logger = logging.getLogger("stage4.splitter")


class PatientSplitError(Exception):
    """Raised when patient partition integrity is violated."""
    pass


class PatientLevelSplitter:
    """Partitions clinical records into Train, Validation, and Test by unique patient ID."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        align_with_stage3: bool = True
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.align_with_stage3 = align_with_stage3

        tot = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(tot - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0 (got {tot})")

    def split_dataset(self, df: pd.DataFrame, patient_col: str = "patient_id") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Assigns 'split' column ensuring each unique patient exists in exactly one partition.
        """
        if patient_col not in df.columns:
            raise ValueError(f"Patient column '{patient_col}' not found in dataframe.")

        df_split = df.copy()

        # If data_split is present from Stage 3 and align_with_stage3 is requested:
        if self.align_with_stage3 and "data_split" in df.columns:
            logger.info("Aligning patient splits with Stage 3 canonical partitions...")
            split_map = {
                "TRAIN": "TRAIN",
                "VALIDATION": "VALIDATION",
                "LOCKED_TEST": "TEST",
                "TEST": "TEST"
            }
            df_split["split"] = df_split["data_split"].map(lambda s: split_map.get(str(s).upper(), "TRAIN"))
        else:
            logger.info("Computing patient-level splits from scratch with seed %d...", self.random_seed)
            unique_patients = sorted(df[patient_col].unique())
            rng = random.Random(self.random_seed)
            rng.shuffle(unique_patients)

            n_total = len(unique_patients)
            n_train = int(round(n_total * self.train_ratio))
            n_val = int(round(n_total * self.val_ratio))

            train_pats = set(unique_patients[:n_train])
            val_pats = set(unique_patients[n_train:n_train + n_val])
            test_pats = set(unique_patients[n_train + n_val:])

            def assign_split(pid: str) -> str:
                if pid in train_pats:
                    return "TRAIN"
                elif pid in val_pats:
                    return "VALIDATION"
                else:
                    return "TEST"

            df_split["split"] = df_split[patient_col].apply(assign_split)

        # Validate partition integrity
        pats_train = set(df_split[df_split["split"] == "TRAIN"][patient_col].unique())
        pats_val = set(df_split[df_split["split"] == "VALIDATION"][patient_col].unique())
        pats_test = set(df_split[df_split["split"] == "TEST"][patient_col].unique())

        leak_train_val = pats_train.intersection(pats_val)
        leak_train_test = pats_train.intersection(pats_test)
        leak_val_test = pats_val.intersection(pats_test)
        total_leakage = len(leak_train_val) + len(leak_train_test) + len(leak_val_test)

        if total_leakage > 0:
            raise PatientSplitError(
                f"FATAL LEAKAGE: Overlapping patients detected between splits! "
                f"Train-Val: {len(leak_train_val)}, Train-Test: {len(leak_train_test)}, Val-Test: {len(leak_val_test)}"
            )

        split_summary = {
            "status": "PASSED (0% LEAKAGE)",
            "total_records": len(df_split),
            "total_unique_patients": len(pats_train | pats_val | pats_test),
            "train": {
                "records": int((df_split["split"] == "TRAIN").sum()),
                "unique_patients": len(pats_train),
                "record_pct": round(float((df_split["split"] == "TRAIN").mean()), 4)
            },
            "validation": {
                "records": int((df_split["split"] == "VALIDATION").sum()),
                "unique_patients": len(pats_val),
                "record_pct": round(float((df_split["split"] == "VALIDATION").mean()), 4)
            },
            "test": {
                "records": int((df_split["split"] == "TEST").sum()),
                "unique_patients": len(pats_test),
                "record_pct": round(float((df_split["split"] == "TEST").mean()), 4)
            },
            "patient_leakage_count": total_leakage
        }

        logger.info(
            "Split summary: Train=%d docs (%d pats), Val=%d docs (%d pats), Test=%d docs (%d pats)",
            split_summary["train"]["records"], split_summary["train"]["unique_patients"],
            split_summary["validation"]["records"], split_summary["validation"]["unique_patients"],
            split_summary["test"]["records"], split_summary["test"]["unique_patients"]
        )

        return df_split, split_summary
