"""
Experiment Tracker, Loss Curves, and Reproducibility Module for Stage 5 SLM.
Manages experiment_registry.csv, per-experiment JSON metrics, reproducibility.json,
and renders loss curves per Sections 16, 17, & 34.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("stage5_slm.experiment_tracker")


class ExperimentTracker:
    """Tracks experiments, logs loss trajectories, and exports registry tables."""

    REGISTRY_COLUMNS = [
        "experiment_id",
        "model",
        "model_revision",
        "dataset_variant",
        "dataset_hash",
        "prompt_version",
        "tokenizer",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "learning_rate",
        "epochs",
        "seed",
        "train_records",
        "validation_records",
        "test_records",
        "training_time_sec",
        "peak_vram_mb",
        "best_checkpoint",
        "status"
    ]

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.metrics_dir = self.results_dir / "metrics"
        self.curves_dir = self.results_dir / "training_curves"
        self.registry_csv = self.results_dir / "experiment_registry.csv"

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.curves_dir.mkdir(parents=True, exist_ok=True)

    def log_experiment(
        self,
        experiment_id: str,
        config_metadata: Dict[str, Any],
        metrics: Dict[str, Any],
        training_history: Optional[List[Dict[str, Any]]] = None
    ) -> Path:
        """
        Logs an experiment run, appends to experiment_registry.csv,
        saves results/metrics/{experiment_id}.json, and generates curves.
        """
        # Save JSON metrics
        metric_file = self.metrics_dir / f"{experiment_id}.json"
        full_payload = {
            "experiment_id": experiment_id,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "configuration": config_metadata,
            "metrics": metrics,
            "training_history": training_history or []
        }
        with open(metric_file, "w", encoding="utf-8") as f:
            json.dump(full_payload, f, indent=2)

        # Update registry CSV
        reg_entry = {
            "experiment_id": experiment_id,
            "model": config_metadata.get("model_name", "unknown"),
            "model_revision": config_metadata.get("model_revision", "main"),
            "dataset_variant": config_metadata.get("dataset_variant", "filtered"),
            "dataset_hash": config_metadata.get("dataset_hash", ""),
            "prompt_version": config_metadata.get("prompt_version", "1.0.0"),
            "tokenizer": config_metadata.get("tokenizer", "cl100k_base"),
            "lora_r": config_metadata.get("lora_r", 0),
            "lora_alpha": config_metadata.get("lora_alpha", 0),
            "lora_dropout": config_metadata.get("lora_dropout", 0.0),
            "learning_rate": config_metadata.get("learning_rate", 0.0),
            "epochs": config_metadata.get("epochs", 0),
            "seed": config_metadata.get("seed", 42),
            "train_records": config_metadata.get("train_records", 0),
            "validation_records": config_metadata.get("val_records", 0),
            "test_records": config_metadata.get("test_records", 0),
            "training_time_sec": float(round(config_metadata.get("training_time_sec", 0.0), 2)),
            "peak_vram_mb": config_metadata.get("peak_vram_mb", 0.0),
            "best_checkpoint": config_metadata.get("best_checkpoint", "final"),
            "status": config_metadata.get("status", "COMPLETED")
        }

        if self.registry_csv.exists():
            df_reg = pd.read_csv(self.registry_csv)
            # Replace if experiment_id exists, else append
            df_reg = df_reg[df_reg["experiment_id"] != experiment_id]
            df_reg = pd.concat([df_reg, pd.DataFrame([reg_entry])], ignore_index=True)
        else:
            df_reg = pd.DataFrame([reg_entry])

        df_reg.to_csv(self.registry_csv, index=False)

        # Plot curves if history is available
        if training_history:
            self.plot_training_curves(experiment_id, training_history)

        return metric_file

    def plot_training_curves(self, experiment_id: str, history: List[Dict[str, Any]]) -> None:
        """Renders training and validation loss curves."""
        steps = [h.get("step", i) for i, h in enumerate(history)]
        train_loss = [h.get("loss", 0.0) for h in history]
        val_loss = [h.get("eval_loss") for h in history if "eval_loss" in h]
        val_steps = [h.get("step", i) for i, h in enumerate(history) if "eval_loss" in h]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, train_loss, label="Training Loss", color="#1f77b4", lw=2)
        if val_loss:
            ax.plot(val_steps, val_loss, label="Validation Loss", color="#d62728", lw=2, marker="o")
        ax.set_title(f"Loss Trajectory: {experiment_id}")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.curves_dir / f"{experiment_id}_loss.png", dpi=300)
        plt.close(fig)

    def export_reproducibility_manifest(self, manifest: Dict[str, Any]) -> Path:
        """Exports results/reproducibility.json capturing exact environment state."""
        out_path = self.results_dir / "reproducibility.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return out_path
