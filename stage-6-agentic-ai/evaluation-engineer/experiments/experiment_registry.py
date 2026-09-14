"""Persistence repository for evaluation experiment runs."""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from schemas.evaluation import OverallEvaluationResult


class ExperimentRegistry:
    """Stores and retrieves evaluation benchmark results for version comparison."""

    def __init__(self, storage_dir: Optional[Path | str] = None):
        if storage_dir is None:
            self.storage_dir = Path(__file__).resolve().parent / "results"
        else:
            self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_experiment(self, result: OverallEvaluationResult) -> str:
        file_path = self.storage_dir / f"{result.evaluation_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        return str(file_path)

    def get_experiment(self, evaluation_id: str) -> Optional[OverallEvaluationResult]:
        file_path = self.storage_dir / f"{evaluation_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return OverallEvaluationResult.model_validate_json(f.read())

    def list_experiments(self) -> List[str]:
        return [p.stem for p in self.storage_dir.glob("*.json")]
