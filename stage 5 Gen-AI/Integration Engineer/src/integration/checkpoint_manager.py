import json
from pathlib import Path
from typing import Dict, Any, Optional
from ..utils.serialization import save_json, load_json
from ..utils.logging import get_logger

logger = get_logger("CheckpointManager")

class CheckpointManager:
    def __init__(self, checkpoint_dir: Optional[str] = None):
        if checkpoint_dir is None:
            base = Path(__file__).resolve().parent.parent.parent
            self.base_dir = base / "runtime" / "checkpoints"
        else:
            self.base_dir = Path(checkpoint_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, batch_id: str, scenario_id: str, step: str) -> Path:
        bdir = self.base_dir / batch_id
        bdir.mkdir(parents=True, exist_ok=True)
        return bdir / f"{scenario_id}_{step}.json"

    def save_checkpoint(self, batch_id: str, scenario_id: str, step: str, data: Dict[str, Any]) -> None:
        p = self._get_path(batch_id, scenario_id, step)
        save_json(data, str(p))
        logger.debug(f"Saved checkpoint: {p.name}")

    def load_checkpoint(self, batch_id: str, scenario_id: str, step: str) -> Optional[Dict[str, Any]]:
        p = self._get_path(batch_id, scenario_id, step)
        if p.exists():
            return load_json(str(p))
        return None

    def has_checkpoint(self, batch_id: str, scenario_id: str, step: str) -> bool:
        return self._get_path(batch_id, scenario_id, step).exists()

    def clear_batch(self, batch_id: str) -> None:
        bdir = self.base_dir / batch_id
        if bdir.exists():
            for f in bdir.glob("*.json"):
                f.unlink()
