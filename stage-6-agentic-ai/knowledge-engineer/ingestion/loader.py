"""Knowledge Base File Loader."""
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any


class KnowledgeLoader:
    """Loads structured JSON knowledge definition files from directory hierarchy."""

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)

    def load_all_files(self) -> List[Tuple[Path, Dict[str, Any]]]:
        """Scan directory recursively and load all valid .json files."""
        loaded = []
        if not self.base_dir.exists():
            return loaded

        for file_path in sorted(self.base_dir.rglob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    loaded.append((file_path, data))
            except Exception as e:
                print(f"[ERROR] Failed to load {file_path}: {e}")
        return loaded
