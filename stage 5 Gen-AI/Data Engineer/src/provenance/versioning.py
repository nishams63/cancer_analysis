"""Version management and timestamp stamping."""
from datetime import datetime, timezone
from typing import Dict, Any

class VersionManager:
    DEFAULT_VERSION = "1.0.0"

    @staticmethod
    def get_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def stamp_metadata(meta: Dict[str, Any], version: str = DEFAULT_VERSION) -> Dict[str, Any]:
        stamped = meta.copy()
        stamped["version"] = version
        stamped["created_at_utc"] = VersionManager.get_timestamp()
        return stamped
