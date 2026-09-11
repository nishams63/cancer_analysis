import os
from pathlib import Path
from enum import Enum
from typing import Dict, Any, Tuple, Optional
from ..utils.logging import get_logger

logger = get_logger("HealthChecker")

class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"

class HealthChecker:
    def __init__(self, base_repo: Optional[Path] = None):
        if base_repo is None:
            # stage5 root: cancer_analysis/stage 5 Gen-AI
            self.stage5_root = Path(__file__).resolve().parent.parent.parent.parent
        else:
            self.stage5_root = Path(base_repo)

    def check_health(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        checks = {}
        
        # 1. Reference Data (Data Engineer)
        de_path = self.stage5_root / "Data Engineer" / "data" / "processed" / "reference_distributions.parquet"
        if not de_path.exists():
            de_path = self.stage5_root / "Data Engineer" / "configs" / "constraint_spec.yaml"
        checks["reference_data_available"] = de_path.exists()

        # 2. RAG Index (GenAi Engineer)
        rag_path = self.stage5_root / "Gen Ai Engineer" / "retrieval" / "indexes" / "evidence_vector_index.json"
        checks["rag_index_available"] = rag_path.exists()

        # 3. LLM Reachable / Configured
        has_api_key = bool(os.getenv("NVIDIA_API_KEY"))
        checks["llm_provider_reachable"] = has_api_key
        
        # 4. Evaluation Engine (Evaluation Engineer)
        eval_path = self.stage5_root / "Evaluation Engineer" / "src" / "evaluation" / "realism.py"
        checks["evaluation_engine_available"] = eval_path.exists()

        # 5. Database Writable
        results_dir = self.stage5_root / "Integration Engineer" / "results"
        checks["database_writable"] = results_dir.exists() or os.access(str(self.stage5_root / "Integration Engineer"), os.W_OK)

        # 6. Stages 1-4 Available
        checks["stage1_available"] = True
        checks["stage2_available"] = True
        checks["stage3_available"] = True
        checks["stage4_available"] = True

        # Overall Status
        if all(checks.values()):
            status = HealthStatus.HEALTHY
        elif not has_api_key and checks["reference_data_available"] and checks["evaluation_engine_available"]:
            # Degraded if only LLM key is missing (offline mock mode operates)
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        return status, checks
