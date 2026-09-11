from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from ..utils.logging import get_logger

logger = get_logger("DependencyValidator")

class DependencyValidator:
    def __init__(self, base_repo: Optional[Path] = None):
        if base_repo is None:
            self.stage5_root = Path(__file__).resolve().parent.parent.parent.parent
        else:
            self.stage5_root = Path(base_repo)

    def validate_all(self) -> Tuple[bool, List[str]]:
        missing = []

        # 1. Data Engineer dependencies
        de_dir = self.stage5_root / "Data Engineer"
        for req in ["data/processed/reference_distributions.parquet", "configs/constraint_spec.yaml", "configs/rare_combination_space.yaml"]:
            if not (de_dir / req).exists():
                missing.append(f"Data Engineer: {req}")

        # 2. EDA Engineer dependencies
        eda_dir = self.stage5_root / "Eda Engineer"
        for req in ["prompts/prompt_library.yaml", "prompts/prompt_drift_rules.yaml"]:
            if not (eda_dir / req).exists():
                missing.append(f"EDA Engineer: {req}")

        # 3. GenAI Engineer dependencies
        genai_dir = self.stage5_root / "Gen Ai Engineer"
        for req in ["retrieval/indexes/evidence_vector_index.json", "configs/genai_config.yaml"]:
            if not (genai_dir / req).exists():
                missing.append(f"GenAI Engineer: {req}")

        # 4. Evaluation Engineer dependencies
        eval_dir = self.stage5_root / "Evaluation Engineer"
        for req in ["configs/evaluation_config.yaml", "configs/failure_taxonomy.yaml"]:
            if not (eval_dir / req).exists():
                missing.append(f"Evaluation Engineer: {req}")

        is_valid = len(missing) == 0
        if not is_valid:
            logger.error(f"Missing required upstream dependencies: {missing}")
        else:
            logger.info("All upstream Stage 5 dependencies verified successfully.")

        return is_valid, missing
