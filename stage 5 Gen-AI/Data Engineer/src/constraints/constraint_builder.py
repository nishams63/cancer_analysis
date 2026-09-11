"""Master constraint specification builder."""
from pathlib import Path
from typing import Dict, Any
from .range_rules import RangeRulesBuilder
from .temporal_rules import TemporalRulesBuilder
from .cooccurrence_rules import CooccurrenceRulesBuilder
from .treatment_rules import TreatmentRulesBuilder
from ..utils.io import save_yaml
from ..utils.logging import get_logger

logger = get_logger("constraint_builder")

class MasterConstraintBuilder:
    def __init__(self, output_path: str | Path = "C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/constraint_spec.yaml"):
        self.output_path = Path(output_path)

    def build_and_save(self) -> Dict[str, Any]:
        logger.info("Building comprehensive constraint specifications...")
        ranges = RangeRulesBuilder().build()
        temporal = TemporalRulesBuilder().build()
        cooc = CooccurrenceRulesBuilder().build()
        treatment = TreatmentRulesBuilder().build()

        spec = {
            "version": "1.0.0",
            "metadata": {
                "generated_by": "Stage5 Data Engineer",
                "purpose": "Define permissible synthetic oncology parameter space for structured sampling"
            },
            "schema_constraints": ranges["schema_constraints"],
            "biomarker_ranges": ranges["biomarker_range_constraints"],
            "vital_signs": ranges["vital_sign_constraints"],
            "temporal_rules": temporal,
            "mutation_rules": cooc,
            "treatment_rules": treatment
        }

        save_yaml(spec, self.output_path)
        logger.info(f"Saved constraint specification to {self.output_path}")
        return spec
