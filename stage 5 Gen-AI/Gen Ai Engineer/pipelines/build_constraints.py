"""Pipeline to build constraint specifications."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constraints.constraint_builder import MasterConstraintBuilder

def run_build_constraints() -> dict:
    builder = MasterConstraintBuilder()
    return builder.build_and_save()

if __name__ == "__main__":
    run_build_constraints()
