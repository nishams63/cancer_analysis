"""Pipeline to build rare combination space."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rare_space.rare_space_builder import RareSpaceBuilder
from src.utils.io import read_parquet

def run_build_rare_space() -> dict:
    cooc_path = Path("stage5/data/processed/mutation_cooccurrence.parquet")
    if not cooc_path.exists():
        from pipelines.build_reference_data import run_build_reference_data
        run_build_reference_data()
    df_cooc = read_parquet(cooc_path)
    builder = RareSpaceBuilder()
    return builder.build_and_save(df_cooc)

if __name__ == "__main__":
    run_build_rare_space()
