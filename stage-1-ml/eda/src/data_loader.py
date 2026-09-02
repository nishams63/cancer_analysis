import pandas as pd
from pathlib import Path
import sys

def get_project_root() -> Path:
    """Returns project root path by searching upwards for 'stage-1-ml'."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / 'stage-1-ml').exists():
            return parent
        if parent.name == 'stage-1-ml':
            return parent.parent
    # Fallback to current directory if not found
    return Path.cwd()

def load_data() -> pd.DataFrame:
    """Loads the master patient dataset."""
    root = get_project_root()
    data_path = root / 'stage-1-ml' / 'data-engineering' / 'data' / 'processed' / 'master_patient_dataset.csv'
    
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    
    print(f"Dataset successfully loaded from: {data_path}")
    print(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df
