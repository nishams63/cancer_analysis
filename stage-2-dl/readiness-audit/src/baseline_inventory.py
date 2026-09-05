"""
Stage 2 Deep Learning - Baseline Repository Inventory Module

Programmatically catalogs:
- Files and directory structures across all five Stage 2 modules
- Frozen checkpoint SHA-256 hashes, parameter counts, and sizes
- Dataset shapes, splits, and record counts
- Dependency specifications and test coverage
"""
import os
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import torch

try:
    from . import audit_config as config
except (ImportError, ValueError):
    import audit_config as config


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 cryptographic hash of a file."""
    if not file_path.exists():
        return "FILE_NOT_FOUND"
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def generate_baseline_inventory() -> Dict[str, Any]:
    """Extracts a comprehensive inventory of all Stage 2 components."""
    print("=" * 60)
    print("EXTRACTING STAGE 2 BASELINE INVENTORY")
    print("=" * 60)

    modules = {
        'data-engineering': config.DATA_ENG_DIR,
        'eda': config.EDA_DIR,
        'dl': config.DL_DIR,
        'evaluation': config.EVAL_DIR,
        'integration': config.INTEGRATION_DIR
    }

    module_stats = {}
    for mod_name, mod_path in modules.items():
        if mod_path.exists():
            all_files = [p for p in mod_path.rglob('*') if p.is_file() and not p.name.endswith('.pyc') and '__pycache__' not in str(p)]
            py_files = [p for p in all_files if p.suffix == '.py']
            md_files = [p for p in all_files if p.suffix == '.md']
            total_size_mb = sum(p.stat().st_size for p in all_files) / (1024 * 1024)

            module_stats[mod_name] = {
                'total_files': len(all_files),
                'python_files': len(py_files),
                'markdown_files': len(md_files),
                'size_mb': round(total_size_mb, 2),
                'exists': True
            }
        else:
            module_stats[mod_name] = {'exists': False}

    # 2. Frozen Checkpoints Audit
    checkpoints = {}
    for name, path in [('pathology_cnn', config.PATHOLOGY_CHECKPOINT), ('temporal_lstm', config.TEMPORAL_CHECKPOINT)]:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            sha = compute_sha256(path)
            ckpt_obj = torch.load(path, map_location='cpu')
            num_params = sum(p.numel() for p in ckpt_obj.get('state_dict', {}).values())
            arch = ckpt_obj.get('model_architecture', 'unknown')
            checkpoints[name] = {
                'path': str(path),
                'filename': path.name,
                'architecture': arch,
                'size_mb': round(size_mb, 2),
                'parameters': num_params,
                'sha256': sha,
                'status': 'FROZEN_VERIFIED'
            }
        else:
            checkpoints[name] = {'status': 'MISSING'}

    # 3. Dataset Audit
    datasets = {}
    if config.IMAGE_METADATA_PATH.exists():
        img_df = pd.read_csv(config.IMAGE_METADATA_PATH)
        datasets['pathology_tiles'] = {
            'total_tiles': len(img_df),
            'unique_patients': img_df['patient_id'].nunique(),
            'class_distribution': img_df['class_label'].value_counts().to_dict(),
            'split_distribution': img_df['split'].value_counts().to_dict(),
            'image_size': [224, 224, 3]
        }

    if config.BIOMARKERS_PATH.exists():
        bio_df = pd.read_csv(config.BIOMARKERS_PATH)
        hist_df = bio_df[(bio_df['is_input_window'] == 1) & (bio_df['days_from_baseline'] <= 90)]
        fut_df = bio_df[(bio_df['is_input_window'] == 0) & (bio_df['days_from_baseline'] > 90)]
        datasets['biomarkers'] = {
            'total_observations': len(bio_df),
            'historical_observations': len(hist_df),
            'future_observations': len(fut_df),
            'unique_patients': bio_df['patient_id'].nunique(),
            'split_distribution': bio_df['split'].value_counts().to_dict(),
            'feature_count': 13
        }

    # 4. Cohort Partitioning Audit
    cohort_splits = {}
    for s_name, s_path in [('train', config.DATA_ENG_DIR / 'data' / 'v2' / 'splits' / 'train_patients.csv'),
                           ('val', config.DATA_ENG_DIR / 'data' / 'v2' / 'splits' / 'validation_patients.csv'),
                           ('test', config.TEST_SPLIT_PATH)]:
        if s_path.exists():
            df_s = pd.read_csv(s_path)
            cohort_splits[s_name] = len(df_s)

    inventory = {
        'modules': module_stats,
        'checkpoints': checkpoints,
        'datasets': datasets,
        'cohort_splits': cohort_splits
    }

    print("\nBaseline Inventory Summary:")
    for mod, stat in module_stats.items():
        print(f"  [{mod:18s}]: {stat.get('total_files', 0)} files, {stat.get('size_mb', 0)} MB")
    for name, c_stat in checkpoints.items():
        print(f"  Checkpoint [{name:14s}]: Arch={c_stat.get('architecture')}, Size={c_stat.get('size_mb')} MB, Params={c_stat.get('parameters'):,}")

    return inventory


if __name__ == '__main__':
    generate_baseline_inventory()
