import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        base = Path(__file__).resolve().parent.parent.parent
        p = base / path
    if not p.exists():
        return {}
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def get_merged_config(config_dir: Optional[str] = None) -> Dict[str, Any]:
    base = Path(__file__).resolve().parent.parent.parent
    cdir = Path(config_dir) if config_dir else base / 'configs'
    
    integration_cfg = load_config(str(cdir / 'integration_config.yaml'))
    pipeline_cfg = load_config(str(cdir / 'pipeline_config.yaml'))
    storage_cfg = load_config(str(cdir / 'storage_config.yaml'))
    dashboard_cfg = load_config(str(cdir / 'dashboard_config.yaml'))
    
    return {
        'integration': integration_cfg,
        'pipeline': pipeline_cfg,
        'storage': storage_cfg,
        'dashboard': dashboard_cfg
    }
