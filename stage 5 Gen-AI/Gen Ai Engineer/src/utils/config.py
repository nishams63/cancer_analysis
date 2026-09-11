import os
import re
import yaml
from typing import Dict, Any
try:
    from dotenv import load_dotenv
    for env_f in [os.path.join("stage5", ".env"), ".env"]:
        if os.path.exists(env_f):
            load_dotenv(env_f)
except ImportError:
    pass


def resolve_env_vars(data: Any) -> Any:
    """Recursively resolve ${VAR:-default} and ${VAR} in strings."""
    if isinstance(data, dict):
        return {k: resolve_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_env_vars(item) for item in data]
    elif isinstance(data, str):
        pattern = re.compile(r'\$\{([^}:]+)(?::-([^}]+))?\}')
        match = pattern.search(data)
        if match:
            var_name = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else ""
            val = os.environ.get(var_name, default_val)
            return pattern.sub(val, data)
        return data
    return data


def load_genai_configs(config_dir: str = None) -> Dict[str, Any]:
    """Load genai, rag, and llm configs into a single unified configuration dictionary."""
    if config_dir is None:
        config_dir = os.path.join("stage5", "genai", "configs")

    configs = {}
    for cfg_name in ["genai_config.yaml", "rag_config.yaml", "llm_config.yaml"]:
        p = os.path.join(config_dir, cfg_name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
                key = cfg_name.replace(".yaml", "")
                configs[key] = resolve_env_vars(raw)
    return configs
