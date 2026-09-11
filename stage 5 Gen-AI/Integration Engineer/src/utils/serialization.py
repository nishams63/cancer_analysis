import json
from pathlib import Path
from typing import Any, Dict
import numpy as np

class SafeJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif hasattr(o, 'to_dict'):
            return o.to_dict()
        elif hasattr(o, '__dict__'):
            return o.__dict__
        elif hasattr(o, 'isoformat'):
            return o.isoformat()
        return str(o)

def json_dumps(obj: Any, indent: int = 2) -> str:
    return json.dumps(obj, cls=SafeJSONEncoder, indent=indent, ensure_ascii=False)

def json_loads(s: str) -> Any:
    return json.loads(s)

def save_json(data: Any, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=SafeJSONEncoder, indent=2, ensure_ascii=False)

def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def safe_to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    if hasattr(obj, '__dict__'):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
    return {'value': str(obj)}
