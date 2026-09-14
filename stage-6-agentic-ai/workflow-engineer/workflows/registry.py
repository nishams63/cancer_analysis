"""Workflow Template Registry and Loader."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from schemas.workflow import Workflow
from engine.workflow_validator import validate_workflow


class WorkflowRegistry:
    """Manages standard workflow templates and catalog discovery."""

    def __init__(self, templates_dir: Optional[Path | str] = None):
        if templates_dir is None:
            self.templates_dir = Path(__file__).resolve().parent / "templates"
        else:
            self.templates_dir = Path(templates_dir)
        self._cache: Dict[str, Workflow] = {}
        self._intent_map: Dict[str, str] = {
            "revenue_decline": "revenue_decline",
            "sales_decline": "revenue_decline",
            "customer_analysis": "customer_analysis",
            "customer_churn": "customer_analysis",
            "anomaly_investigation": "anomaly_investigation",
            "outlier_detection": "anomaly_investigation",
            "predictive_analysis": "predictive_analysis",
            "model_training": "predictive_analysis",
            "dataset_analysis": "dataset_analysis",
            "general_eda": "dataset_analysis",
        }
        self.load_all()

    def load_all(self) -> int:
        """Scan and parse all template JSON files into cache."""
        self._cache.clear()
        if not self.templates_dir.exists():
            return 0

        for file_path in sorted(self.templates_dir.glob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                wf = Workflow.from_dict(data)
                self._cache[file_path.stem] = wf
                self._cache[wf.workflow_id] = wf
            except Exception as e:
                print(f"[ERROR] Failed to load template {file_path}: {e}")

        return len(self._cache)

    def get_template(self, template_key_or_id: str) -> Optional[Workflow]:
        """Retrieve workflow template by filename stem or workflow_id."""
        return self._cache.get(template_key_or_id)

    def get_template_for_intent(self, intent_type: str) -> Optional[Workflow]:
        """Retrieve standard workflow template matching an analytical intent."""
        key = self._intent_map.get(intent_type.lower().strip())
        if not key:
            key = "dataset_analysis"
        return self.get_template(key)

    def list_templates(self) -> List[Dict[str, Any]]:
        """Return catalog of all unique loaded workflow templates."""
        seen_ids = set()
        catalog = []
        for key, wf in self._cache.items():
            if wf.workflow_id in seen_ids:
                continue
            seen_ids.add(wf.workflow_id)
            val = validate_workflow(wf)
            catalog.append({
                "workflow_id": wf.workflow_id,
                "name": wf.name,
                "version": wf.version,
                "goal": wf.goal,
                "task_count": len(wf.tasks),
                "entry_task": wf.entry_task,
                "terminal_tasks": wf.terminal_tasks,
                "is_valid": val.valid,
            })
        return catalog
