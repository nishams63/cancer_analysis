"""Tool Registry enforcing strict allowlisting and security."""
from __future__ import annotations
from typing import Callable, Dict, Any, List, Optional
from .contracts import ToolMetadata, ToolRiskLevel
from . import adapters


class ToolRegistry:
    """Authoritative registry mapping tool names to verified callable functions."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._metadata: Dict[str, ToolMetadata] = {}

    @property
    def tools(self) -> Dict[str, Callable]:
        return self._tools

    def register(self, name: str, fn: Callable, metadata: Optional[ToolMetadata] = None) -> None:
        """Register a tool function with optional metadata."""
        self._tools[name] = fn
        self._metadata[name] = metadata or ToolMetadata(
            name=name,
            description=fn.__doc__ or f"Analytical tool '{name}'",
            risk_level=ToolRiskLevel.LOW,
        )

    def get(self, name: str) -> Optional[Callable]:
        """Retrieve callable by name."""
        return self._tools.get(name)

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        return self._metadata.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": m.name, "description": m.description, "risk_level": m.risk_level.value}
            for m in self._metadata.values()
        ]

    def execute(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        allowed_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute a registered tool under security and allowlist constraints."""
        if allowed_tools is not None and tool_name not in allowed_tools:
            raise PermissionError(
                f"Security Violation: Tool '{tool_name}' is not permitted for the active task. "
                f"Permitted tools: {allowed_tools}"
            )

        fn = self.get(tool_name)
        if not fn:
            raise ValueError(f"Unknown tool: '{tool_name}'. Tool must be explicitly registered.")

        try:
            result = fn(**inputs)
            if not isinstance(result, dict):
                return {"status": "success", "result": result}
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def to_openai_specs(self, allowed_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        specs = []
        for name, meta in self._metadata.items():
            if allowed_tools and name not in allowed_tools:
                continue
            specs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": meta.description,
                    "parameters": meta.input_schema or {
                        "type": "object",
                        "properties": {},
                    },
                },
            })
        return specs


def get_default_registry() -> ToolRegistry:
    """Factory instantiating the default AADA analytical tool registry."""
    reg = ToolRegistry()
    reg.register("load_dataset", adapters.load_dataset)
    reg.register("profile_dataset", adapters.profile_dataset)
    reg.register("validate_data_quality", adapters.validate_data_quality)
    reg.register("clean_dataset", adapters.clean_dataset)
    reg.register("eda_analysis", adapters.eda_analysis)
    reg.register("statistical_analysis", adapters.statistical_analysis)
    reg.register("detect_anomalies", adapters.detect_anomalies)
    reg.register("segment_customers", adapters.segment_customers)
    reg.register(
        "train_model",
        adapters.train_model,
        metadata=ToolMetadata(name="train_model", description="Train predictive model", risk_level=ToolRiskLevel.HIGH),
    )
    reg.register("root_cause_analysis", adapters.root_cause_analysis)
    reg.register("generate_visualization", adapters.generate_visualization)
    reg.register("generate_recommendation", adapters.generate_recommendation)
    reg.register("generate_report", adapters.generate_report)
    reg.register("retrieve_knowledge", adapters.retrieve_knowledge)
    reg.register(
        "human_review",
        adapters.human_review,
        metadata=ToolMetadata(name="human_review", description="Escalate for human review", risk_level=ToolRiskLevel.HIGH),
    )
    return reg
