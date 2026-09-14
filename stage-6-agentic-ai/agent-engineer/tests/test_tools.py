"""Tests for tool registry, allowlisting, and analytical adapters."""
import pytest
from tools.registry import ToolRegistry, get_default_registry
from tools.contracts import ToolRiskLevel


def test_default_registry_tools():
    registry = get_default_registry()
    tools = registry.list_tools()
    assert len(tools) >= 15
    assert "load_dataset" in registry.tools
    assert "eda_analysis" in registry.tools
    assert "root_cause_analysis" in registry.tools
    assert "train_model" in registry.tools


def test_tool_allowlisting():
    registry = get_default_registry()
    # Executing tool within allowed list
    res = registry.execute("load_dataset", {"file_path": "sales.csv"}, allowed_tools=["load_dataset"])
    assert res["status"] == "success"

    # Executing tool NOT in allowed list
    with pytest.raises(PermissionError):
        registry.execute("train_model", {}, allowed_tools=["load_dataset"])


def test_unknown_tool():
    registry = get_default_registry()
    with pytest.raises(ValueError):
        registry.execute("non_existent_tool", {})


def test_high_risk_tool_flag():
    registry = get_default_registry()
    train_meta = registry.get_metadata("train_model")
    assert train_meta.risk_level == ToolRiskLevel.HIGH

    load_meta = registry.get_metadata("load_dataset")
    assert load_meta.risk_level == ToolRiskLevel.LOW


def test_analytical_tool_executions(default_registry):
    # Test EDA
    eda_res = default_registry.execute("eda_analysis", {"target_col": "revenue"})
    assert "correlations" in eda_res
    assert eda_res["correlations"]["price_per_unit"] == 0.74

    # Test Anomaly Detection
    anom_res = default_registry.execute("detect_anomalies", {"column": "revenue"})
    assert anom_res["anomalies_detected"] == 14

    # Test Root Cause Analysis
    rca_res = default_registry.execute("root_cause_analysis", {})
    assert "ranked_causes" in rca_res
    assert len(rca_res["ranked_causes"]) >= 2
