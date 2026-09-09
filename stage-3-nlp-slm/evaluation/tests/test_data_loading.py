"""
Unit tests for data loading and schema validation in evaluation module.
"""

import pytest
import pandas as pd
from data_loader import (
    load_train_data,
    load_validation_data,
    load_locked_test_data,
    validate_schema,
    MANDATORY_COLUMNS
)


def test_load_train_data():
    df = load_train_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4261
    assert df["data_split"].iloc[0] == "TRAIN"


def test_load_validation_data():
    df = load_validation_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 909
    assert df["data_split"].iloc[0] == "VALIDATION"


def test_load_locked_test_data():
    df = load_locked_test_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 928
    assert df["data_split"].iloc[0] == "LOCKED_TEST"


def test_schema_validation_success():
    df = load_validation_data().head(10)
    # Should execute without error
    validate_schema(df, "TEST_SAMPLE")


def test_schema_validation_missing_column():
    df = load_validation_data().head(10).drop(columns=["urgency_level"])
    with pytest.raises(ValueError, match="Schema violation"):
        validate_schema(df, "BROKEN_SAMPLE")


def test_schema_validation_null_value():
    df = load_validation_data().head(10).copy()
    df.loc[0, "text"] = None
    with pytest.raises(ValueError, match="Null values detected"):
        validate_schema(df, "NULL_SAMPLE")
