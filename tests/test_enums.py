"""
Tests for ai4ts.data_protocols.enums module.
"""

import pytest

from ai4ts.data_protocols.enums import (
    TaskType,
    FeatureType,
    ColumnRole,
    ScalerType,
    MissingValueStrategy,
    InputFormat,
)


class TestTaskType:
    """Tests for TaskType enum."""

    def test_all_task_types_exist(self):
        assert TaskType.IMPUTATION == "imputation"
        assert TaskType.FORECASTING == "forecasting"
        assert TaskType.CLASSIFICATION == "classification"
        assert TaskType.ANOMALY_DETECTION == "anomaly_detection"
        assert TaskType.CLUSTERING == "clustering"

    def test_task_type_count(self):
        assert len(TaskType) == 5

    def test_task_type_is_string(self):
        for t in TaskType:
            assert isinstance(t.value, str)

    def test_task_type_membership(self):
        assert "imputation" in [t.value for t in TaskType]


class TestFeatureType:
    """Tests for FeatureType enum."""

    def test_all_feature_types(self):
        assert FeatureType.NUMERIC == "numeric"
        assert FeatureType.STRING == "string"
        assert FeatureType.INTEGER == "integer"

    def test_feature_type_count(self):
        assert len(FeatureType) == 3


class TestColumnRole:
    """Tests for ColumnRole enum."""

    def test_all_roles(self):
        assert ColumnRole.FEATURE == "feature"
        assert ColumnRole.SAMPLE_ID == "sample_id"
        assert ColumnRole.TIMESTAMP == "timestamp"
        assert ColumnRole.CLASSIFICATION_TARGET == "classification_target"
        assert ColumnRole.SAMPLE_LABEL == "sample_label"
        assert ColumnRole.STEP_LABEL == "step_label"

    def test_role_count(self):
        assert len(ColumnRole) == 6


class TestScalerType:
    """Tests for ScalerType enum."""

    def test_all_scaler_types(self):
        assert ScalerType.STANDARD == "standard"
        assert ScalerType.MINMAX == "minmax"
        assert ScalerType.NONE == "none"


class TestMissingValueStrategy:
    """Tests for MissingValueStrategy enum."""

    def test_all_strategies(self):
        assert MissingValueStrategy.FORWARD_FILL == "forward_fill"
        assert MissingValueStrategy.ZERO_FILL == "zero_fill"
        assert MissingValueStrategy.MEAN_FILL == "mean_fill"
        assert MissingValueStrategy.NAN_KEEP == "nan_keep"

    def test_strategy_count(self):
        assert len(MissingValueStrategy) == 4


class TestInputFormat:
    """Tests for InputFormat enum."""

    def test_all_formats(self):
        assert InputFormat.CSV == "csv"
        assert InputFormat.NUMPY == "numpy"
        assert InputFormat.DATAFRAME == "dataframe"
