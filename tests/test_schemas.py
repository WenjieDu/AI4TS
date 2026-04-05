"""
Tests for ai4ts.data_protocols.schemas module.
"""

import pytest

from ai4ts.data_protocols.schemas import (
    ColumnDescriptor,
    TimeSeriesMetadata,
    ProcessingConfig,
    ScalerParams,
    TensorDescriptor,
    ResultDescriptor,
    DataUploadManifest,
)
from ai4ts.data_protocols.enums import (
    ColumnRole,
    FeatureType,
    ScalerType,
    TaskType,
    MissingValueStrategy,
    InputFormat,
)
from ai4ts.data_protocols.constants import (
    DEFAULT_TS_CONTEXT_LEN,
    DEFAULT_FORECASTING_HORIZON,
    DEFAULT_N_ANCHOR_POINTS,
    DEFAULT_ARTIFICIAL_MISSING_RATE,
)


class TestColumnDescriptor:
    """Tests for ColumnDescriptor schema."""

    def test_minimal_creation(self):
        cd = ColumnDescriptor(name="temp", role=ColumnRole.FEATURE)
        assert cd.name == "temp"
        assert cd.role == ColumnRole.FEATURE
        assert cd.dtype is None
        assert cd.has_missing is False
        assert cd.missing_count == 0
        assert cd.missing_rate == 0.0

    def test_full_creation(self):
        cd = ColumnDescriptor(
            name="pressure",
            role=ColumnRole.FEATURE,
            dtype=FeatureType.NUMERIC,
            has_missing=True,
            missing_count=42,
            missing_rate=0.15,
        )
        assert cd.has_missing is True
        assert cd.missing_count == 42
        assert cd.missing_rate == pytest.approx(0.15)

    def test_sample_id_role(self):
        cd = ColumnDescriptor(name="SAMPLE_ID", role=ColumnRole.SAMPLE_ID)
        assert cd.role == ColumnRole.SAMPLE_ID


class TestTimeSeriesMetadata:
    """Tests for TimeSeriesMetadata schema."""

    def test_minimal_creation(self):
        meta = TimeSeriesMetadata(
            n_samples=10,
            n_timesteps_per_sample=[50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
            n_features=5,
            n_total_columns=8,
        )
        assert meta.n_samples == 10
        assert meta.n_features == 5
        assert meta.is_single_sample is False
        assert meta.min_timesteps == 50
        assert meta.max_timesteps == 50

    def test_single_sample(self):
        meta = TimeSeriesMetadata(
            n_samples=1,
            n_timesteps_per_sample=[200],
            n_features=3,
            n_total_columns=5,
        )
        assert meta.is_single_sample is True
        assert meta.min_timesteps == 200
        assert meta.max_timesteps == 200

    def test_variable_length_samples(self):
        meta = TimeSeriesMetadata(
            n_samples=3,
            n_timesteps_per_sample=[30, 50, 40],
            n_features=2,
            n_total_columns=4,
        )
        assert meta.min_timesteps == 30
        assert meta.max_timesteps == 50

    def test_empty_timesteps(self):
        meta = TimeSeriesMetadata(
            n_samples=0,
            n_timesteps_per_sample=[],
            n_features=3,
            n_total_columns=3,
        )
        assert meta.min_timesteps == 0
        assert meta.max_timesteps == 0

    def test_missing_value_tracking(self):
        meta = TimeSeriesMetadata(
            n_samples=1,
            n_timesteps_per_sample=[100],
            n_features=5,
            n_total_columns=7,
            has_missing_values=True,
            overall_missing_rate=0.15,
        )
        assert meta.has_missing_values is True
        assert meta.overall_missing_rate == pytest.approx(0.15)


class TestProcessingConfig:
    """Tests for ProcessingConfig schema."""

    def test_defaults(self):
        config = ProcessingConfig()
        assert config.ts_context_len == DEFAULT_TS_CONTEXT_LEN
        assert config.forecasting_horizon == DEFAULT_FORECASTING_HORIZON
        assert config.n_anchor_points == DEFAULT_N_ANCHOR_POINTS
        assert config.artificial_missing_rate == DEFAULT_ARTIFICIAL_MISSING_RATE
        assert config.scaler_type == ScalerType.STANDARD
        assert config.missing_value_strategy == MissingValueStrategy.FORWARD_FILL

    def test_max_input_steps(self):
        config = ProcessingConfig(ts_context_len=64, n_anchor_points=3)
        assert config.max_input_steps == 67

    def test_nan_keep_strategy(self):
        config = ProcessingConfig(missing_value_strategy=MissingValueStrategy.NAN_KEEP)
        assert config.missing_value_strategy == MissingValueStrategy.NAN_KEEP

    def test_custom_config(self):
        config = ProcessingConfig(
            ts_context_len=32,
            forecasting_horizon=5,
            n_anchor_points=0,
            scaler_type=ScalerType.MINMAX,
        )
        assert config.ts_context_len == 32
        assert config.max_input_steps == 32


class TestScalerParams:
    """Tests for ScalerParams schema."""

    def test_standard_scaler_params(self):
        params = ScalerParams(
            scaler_type=ScalerType.STANDARD,
            feature_names=["a", "b"],
            mean=[0.5, 1.0],
            std=[0.1, 0.2],
        )
        assert params.mean == [0.5, 1.0]
        assert params.std == [0.1, 0.2]
        assert params.data_min is None

    def test_minmax_scaler_params(self):
        params = ScalerParams(
            scaler_type=ScalerType.MINMAX,
            feature_names=["x"],
            data_min=[0.0],
            data_max=[1.0],
            scale=[1.0],
        )
        assert params.data_min == [0.0]
        assert params.data_max == [1.0]


class TestTensorDescriptor:
    """Tests for TensorDescriptor schema."""

    def test_creation(self):
        td = TensorDescriptor(shape=(10, 130, 5))
        assert td.shape == (10, 130, 5)
        assert td.dtype == "float32"
        assert "missing_mask" in td.masks
        assert "padding_mask" in td.masks


class TestResultDescriptor:
    """Tests for ResultDescriptor schema."""

    def test_creation(self):
        rd = ResultDescriptor(task=TaskType.IMPUTATION)
        assert rd.task == TaskType.IMPUTATION
        assert rd.output_format == InputFormat.DATAFRAME
        assert rd.inverse_scaled is True


class TestDataUploadManifest:
    """Tests for DataUploadManifest schema."""

    def test_defaults(self):
        manifest = DataUploadManifest()
        assert manifest.input_format == InputFormat.CSV
        assert manifest.column_mappings is None
        assert isinstance(manifest.processing_config, ProcessingConfig)

    def test_with_column_mappings(self):
        manifest = DataUploadManifest(
            column_mappings={"time": "TIMESTAMP", "id": "SAMPLE_ID"},
            task_hints=[TaskType.IMPUTATION],
        )
        assert manifest.column_mappings["time"] == "TIMESTAMP"
        assert TaskType.IMPUTATION in manifest.task_hints
