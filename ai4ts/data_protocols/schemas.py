"""
Pydantic models describing time series data structure at each stage of the pipeline.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator

from .enums import TaskType, FeatureType, ColumnRole, ScalerType, MissingValueStrategy, InputFormat
from .constants import (
    DEFAULT_TS_CONTEXT_LEN,
    DEFAULT_FORECASTING_HORIZON,
    DEFAULT_N_ANCHOR_POINTS,
    DEFAULT_ARTIFICIAL_MISSING_RATE,
    DEFAULT_CAST_DTYPE,
)


class ColumnDescriptor(BaseModel):
    """Describes a single column in the input dataset."""
    name: str
    role: ColumnRole
    dtype: FeatureType | None = None  # None for non-feature columns
    has_missing: bool = False
    missing_count: int = 0
    missing_rate: float = 0.0


class TimeSeriesMetadata(BaseModel):
    """Metadata describing a validated time series dataset."""
    n_samples: int = Field(..., description="Number of distinct time series samples")
    n_timesteps_per_sample: list[int] = Field(..., description="Timestep count per sample")
    n_features: int = Field(..., description="Number of numeric feature columns")
    n_total_columns: int = Field(..., description="Total column count in raw data")
    columns: list[ColumnDescriptor] = Field(default_factory=list)
    has_sample_id: bool = False
    has_timestamp: bool = False
    has_classification_target: bool = False
    has_missing_values: bool = False
    overall_missing_rate: float = 0.0
    sample_ids: list[Any] = Field(default_factory=list)
    numeric_feature_names: list[str] = Field(default_factory=list)
    string_feature_names: list[str] = Field(default_factory=list)
    integer_feature_names: list[str] = Field(default_factory=list)
    classification_target_names: list[str] = Field(default_factory=list)

    @property
    def min_timesteps(self) -> int:
        return min(self.n_timesteps_per_sample) if self.n_timesteps_per_sample else 0

    @property
    def max_timesteps(self) -> int:
        return max(self.n_timesteps_per_sample) if self.n_timesteps_per_sample else 0

    @property
    def is_single_sample(self) -> bool:
        return self.n_samples == 1


class ProcessingConfig(BaseModel):
    """Configuration for the data transformation pipeline."""
    ts_context_len: int = Field(default=DEFAULT_TS_CONTEXT_LEN, ge=2, description="Context window length")
    forecasting_horizon: int = Field(default=DEFAULT_FORECASTING_HORIZON, ge=1, description="Forecast timesteps")
    forecasting_features: list[str] | None = Field(default=None, description="Features to forecast; None=all numeric")
    n_anchor_points: int = Field(default=DEFAULT_N_ANCHOR_POINTS, ge=0, description="Anchor reference points")
    artificial_missing_rate: float = Field(default=DEFAULT_ARTIFICIAL_MISSING_RATE, ge=0.0, le=1.0)
    scaler_type: ScalerType = ScalerType.STANDARD
    missing_value_strategy: MissingValueStrategy = MissingValueStrategy.FORWARD_FILL
    cast_dtype: str = DEFAULT_CAST_DTYPE
    sliding_stride: int | None = Field(default=None, description="None=auto-compute based on data size")
    tasks: list[TaskType] = Field(default_factory=lambda: list(TaskType))

    @property
    def max_input_steps(self) -> int:
        """Total sequence length after anchor insertion."""
        return self.ts_context_len + self.n_anchor_points


class ScalerParams(BaseModel):
    """Serializable scaler parameters for cross-service transfer."""
    scaler_type: ScalerType
    feature_names: list[str]
    # For StandardScaler
    mean: list[float] | None = None
    std: list[float] | None = None
    # For MinMaxScaler
    data_min: list[float] | None = None
    data_max: list[float] | None = None
    scale: list[float] | None = None


class TensorDescriptor(BaseModel):
    """Describes the structure of processed tensor data."""
    shape: tuple[int, ...]  # (n_samples, max_input_steps, n_features)
    dtype: str = DEFAULT_CAST_DTYPE
    n_anchor_points: int = DEFAULT_N_ANCHOR_POINTS
    padding_lengths: list[int] = Field(default_factory=list, description="Padding length per sample")
    scaler_params: ScalerParams | None = None
    masks: list[str] = Field(default_factory=lambda: ["missing_mask", "padding_mask"])
    metadata: TimeSeriesMetadata | None = None
    config: ProcessingConfig | None = None


class ResultDescriptor(BaseModel):
    """Describes how to interpret and reconstruct model output."""
    task: TaskType
    output_format: InputFormat = InputFormat.DATAFRAME
    shape: tuple[int, ...] | None = None
    column_names: list[str] = Field(default_factory=list)
    inverse_scaled: bool = True
    scaler_params: ScalerParams | None = None
    metadata: TimeSeriesMetadata | None = None


class DataUploadManifest(BaseModel):
    """Manifest sent with data uploads to guide server-side processing."""
    input_format: InputFormat = InputFormat.CSV
    column_mappings: dict[str, str] | None = Field(default=None, description="Map user column names → standard names")
    processing_config: ProcessingConfig = Field(default_factory=ProcessingConfig)
    task_hints: list[TaskType] = Field(default_factory=list)
