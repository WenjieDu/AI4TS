"""
Data protocols for the TimeSeries AI platform.

This subpackage defines the data formats, validation rules, and conversion utilities
used across all TimeSeries AI components (SDK, backend, model library).
"""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from .constants import (
    COL_NAME_SAMPLE_ID,
    COL_NAME_SAMPLE_LABEL,
    COL_NAME_STEP_LABEL,
    COL_NAME_TIMESTAMP,
    CLAF_TARGET_SYMBOL,
    RESERVED_COLUMNS,
    DEFAULT_TS_CONTEXT_LEN,
    DEFAULT_FORECASTING_HORIZON,
    DEFAULT_N_ANCHOR_POINTS,
    DEFAULT_ARTIFICIAL_MISSING_RATE,
    AT_LEAST_SAMPLE_NUM,
    SUPPORTED_FILE_TYPES,
    SUPPORTED_FILE_EXTENSIONS,
    TEXT_RESPONSE_PREFIX,
    JSON_RESPONSE_PREFIX,
)

from .enums import (
    TaskType,
    FeatureType,
    ColumnRole,
    ScalerType,
    MissingValueStrategy,
    InputFormat,
)

from .schemas import (
    ColumnDescriptor,
    TimeSeriesMetadata,
    ProcessingConfig,
    ScalerParams,
    TensorDescriptor,
    ResultDescriptor,
    DataUploadManifest,
)

from .validation import (
    ValidationError,
    validate_csv_file,
    validate_numpy_array,
    validate_dataframe,
    validate_processing_config,
)

from .conversion import (
    FeatureSeparator,
    TimeSeriesConverter,
)

from .utils import (
    separate_features,
    compute_sliding_stride,
    compute_n_samples,
    forward_fill,
    create_missing_mask,
    fill_and_get_mask,
    insert_anchor_points,
    pad_sequence,
    create_padding_mask,
    inject_artificial_missing,
)

from .profile import (
    DataProfile,
    DatasetStats,
    SchemaMapping,
    TimestampInfo,
    analyze_dataset,
)

from .registry import (
    WindowEntry,
    WindowRegistry,
)

from .pipeline import (
    prepare_for_pypots,
    split_by_sample_id,
    split_source_rows,
    reconstruct_from_predictions,
)

__all__ = [
    # Constants
    "COL_NAME_SAMPLE_ID", "COL_NAME_SAMPLE_LABEL", "COL_NAME_STEP_LABEL",
    "COL_NAME_TIMESTAMP", "CLAF_TARGET_SYMBOL", "RESERVED_COLUMNS",
    "DEFAULT_TS_CONTEXT_LEN", "DEFAULT_FORECASTING_HORIZON",
    "DEFAULT_N_ANCHOR_POINTS", "DEFAULT_ARTIFICIAL_MISSING_RATE",
    "AT_LEAST_SAMPLE_NUM", "SUPPORTED_FILE_TYPES", "SUPPORTED_FILE_EXTENSIONS",
    "TEXT_RESPONSE_PREFIX", "JSON_RESPONSE_PREFIX",
    # Enums
    "TaskType", "FeatureType", "ColumnRole", "ScalerType",
    "MissingValueStrategy", "InputFormat",
    # Schemas
    "ColumnDescriptor", "TimeSeriesMetadata", "ProcessingConfig",
    "ScalerParams", "TensorDescriptor", "ResultDescriptor", "DataUploadManifest",
    # Validation
    "ValidationError", "validate_csv_file", "validate_numpy_array",
    "validate_dataframe", "validate_processing_config",
    # Conversion
    "FeatureSeparator", "TimeSeriesConverter",
    # Utils
    "separate_features", "compute_sliding_stride", "compute_n_samples",
    "forward_fill", "create_missing_mask", "fill_and_get_mask",
    "insert_anchor_points", "pad_sequence", "create_padding_mask",
    "inject_artificial_missing",
    # Profile
    "DataProfile", "DatasetStats", "SchemaMapping", "TimestampInfo",
    "analyze_dataset",
    # Registry
    "WindowEntry", "WindowRegistry",
    # Pipeline
    "prepare_for_pypots", "split_by_sample_id", "split_source_rows", "reconstruct_from_predictions",
]

# Provider-free source-bound planning; preparation remains opt-in.
from .ingestion import (
    inspect_csv, create_ingestion_plan, validate_ingestion_plan,
    render_ingestion_preview, verify_prepared_plan,
)
__all__ += ["inspect_csv", "create_ingestion_plan", "validate_ingestion_plan", "render_ingestion_preview", "verify_prepared_plan"]

from .temporal_preprocessing import (
    create_temporal_plan, validate_temporal_plan, run_temporal_preprocessing, inspect_temporal_result,
)
__all__ += ["create_temporal_plan", "validate_temporal_plan", "run_temporal_preprocessing", "inspect_temporal_result"]

from .reshape import create_reshape_plan, validate_reshape_plan, run_reshape, inspect_reshape_result
__all__ += ["create_reshape_plan", "validate_reshape_plan", "run_reshape", "inspect_reshape_result"]

from .time_basis import preview_time_basis
__all__ += ["preview_time_basis"]

from .merge import create_merge_plan, validate_merge_plan, run_merge, inspect_merge_result
__all__ += ["create_merge_plan", "validate_merge_plan", "run_merge", "inspect_merge_result"]

from .covariates import export_tsfm_frames, validate_future_covariates
__all__ += ["export_tsfm_frames", "validate_future_covariates"]
