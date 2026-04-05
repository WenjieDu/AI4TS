"""
Input data validation utilities for the TimeSeries AI data protocol.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import COL_NAME_SAMPLE_ID, COL_NAME_TIMESTAMP, RESERVED_COLUMNS
from .enums import ColumnRole, FeatureType, InputFormat
from .schemas import ColumnDescriptor, TimeSeriesMetadata, ProcessingConfig
from .utils import separate_features, get_feature_type, ensure_sample_id, ensure_timestamp


class ValidationError(Exception):
    """Raised when input data fails validation."""
    pass


def validate_csv_file(path: str | Path) -> pd.DataFrame:
    """
    Read and validate a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Parsed and validated DataFrame.

    Raises
    ------
    ValidationError
        If the file cannot be read or contains no data.
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"File not found: {path}")
    if path.suffix.lower() not in (".csv",):
        raise ValidationError(f"Unsupported file extension: {path.suffix}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValidationError(f"Failed to parse CSV file: {e}")

    if df.empty:
        raise ValidationError("CSV file contains no data rows.")
    if len(df.columns) < 1:
        raise ValidationError("CSV file must have at least one column.")

    return df


def validate_numpy_array(
    arr: np.ndarray,
    feature_names: list[str] | None = None,
    sample_ids: list | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate a numpy array and convert to DataFrame.

    Accepts either:
    - 2D array with shape (timesteps, features) — treated as a single sample
    - 3D array with shape (n_samples, timesteps, features) — multi-sample dataset

    NaN values in the input are preserved in the output DataFrame.

    Parameters
    ----------
    arr : np.ndarray
        Input array. Must be 2D or 3D.
    feature_names : list[str] or None
        Column names. If None, auto-generated as feat_0, feat_1, ...
    sample_ids : list or None
        Sample identifiers for 3D arrays. If None, uses 0, 1, 2, ...

    Returns
    -------
    tuple of (DataFrame, feature_names)
        Converted DataFrame and the feature names used.

    Raises
    ------
    ValidationError
        If array shape is invalid or data is empty.
    """
    if arr.ndim not in (2, 3):
        raise ValidationError(
            f"Expected 2D (timesteps, features) or 3D (samples, timesteps, features) array, "
            f"got {arr.ndim}D with shape {arr.shape}."
        )

    # Check for non-NaN numeric content (NaN itself is numeric/float)
    if not np.issubdtype(arr.dtype, np.number):
        raise ValidationError(f"Array must be numeric, got dtype {arr.dtype}.")

    if arr.ndim == 2:
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValidationError(f"Array must not be empty, got shape {arr.shape}.")

        n_features = arr.shape[1]
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            raise ValidationError(
                f"feature_names length ({len(feature_names)}) != array columns ({n_features})."
            )
        df = pd.DataFrame(arr, columns=feature_names)
        return df, feature_names

    # 3D array: (n_samples, timesteps, features)
    n_samples, n_timesteps, n_features = arr.shape
    if n_samples == 0 or n_timesteps == 0 or n_features == 0:
        raise ValidationError(f"Array must not be empty, got shape {arr.shape}.")

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValidationError(
            f"feature_names length ({len(feature_names)}) != array features ({n_features})."
        )

    if sample_ids is not None and len(sample_ids) != n_samples:
        raise ValidationError(
            f"sample_ids length ({len(sample_ids)}) != number of samples ({n_samples})."
        )

    frames = []
    for i in range(n_samples):
        sample_df = pd.DataFrame(arr[i], columns=feature_names)
        sample_df[COL_NAME_SAMPLE_ID] = sample_ids[i] if sample_ids is not None else i
        frames.append(sample_df)

    df = pd.concat(frames, ignore_index=True)
    return df, feature_names


def validate_dataframe(
    df: pd.DataFrame,
    config: ProcessingConfig | None = None,
) -> TimeSeriesMetadata:
    """
    Validate a DataFrame and extract time series metadata.

    Ensures the data has the required structure, identifies column roles,
    detects feature types, and gathers statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    config : ProcessingConfig or None
        Processing configuration for context-aware validation.

    Returns
    -------
    TimeSeriesMetadata
        Metadata describing the validated dataset.

    Raises
    ------
    ValidationError
        If data fails structural validation.
    """
    if df.empty:
        raise ValidationError("DataFrame is empty.")

    # Ensure required columns exist (auto-add if missing)
    df = ensure_sample_id(df)
    df = ensure_timestamp(df)

    # Separate features by type
    numeric_feats, string_feats, int_feats, claf_targets = separate_features(df)

    if len(numeric_feats) == 0:
        raise ValidationError("No numeric feature columns found. At least one is required.")

    # Build column descriptors
    columns = []
    for col in df.columns:
        if col == COL_NAME_SAMPLE_ID:
            columns.append(ColumnDescriptor(name=col, role=ColumnRole.SAMPLE_ID))
        elif col == COL_NAME_TIMESTAMP:
            columns.append(ColumnDescriptor(name=col, role=ColumnRole.TIMESTAMP))
        elif col in claf_targets:
            columns.append(ColumnDescriptor(
                name=col, role=ColumnRole.CLASSIFICATION_TARGET,
                dtype=get_feature_type(df[col]),
            ))
        elif col in RESERVED_COLUMNS:
            role = {
                "SAMPLE_LABEL": ColumnRole.SAMPLE_LABEL,
                "STEP_LABEL": ColumnRole.STEP_LABEL,
            }.get(col, ColumnRole.FEATURE)
            columns.append(ColumnDescriptor(name=col, role=role))
        else:
            ft = get_feature_type(df[col])
            missing_count = int(df[col].isna().sum())
            columns.append(ColumnDescriptor(
                name=col,
                role=ColumnRole.FEATURE,
                dtype=ft,
                has_missing=missing_count > 0,
                missing_count=missing_count,
                missing_rate=missing_count / len(df) if len(df) > 0 else 0.0,
            ))

    # Compute sample-level stats
    has_sample_id = COL_NAME_SAMPLE_ID in df.columns
    sample_groups = df.groupby(COL_NAME_SAMPLE_ID)
    sample_ids = list(sample_groups.groups.keys())
    n_timesteps_per_sample = [len(group) for _, group in sample_groups]

    # Missing value stats on numeric features
    numeric_df = df[numeric_feats]
    total_values = numeric_df.size
    total_missing = int(numeric_df.isna().sum().sum())
    overall_missing_rate = total_missing / total_values if total_values > 0 else 0.0

    # Validate against config if provided
    if config is not None:
        # For multi-sample datasets, each sample must fit in context window
        n_samples = len(sample_ids)
        if n_samples > 1:
            max_len = max(n_timesteps_per_sample)
            if max_len > config.ts_context_len:
                raise ValidationError(
                    f"Sample with {max_len} timesteps exceeds ts_context_len={config.ts_context_len}. "
                    "For multi-sample datasets, each sample must fit within the context window."
                )

    return TimeSeriesMetadata(
        n_samples=len(sample_ids),
        n_timesteps_per_sample=n_timesteps_per_sample,
        n_features=len(numeric_feats),
        n_total_columns=len(df.columns),
        columns=columns,
        has_sample_id=has_sample_id,
        has_timestamp=COL_NAME_TIMESTAMP in df.columns,
        has_classification_target=len(claf_targets) > 0,
        has_missing_values=total_missing > 0,
        overall_missing_rate=overall_missing_rate,
        sample_ids=sample_ids,
        numeric_feature_names=numeric_feats,
        string_feature_names=string_feats,
        integer_feature_names=int_feats,
        classification_target_names=claf_targets,
    )


def validate_processing_config(
    config: ProcessingConfig,
    metadata: TimeSeriesMetadata,
) -> ProcessingConfig:
    """
    Validate processing config against dataset metadata.

    Parameters
    ----------
    config : ProcessingConfig
        Configuration to validate.
    metadata : TimeSeriesMetadata
        Dataset metadata for context-aware checks.

    Returns
    -------
    ProcessingConfig
        Validated (possibly adjusted) config.

    Raises
    ------
    ValidationError
        If config is incompatible with the data.
    """
    if config.ts_context_len < 2:
        raise ValidationError("ts_context_len must be >= 2.")

    if config.forecasting_horizon < 1:
        raise ValidationError("forecasting_horizon must be >= 1.")

    # Check forecasting features exist
    if config.forecasting_features is not None:
        unknown = set(config.forecasting_features) - set(metadata.numeric_feature_names)
        if unknown:
            raise ValidationError(f"Unknown forecasting features: {unknown}")

    return config
