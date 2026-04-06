"""
Data profiling utilities for time series datasets.

Generates a lightweight DataProfile JSON that agents can read to understand
dataset characteristics and make decisions about processing strategy, without
needing to load the actual data into their context window.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import COL_NAME_SAMPLE_ID, COL_NAME_TIMESTAMP, RESERVED_COLUMNS, CLAF_TARGET_SYMBOL
from .utils import separate_features


@dataclass
class DatasetStats:
    """Aggregate statistics of the dataset."""
    n_samples: int = 0
    n_features: int = 0
    total_rows: int = 0
    missing_rate: float = 0.0
    min_sample_length: int = 0
    max_sample_length: int = 0
    avg_sample_length: float = 0.0
    has_variable_length: bool = False


@dataclass
class SchemaMapping:
    """Maps column names to their roles in the data protocol."""
    sample_id_col: str | None = None
    timestamp_col: str | None = None
    feature_cols: list[str] = field(default_factory=list)
    label_col: str | None = None
    n_classes: int | None = None


@dataclass
class TimestampInfo:
    """Information about timestamp regularity."""
    has_timestamp: bool = False
    is_regular: bool = True
    median_step_seconds: float | None = None
    step_variance: float | None = None
    warning: str | None = None


@dataclass
class DataProfile:
    """Complete profile of a time series dataset for agent consumption.

    This lightweight JSON structure tells the agent everything it needs to know
    about the dataset without exposing individual sample data.
    """
    dataset_stats: DatasetStats = field(default_factory=DatasetStats)
    schema_mapping: SchemaMapping = field(default_factory=SchemaMapping)
    timestamp_info: TimestampInfo = field(default_factory=TimestampInfo)
    task_type: str | None = None
    split_info: dict[str, Any] | None = None
    sample_lengths: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a plain dict suitable for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DataProfile":
        """Reconstruct from a plain dict."""
        return cls(
            dataset_stats=DatasetStats(**d.get("dataset_stats", {})),
            schema_mapping=SchemaMapping(**d.get("schema_mapping", {})),
            timestamp_info=TimestampInfo(**d.get("timestamp_info", {})),
            task_type=d.get("task_type"),
            split_info=d.get("split_info"),
            sample_lengths=d.get("sample_lengths", []),
        )


def analyze_dataset(
    csv_path: str | Path,
    task_type: str | None = None,
) -> DataProfile:
    """Analyze a CSV file and produce a DataProfile.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.
    task_type : str or None
        Optional task type hint (imputation, classification, forecasting, clustering,
        anomaly_detection). If None, auto-detected from presence of CLAF_TARGET columns.

    Returns
    -------
    DataProfile
        Lightweight profile describing the dataset.
    """
    df = pd.read_csv(csv_path)

    # Detect column roles
    sample_id_col = COL_NAME_SAMPLE_ID if COL_NAME_SAMPLE_ID in df.columns else None
    timestamp_col = COL_NAME_TIMESTAMP if COL_NAME_TIMESTAMP in df.columns else None

    label_col = None
    for col in df.columns:
        if CLAF_TARGET_SYMBOL in col:
            label_col = col
            break

    reserved = RESERVED_COLUMNS | ({label_col} if label_col else set())
    feature_cols = [
        c for c in df.columns
        if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Compute per-sample lengths
    if sample_id_col and sample_id_col in df.columns:
        groups = df.groupby(sample_id_col, sort=True)
        lengths = groups.size().values.tolist()
    else:
        lengths = [len(df)]

    n_samples = len(lengths)
    min_len = int(min(lengths))
    max_len = int(max(lengths))
    avg_len = float(np.mean(lengths))
    has_variable_length = min_len != max_len

    # Missing rate on feature columns
    if feature_cols:
        feature_data = df[feature_cols]
        total_elements = feature_data.size
        nan_count = int(feature_data.isna().sum().sum())
        missing_rate = nan_count / total_elements if total_elements > 0 else 0.0
    else:
        missing_rate = 0.0

    # Classification info
    n_classes = None
    if label_col:
        n_classes = int(df[label_col].dropna().nunique())

    # Auto-detect task type
    if task_type is None:
        task_type = "classification" if label_col else "imputation"

    # Timestamp regularity check
    ts_info = TimestampInfo(has_timestamp=timestamp_col is not None)
    if timestamp_col and timestamp_col in df.columns:
        ts_info = _check_timestamp_regularity(df, sample_id_col, timestamp_col)

    return DataProfile(
        dataset_stats=DatasetStats(
            n_samples=n_samples,
            n_features=len(feature_cols),
            total_rows=len(df),
            missing_rate=round(missing_rate, 6),
            min_sample_length=min_len,
            max_sample_length=max_len,
            avg_sample_length=round(avg_len, 2),
            has_variable_length=has_variable_length,
        ),
        schema_mapping=SchemaMapping(
            sample_id_col=sample_id_col,
            timestamp_col=timestamp_col,
            feature_cols=feature_cols,
            label_col=label_col,
            n_classes=n_classes,
        ),
        timestamp_info=ts_info,
        task_type=task_type,
        sample_lengths=lengths,
    )


def _check_timestamp_regularity(
    df: pd.DataFrame,
    sample_id_col: str | None,
    timestamp_col: str,
) -> TimestampInfo:
    """Check whether timestamps have a regular frequency.

    If step variance > 0, the data has irregular sampling and the profile
    includes a warning message for the agent.
    """
    info = TimestampInfo(has_timestamp=True)

    try:
        ts = pd.to_datetime(df[timestamp_col])
    except Exception:
        info.warning = f"Could not parse '{timestamp_col}' as datetime."
        return info

    # Analyze first sample only for efficiency
    if sample_id_col and sample_id_col in df.columns:
        first_id = df[sample_id_col].iloc[0]
        ts = ts[df[sample_id_col] == first_id]

    if len(ts) < 2:
        return info

    diffs = ts.diff().dropna().dt.total_seconds()
    if len(diffs) == 0:
        return info

    median_step = float(diffs.median())
    variance = float(diffs.var())

    info.median_step_seconds = median_step
    info.step_variance = variance
    info.is_regular = variance < 1e-6 * (median_step ** 2) if median_step > 0 else True

    if not info.is_regular:
        info.warning = (
            f"Irregular sampling detected: median step={median_step:.1f}s, "
            f"variance={variance:.1f}. The data may need resampling before "
            f"model training. Proceed with caution or ask the user to preprocess."
        )

    return info
