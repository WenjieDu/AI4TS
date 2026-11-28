"""
Data profiling utilities for time series datasets.

Generates a lightweight DataProfile JSON that agents can read to understand
dataset characteristics and make decisions about processing strategy, without
needing to load the actual data into their context window.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import COL_NAME_SAMPLE_ID, COL_NAME_TIMESTAMP, RESERVED_COLUMNS, CLAF_TARGET_SYMBOL


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
    per_feature_missing_rate: dict[str, float] = field(default_factory=dict)
    per_feature_value_domain: dict[str, dict[str, Any]] = field(default_factory=dict)


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
    about the dataset without exposing individual feature values.
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


def _parse_feature_values(df, csv_path, roles, features, categorical):
    """Preserve literal IDs and labels while retaining the numeric CSV defaults."""
    columns = [column for column in features if column in categorical]
    strings = roles + columns
    if strings:
        df[strings] = pd.read_csv(
            csv_path, usecols=strings, dtype=str, keep_default_na=False, engine="python" if columns else "c"
        )[strings]
    if columns:
        df[columns] = df[columns].replace("", np.nan)
    numeric = [column for column in features if column not in categorical]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="raise")
    return numeric


def _feature_value_domain(values: np.ndarray, entity_rows: dict[str | None, np.ndarray]) -> dict[str, Any]:
    finite = np.isfinite(values)
    observed = values[finite]
    if not observed.size:
        return {
            "nonnegative": None,
            "integer_valued": None,
            "constant": None,
            "one_signed": None,
            "leading_missing": None,
            "longest_gap": None,
            "reason": "no_finite_observations",
        }

    # Bounded summary: profiles must stay small for agents even with many entities.
    leading_missing = {"length": 0, "entity": None, "n_entities": 0}
    longest_gap = {"length": 0, "entity": None}
    for entity, rows in entity_rows.items():
        positions = np.flatnonzero(finite[rows])
        lead = int(positions[0]) if positions.size else len(rows)
        leading_missing["n_entities"] += lead > 0
        if lead > leading_missing["length"]:
            leading_missing.update(length=lead, entity=entity)
        # Only runs bounded by two finite observations count as internal gaps.
        gap_length = int(np.diff(positions).max(initial=1)) - 1
        if gap_length > longest_gap["length"]:
            longest_gap = {"length": gap_length, "entity": entity}

    smallest, largest = observed.min(), observed.max()
    return {
        "nonnegative": bool(smallest >= 0),
        "integer_valued": values.dtype.kind in "biu" or bool((observed == np.trunc(observed)).all()),
        "constant": bool(smallest == largest),
        "one_signed": bool(smallest >= 0 or largest <= 0),
        "leading_missing": leading_missing,
        "longest_gap": longest_gap,
    }


def analyze_dataset(
    csv_path: str | Path,
    task_type: str | None = None,
    schema_mapping: SchemaMapping | dict | None = None,
    *,
    categorical_columns: tuple[str, ...] = (),
) -> DataProfile:
    """Analyze a CSV file and produce a DataProfile.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.
    task_type : str or None
        Optional task type hint (imputation, classification, forecasting, clustering,
        anomaly_detection). If None, auto-detected from presence of CLAF_TARGET columns.
    schema_mapping : SchemaMapping or dict or None
        Confirmed column roles.
    categorical_columns : tuple of str
        Explicit categorical features. Keep literal labels and missing cells.

    Returns
    -------
    DataProfile
        Lightweight profile describing the dataset.
    """
    # Check raw names before pandas silently renames duplicate or empty headers.
    with open(csv_path, encoding="utf-8-sig", newline="") as source:
        columns = next(csv.reader(source), [])
    if not columns or len(set(columns)) != len(columns) or any(not name.strip() for name in columns):
        raise ValueError("CSV must have unique nonempty columns")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV must contain data")

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
        if c not in reserved and (c in categorical_columns or pd.api.types.is_numeric_dtype(df[c]))
    ]
    if schema_mapping is not None:
        schema = SchemaMapping(**schema_mapping) if isinstance(schema_mapping, dict) else schema_mapping
        sample_id_col, timestamp_col, label_col = schema.sample_id_col, schema.timestamp_col, schema.label_col
        feature_cols = list(schema.feature_cols)
    roles = [c for c in (sample_id_col, timestamp_col, label_col) if c is not None]
    selected = roles + feature_cols
    if len(set(selected)) != len(selected) or not set(selected).issubset(df.columns) or not feature_cols:
        raise ValueError("Column roles must be distinct, present, and include numeric features")
    numeric_cols = _parse_feature_values(df, csv_path, roles, feature_cols, categorical_columns)

    # Compute per-sample lengths
    if sample_id_col and sample_id_col in df.columns:
        entity_rows = df.groupby(sample_id_col, sort=True).indices
    else:
        entity_rows = {None: df.index.to_numpy()}
    lengths = [len(rows) for rows in entity_rows.values()]

    n_samples = len(lengths)
    min_len = int(min(lengths))
    max_len = int(max(lengths))
    avg_len = float(np.mean(lengths))
    has_variable_length = min_len != max_len

    feature_data = df[feature_cols]
    missing_counts = feature_data.isna().sum()
    missing_rate = int(missing_counts.sum()) / feature_data.size
    per_feature_missing_rate = (missing_counts / len(df)).to_dict()

    # Classification info
    n_classes = None
    if label_col:
        n_classes = int(df[label_col].dropna().nunique())

    # Auto-detect task type
    if task_type is None:
        task_type = "classification" if label_col else "imputation"

    ts_info = TimestampInfo(has_timestamp=timestamp_col is not None)
    if timestamp_col and timestamp_col in df.columns:
        ts_info, entity_rows = _profile_timestamps(df, entity_rows, timestamp_col)

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
            per_feature_missing_rate=per_feature_missing_rate,
            per_feature_value_domain={
                column: _feature_value_domain(df[column].to_numpy(), entity_rows) for column in numeric_cols
            },
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


def _profile_timestamps(
    df: pd.DataFrame,
    entity_rows: dict[str | None, np.ndarray],
    timestamp_col: str,
) -> tuple[TimestampInfo, dict[str | None, np.ndarray]]:
    """Summarize timestamp spacing and return chronological rows when timestamps are complete."""
    info = TimestampInfo(has_timestamp=True)

    try:
        ts = pd.to_datetime(df[timestamp_col], format="mixed", utc=True)
    except Exception:
        info.warning = f"Could not parse '{timestamp_col}' as datetime."
        return info, entity_rows

    ordered_rows = {
        entity: ts.iloc[rows].sort_values(kind="stable").index.to_numpy() for entity, rows in entity_rows.items()
    }
    diffs = pd.concat([ts.iloc[rows].diff().dropna() for rows in ordered_rows.values()]).dt.total_seconds()
    if ts.notna().all():
        entity_rows = ordered_rows
    if len(diffs) == 0:
        return info, entity_rows

    median_step = float(diffs.median())
    variance = float(diffs.var(ddof=0))

    info.median_step_seconds = median_step
    info.step_variance = variance
    info.is_regular = variance < 1e-6 * (median_step**2) if median_step > 0 else True

    if not info.is_regular:
        info.warning = (
            f"Irregular sampling detected: median step={median_step:.1f}s, "
            f"variance={variance:.1f}. The data may need resampling before "
            f"model training. Proceed with caution or ask the user to preprocess."
        )

    return info, entity_rows
