"""Raw-scale entity-time forecasting frames with explicit covariate evidence."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .ingestion import MAX_FEATURE_CELLS, inspect_csv
from .preprocessing import state_sha256
from .registry import file_sha256
from .semantics import resolve_covariates
from .temporal_artifacts import write_json


def _names(columns):
    return all(isinstance(column, str) and column.strip() for column in columns) and len(set(columns)) == len(columns)


def _frame_columns(frame, name):
    if not isinstance(frame, pd.DataFrame) or not _names(frame.columns):
        raise ValueError(f"{name} must have unique nonempty column names")
    return set(frame.columns)


def _normalize(frame, keys, numeric, name):
    id_col, time_col = keys
    ids = frame[id_col]
    if not ids.astype(object).map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError(f"{name} identifiers must be nonempty strings")
    raw = frame[time_col]
    invalid = f"Invalid or missing {name.lower()} timestamp. Supply an absolute ISO 8601 date or datetime"
    if not pd.api.types.is_datetime64_any_dtype(raw):
        raw = raw.astype("string").str.strip()
        if pd.to_numeric(raw, errors="coerce").notna().any():
            raise ValueError(f"{name} numeric timestamps require upstream epoch conversion")
        if not raw.str.match(r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?:$|[Tt ])", na=False).all():
            raise ValueError(invalid)
    times = pd.to_datetime(raw, format="ISO8601", errors="coerce", utc=True)
    if times.isna().any():
        raise ValueError(invalid)
    frame = frame.reset_index(drop=True)
    frame[id_col], frame[time_col] = ids.astype("string").array, times.array
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate {name.lower()} timestamps within an entity")
    for column in frame.columns.difference(keys):
        values = frame[column].replace("", np.nan)
        if column in numeric:
            values = values.replace({"NaN": np.nan, "nan": np.nan})
            try:
                numbers = pd.to_numeric(values.astype("string"), errors="raise")
            except (ValueError, TypeError) as error:
                raise ValueError(f"{name} column {column!r} must be real numeric or missing") from error
            if np.isinf(numbers.to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError(f"{name} column {column!r} contains infinite values")
        frame[column] = values
    return frame.sort_values(keys, ignore_index=True)


def _cadence(context, id_col, time_col, frequency):
    intervals = context.groupby(id_col, sort=False)[time_col].diff().dropna()
    if frequency is None:
        if intervals.empty:
            raise ValueError("Supply a frequency when history has no within-entity intervals")
        step = intervals.iloc[0]
    else:
        if not isinstance(frequency, str):
            raise ValueError("Frequency must be a positive fixed duration string")
        try:
            step = pd.Timedelta(frequency)
        except (ValueError, OverflowError) as error:
            raise ValueError("Invalid frequency. Supply a positive fixed duration") from error
        if pd.isna(step) or step <= pd.Timedelta(0):
            raise ValueError("Frequency must be a positive fixed duration")
    if not intervals.eq(step).all():
        raise ValueError("History must have one regular cadence across all entities")
    return step


def _prepare_frames(
    history, future, sample_id_col, timestamp_col, feature_columns, target_features, n_pred_steps, covariates, frequency
):
    keys = [sample_id_col, timestamp_col]
    if not _names(keys):
        raise ValueError("Specify distinct entity and timestamp columns")
    if not isinstance(feature_columns, list) or not _names(feature_columns) or set(keys).intersection(feature_columns):
        raise ValueError("Selected features must be unique and exclude the identifier and timestamp")
    if (
        not isinstance(target_features, list)
        or not target_features
        or not _names(target_features)
        or not set(target_features).issubset(feature_columns)
    ):
        raise ValueError("Select a nonempty ordered target subset of the features")
    if type(n_pred_steps) is not int or n_pred_steps < 1:
        raise ValueError("n_pred_steps must be a positive integer")
    history_columns = _frame_columns(history, "History")
    if history.empty:
        raise ValueError("History must be nonempty")
    context_columns = keys + feature_columns
    missing = [column for column in context_columns if column not in history_columns]
    if missing:
        raise ValueError(f"Missing history columns: {missing}")

    roles = resolve_covariates(covariates, feature_columns, target_features)
    known = [column for column, item in roles.items() if item["role"] == "known"]
    numeric = set(target_features) | {column for column, item in roles.items() if item["kind"] == "numeric"}
    context = _normalize(history[context_columns], keys, numeric, "History")
    step = _cadence(context, sample_id_col, timestamp_col, frequency)
    origins = context.groupby(sample_id_col, sort=False)[timestamp_col].last()
    future_columns = keys + known
    if future is None and known:
        raise ValueError("A future table is required for declared known covariates")
    supplied_future = future is not None
    if supplied_future:
        actual = _frame_columns(future, "Future")
        if actual.intersection(target_features):
            raise ValueError("Future cannot contain target columns")
        if actual - history_columns:
            raise ValueError(f"Future contains columns absent from history: {sorted(actual - history_columns)}")
        missing = [column for column in future_columns if column not in actual]
        if missing:
            raise ValueError(f"Missing future columns: {missing}")
        if actual.difference(future_columns):
            raise ValueError("Future may contain only the identifier, timestamp and declared known covariates")
        future = _normalize(future[future_columns], keys, numeric, "Future")
        sizes = future.groupby(sample_id_col, sort=False).size()
        if set(sizes.index) != set(origins.index):
            raise ValueError("Future must cover the same entities as history")
        if not sizes.eq(n_pred_steps).all():
            raise ValueError(f"Future must have exactly {n_pred_steps} rows per entity")
    else:
        if len(origins) * n_pred_steps * len(keys) > MAX_FEATURE_CELLS:
            raise ValueError("Generated future grid exceeds the protocol cell limit")
        future = origins.index.repeat(n_pred_steps).to_frame(index=False)

    try:
        offsets = pd.timedelta_range(start=step, periods=n_pred_steps, freq=step)
        positions = future.groupby(sample_id_col, sort=False).cumcount()
        expected = future[sample_id_col].map(origins) + offsets.take(positions.to_numpy())
    except (ValueError, OverflowError) as error:
        raise ValueError("The forecast horizon exceeds the supported timestamp range") from error
    if not supplied_future:
        future[timestamp_col] = expected
    elif not future[timestamp_col].eq(expected).all():
        raise ValueError("Future timestamps must continue each entity's history grid")
    return context, future, roles, step


def validate_future_covariates(
    history: pd.DataFrame,
    future: pd.DataFrame | None,
    *,
    sample_id_col: str,
    timestamp_col: str,
    feature_columns: list[str],
    target_features: list[str],
    n_pred_steps: int,
    covariates: dict | None = None,
    frequency: str | None = None,
) -> pd.DataFrame:
    """Return a validated future grid ordered by entity and UTC timestamp."""
    _, future, _, _ = _prepare_frames(
        history, future, sample_id_col, timestamp_col, feature_columns, target_features, n_pred_steps, covariates,
        frequency,
    )
    return future


def _read_csv(path):
    facts = inspect_csv(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", engine="python")
    if file_sha256(path) != facts["sha256"]:
        raise ValueError("Source changed while reading forecasting frames")
    return frame, facts["sha256"]


def export_tsfm_frames(
    history_csv: str | Path,
    output: str | Path,
    *,
    sample_id_col: str,
    timestamp_col: str,
    feature_columns: list[str],
    target_features: list[str],
    n_pred_steps: int,
    covariates: dict | None = None,
    future_csv: str | Path | None = None,
    frequency: str | None = None,
) -> dict:
    """Write validated raw-scale context and future tables with a source-bound manifest."""
    history, source_sha = _read_csv(history_csv)
    future, future_sha = _read_csv(future_csv) if future_csv is not None else (None, None)
    context, future, roles, step = _prepare_frames(
        history, future, sample_id_col, timestamp_col, feature_columns, target_features, n_pred_steps, covariates,
        frequency,
    )
    consequences = [
        {"dropped_rows": 0},
        {"unselected_context_columns": [column for column in history if column not in context]},
        {"ordering": "entity_and_utc_timestamp"},
    ]
    if future_csv is None:
        consequences.append({"generated_future_rows": len(future)})
    manifest = {
        "schema": "ai4ts-tsfm-frames-v1",
        "status": "success",
        "sample_id_col": sample_id_col,
        "timestamp_col": timestamp_col,
        "feature_columns": feature_columns,
        "target_features": target_features,
        "covariates": roles,
        "covariates_sha256": state_sha256(roles),
        "n_pred_steps": n_pred_steps,
        "frequency": str(step),
        "time_basis": "regular_timestamp",
        "timezone": "UTC",
        "source_sha256": source_sha,
        "future_source_sha256": future_sha,
        "consequences": consequences,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    try:
        context.to_csv(root / "context.csv", index=False, mode="x")
        future.to_csv(root / "future.csv", index=False, mode="x")
        manifest["files"] = {name: file_sha256(root / name) for name in ("context.csv", "future.csv")}
        write_json(root / "manifest.json", manifest)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return manifest
