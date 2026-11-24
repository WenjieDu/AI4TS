"""
Data transformation pipeline: CSV → PyPOTS-ready H5 + Window Registry.

This module implements the core transformation logic that bridges ai4ts CSV
data (variable-length, multi-sample) to PyPOTS model input (fixed-length 3D
tensors). It also provides reconstruction utilities to reverse the process
after model inference.

Key design decisions:
- Classification tasks: never use sliding windows (labels are per-sample).
  Pad shorter samples to max_len with NaN.
- Non-classification tasks: use non-overlapping sliding windows when samples
  are long (>200 steps) or variable-length. Otherwise, direct pad-to-max.
- All splits happen at SAMPLE_ID level to prevent data leakage.
- A WindowRegistry tracks every output window back to its source.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import COL_NAME_SAMPLE_ID
from .profile import SchemaMapping, analyze_dataset
from .registry import WindowEntry, WindowRegistry, file_sha256

# Samples with uniform length at or below this threshold skip sliding windows
_DIRECT_THRESHOLD = 200


def prepare_for_pypots(
    csv_path: str | Path,
    task: str = "imputation",
    window_size: int | None = None,
    missing_rate: float = 0.1,
    set_type: str = "train",
    seed: int = 2024,
    schema_mapping: SchemaMapping | dict | None = None,
    allow_irregular: bool = False,
    row_indices: list[int] | None = None,
    split_info: dict | None = None,
    preprocessing: str | None = None,
    scaler_state: dict | None = None,
    task_state: dict | None = None,
    n_pred_steps: int | None = None,
    target_features: list[str] | None = None,
    feature_units: dict | None = None,
    unit_state: dict | None = None,
    ingestion_plan: dict | None = None,
) -> dict[str, Any]:
    """Convert a CSV file to PyPOTS-ready arrays with a window registry.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file following the ai4ts data protocol.
    task : str
        Task type: imputation, classification, forecasting, clustering, anomaly_detection.
    window_size : int or None
        Fixed window size for sliding window. If None, auto-determined:
        - Classification: max sample length (no windowing)
        - Others: max sample length if ≤ _DIRECT_THRESHOLD, else 100
    missing_rate : float
        Input masking rate. Standardized train/val/test are masked; legacy raw
        training and full-source inference retain natural observations.
    set_type : str
        One of "train", "val", "test", "full"; full never injects missingness.
    seed : int
        Random seed for reproducibility.
    schema_mapping : SchemaMapping or dict or None
        Confirmed column roles, reused across datasets.
    allow_irregular : bool
        Explicit consent to analyze irregular observations by record index.

    row_indices : list of int or None
        Source record positions assigned to this split, before windowing.
    split_info : dict or None
        Recorded split policy and role.
    preprocessing : str or None
        Opt-in ``standard-v1`` training-only scaling; None retains raw scale.
    scaler_state : dict or None
        Saved training-only state, required for standardized val/test/full.
    task_state : dict or None
        Saved classification/forecasting training roles, windows, and labels.
        Required for their val/test/full preparation; never refitted.
    n_pred_steps : int or None
        Positive forecast horizon, fixed by training state.
    target_features : list of str or None
        Confirmed ordered forecast feature subset, fixed by training state.
    feature_units : dict or None
        Optional confirmed feature units or source-bound unit declaration.
        Independent of column roles and numerical preprocessing.
    ingestion_plan : dict or None
        Optional ready source-bound ingestion plan, checked against actual windows.
    unit_state : dict or None
        Optional saved training units to inherit/check for evaluation inputs.

    Returns
    -------
    dict with keys:
        - "X": np.ndarray shape [n_windows, window_size, n_features]
        - "X_ori": np.ndarray (same shape, without artificial missing)
        - "y": np.ndarray [n_windows] (if classification labels exist)
        - "registry": WindowRegistry
        - "profile": DataProfile
    """
    if ingestion_plan is not None:
        from .ingestion import validate_ingestion_plan, verify_prepared_plan

        validate_ingestion_plan(ingestion_plan, csv_path)
        params = ingestion_plan["parameters"]
        if task != ingestion_plan["request"]["task"] or set_type not in ingestion_plan["preview"]:
            raise ValueError("Plan/task/split mismatch")
        if schema_mapping is not None and schema_mapping != ingestion_plan["roles"]:
            raise ValueError("Plan/column role override conflict")
        if window_size is not None and window_size != params["window_size"]:
            raise ValueError("Plan/window override conflict")
        if missing_rate != params["missing_rate"] or seed != params["seed"]:
            raise ValueError("Plan/masking override conflict")
        if n_pred_steps is not None and n_pred_steps != params["horizon"]:
            raise ValueError("Plan/horizon override conflict")
        if target_features is not None and target_features != params["targets"]:
            raise ValueError("Plan/target override conflict")
        expected_rows = ingestion_plan["split"]["rows"].get(set_type)
        if row_indices is not None and row_indices != expected_rows:
            raise ValueError("Plan/source row override conflict")
        if split_info is not None and any(split_info.get(k) != v for k, v in ingestion_plan["split"].items()):
            raise ValueError("Plan/split override conflict")
        schema_mapping, window_size = ingestion_plan["roles"], params["window_size"]
        allow_irregular = params["allow_irregular"]
        row_indices, split_info = expected_rows, dict(ingestion_plan["split"], role=set_type)
        n_pred_steps, target_features = params["horizon"], params["targets"]
        if ingestion_plan.get("feature_semantics") and unit_state is None:
            unit_state = ingestion_plan["feature_semantics"]
    if set_type not in ("train", "val", "test", "full") or not np.isfinite(missing_rate) or not 0 <= missing_rate <= 1:
        raise ValueError("Invalid split role or artificial missing rate")
    if task in ("classification", "forecasting", "clustering", "anomaly_detection", "representation"):
        from .task_data import prepare_task

        result = prepare_task(
            csv_path,
            task,
            window_size,
            missing_rate,
            set_type,
            seed,
            schema_mapping,
            allow_irregular,
            row_indices,
            split_info,
            preprocessing,
            scaler_state,
            task_state,
            n_pred_steps,
            target_features,
            feature_units,
            unit_state,
        )
        if ingestion_plan is not None:
            verify_prepared_plan(ingestion_plan, set_type, result)
        return result
    if task_state is not None or n_pred_steps is not None or target_features is not None:
        raise ValueError("Task state and forecast options require classification/forecasting")
    if preprocessing not in (None, "standard-v1") or (preprocessing and task != "imputation"):
        raise ValueError("standard-v1 currently supports imputation only")
    if scaler_state is not None and preprocessing is None:
        raise ValueError("Scaler state requires explicit standard-v1 preprocessing")
    df, profile = _read_source(csv_path, task, schema_mapping, allow_irregular)
    if split_info and split_info.get("policy") == "entity_chronological":
        validate_source_split(df, vars(profile.schema_mapping), split_info, task, row_indices, set_type)
    source_df = df
    source_row_count = len(df)
    if row_indices is not None:
        if not row_indices or len(set(row_indices)) != len(row_indices) or not set(row_indices).issubset(df.index):
            raise ValueError("Split row indices must be nonempty, unique and in the source")
        df = df.loc[df.index.isin(row_indices)]
    schema = profile.schema_mapping
    sample_id_col, label_col, feature_cols = schema.sample_id_col, schema.label_col, schema.feature_cols
    from .semantics import prepare_feature_semantics

    semantics = prepare_feature_semantics(feature_units, feature_cols, file_sha256(csv_path), unit_state)
    if set_type == "train" and df[feature_cols].isna().all().any():
        raise ValueError("Training features must contain observed values")
    if preprocessing:
        from .preprocessing import fit_scaler, training_sha256, transform_values, validate_scaler

        if not 0 <= missing_rate < 1:
            raise ValueError("standard-v1 missing rate must be in [0, 1)")
        if scaler_state is None:
            if set_type != "train":
                raise ValueError("standard-v1 requires the saved training scaler for val/test/full")
            scaler_state = fit_scaler(df[feature_cols].to_numpy(), feature_cols)
        training_rows = (split_info or {}).get("rows", {}).get("train")
        training_values = source_df.loc[source_df.index.isin(training_rows), feature_cols] if training_rows else None
        if set_type == "train":
            training_values = df[feature_cols]
        training_sha = training_sha256(training_values, feature_cols) if training_values is not None else None
        validate_scaler(scaler_state, feature_cols, training_sha)
        df = df.copy()
        df[feature_cols] = transform_values(df[feature_cols].to_numpy(), scaler_state)

    # Extract per-sample data arrays
    samples, sample_ids = _extract_samples(df, sample_id_col, feature_cols)

    # Extract per-sample labels if present
    labels = None
    if label_col and task == "classification":
        labels = _extract_sample_labels(df, sample_id_col, label_col)
    elif label_col and pd.to_numeric(df[label_col], errors="coerce").notna().all():
        # Retain the legacy optional numeric y without encoding passthrough strings.
        labels = _extract_sample_labels(
            df.assign(**{label_col: pd.to_numeric(df[label_col])}), sample_id_col, label_col
        )

    # Determine windowing strategy and window_size
    lengths = [s.shape[0] for s in samples]
    is_classification = task == "classification"
    strategy, effective_window_size = _determine_strategy(lengths, is_classification, window_size)

    # Apply transformation
    X_ori, registry = _transform(samples, sample_ids, effective_window_size, strategy, is_classification)
    registry.n_features = len(feature_cols)
    registry.task_type = task
    registry.format_version = 2
    registry.source_sha256 = file_sha256(csv_path)
    registry.source_row_count = source_row_count
    registry.split_info = split_info or {}
    if preprocessing:
        from .preprocessing import state_sha256

        registry.preprocessing = preprocessing
        registry.scaler_sha256 = state_sha256(scaler_state)
    registry.columns = list(df.columns)
    registry.column_roles = {
        "sample_id_col": sample_id_col,
        "timestamp_col": schema.timestamp_col,
        "label_col": label_col,
        "feature_cols": feature_cols,
    }
    registry.feature_columns = list(feature_cols)
    if semantics is not None:
        from .preprocessing import state_sha256

        registry.feature_semantics, registry.feature_semantics_sha256 = semantics, state_sha256(semantics)
    registry.ordering = "timestamp" if schema.timestamp_col else "row_order"
    source_rows = (
        {sid: list(group.index) for sid, group in df.groupby(sample_id_col, sort=True)}
        if sample_id_col
        else {0: list(df.index)}
    )
    for entry in registry.entries:
        entry.row_indices = source_rows[entry.original_sample_id][entry.start_step : entry.end_step]

    # Build X (with artificial missing for val/test)
    if preprocessing and set_type != "full":
        from benchpots.utils.missingness import create_missingness

        split_seed = int(np.random.SeedSequence([seed, ("train", "val", "test").index(set_type)]).generate_state(1)[0])
        # BenchPOTS point masking uses the global NumPy RNG; restore the caller's state.
        rng_state = np.random.get_state()
        try:
            np.random.seed(split_seed)
            X = create_missingness(X_ori, missing_rate, "point") if missing_rate else X_ori.copy()
        finally:
            np.random.set_state(rng_state)
        registry.split_info = dict(
            registry.split_info, masking={"pattern": "point", "seed": split_seed, "rate": missing_rate}
        )
    elif set_type in ("train", "full"):
        X = X_ori.copy()
    else:
        X = _inject_artificial_missing(X_ori, missing_rate, seed)

    # Build labels array aligned to windows
    y = None
    if labels is not None:
        y = _align_labels_to_windows(labels, sample_ids, registry)

    result = {
        "X": X,
        "X_ori": X_ori,
        "registry": registry,
        "profile": profile,
    }
    if preprocessing:
        result["scaler_state"] = scaler_state
    if semantics is not None:
        result["feature_semantics"] = semantics
    valid_mask = np.zeros_like(X_ori, dtype=bool)
    for entry in registry.entries:
        valid_mask[entry.window_id, : entry.actual_length] = True
    result["valid_mask"] = valid_mask
    result["indicating_mask"] = valid_mask & np.isfinite(X_ori) & np.isnan(X)
    if y is not None:
        result["y"] = y

    if ingestion_plan is not None:
        verify_prepared_plan(ingestion_plan, set_type, result)
    return result


def split_by_sample_id(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 2024,
    sample_id_col: str | None = None,
) -> tuple[list, list, list]:
    """Split sample IDs into train/val/test groups.

    Splitting is done strictly at the SAMPLE_ID level. No sample's windows
    will appear in multiple splits.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    train_ratio, val_ratio, test_ratio : float
        Split ratios (must sum to 1.0).
    seed : int
        Random seed.
    sample_id_col : str or None
        Column name for sample IDs. If None, uses COL_NAME_SAMPLE_ID.

    Returns
    -------
    tuple of (train_ids, val_ids, test_ids)
    """
    if sample_id_col is None:
        sample_id_col = COL_NAME_SAMPLE_ID

    if sample_id_col not in df.columns:
        raise ValueError(f"Column '{sample_id_col}' not found in DataFrame")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum:.6f}")

    unique_ids = sorted(df[sample_id_col].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_ids)

    n = len(unique_ids)
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(r < 0 for r in ratios) or n < sum(r > 0 for r in ratios):
        raise ValueError("Insufficient samples for nonempty requested splits")
    counts = [max(int(n * r), int(r > 0)) for r in ratios]
    while sum(counts) > n:
        largest = max(range(3), key=lambda i: counts[i])
        counts[largest] -= 1
    # Allocate the remaining records to training, preserving small held-out sets.
    counts[0 if train_ratio > 0 else max(range(3), key=lambda i: ratios[i])] += n - sum(counts)
    n_train, n_val = counts[:2]

    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train : n_train + n_val]
    test_ids = unique_ids[n_train + n_val :]

    return list(train_ids), list(val_ids), list(test_ids)


def _read_source(csv_path, task, schema_mapping, allow_irregular):
    """Read and validate source roles and ordering without windowing."""
    profile = analyze_dataset(csv_path, task_type=task, schema_mapping=schema_mapping)

    schema = profile.schema_mapping
    sample_id_col = schema.sample_id_col
    label_col = schema.label_col
    feature_cols = schema.feature_cols
    roles = [c for c in (sample_id_col, schema.timestamp_col, label_col) if c]
    df = pd.read_csv(csv_path)
    if roles:
        df[roles] = pd.read_csv(csv_path, usecols=roles, dtype=str, keep_default_na=False)[roles]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="raise")
    if np.isinf(df[feature_cols].to_numpy(dtype=float)).any():
        raise ValueError("Features contain infinite values")
    if sample_id_col and (df[sample_id_col].isna().any() or df[sample_id_col].eq("").any()):
        raise ValueError("Sample identifiers cannot be missing")
    if schema.timestamp_col:
        ts = pd.to_datetime(df[schema.timestamp_col], format="mixed", utc=True, errors="raise")
        if ts.isna().any():
            raise ValueError("Timestamps cannot be missing")
        keys = [df[sample_id_col], ts] if sample_id_col else [ts]
        if pd.MultiIndex.from_arrays(keys).duplicated().any():
            raise ValueError("Duplicate timestamps within a sample")
        if not profile.timestamp_info.is_regular and not allow_irregular:
            raise ValueError("Irregular sampling requires explicit record-index analysis")
        df = df.loc[ts.sort_values(kind="stable").index]

    if not feature_cols:
        raise ValueError(f"No numeric feature columns found in {csv_path}")

    return df, profile


def split_source_rows(
    csv_path: str | Path,
    schema_mapping: SchemaMapping | dict | None = None,
    seed: int = 2024,
    allow_irregular: bool = False,
    policy: str | None = None,
) -> dict:
    """Assign original records to nonempty train/val/test splits before windowing.

    Independent entities are held out as entities. One series is partitioned
    chronologically (or by declared row order), with at least two rows per split.
    Two entities cannot form an independent three-way comparison.

    Parameters
    ----------
    csv_path : str or Path
        Original CSV snapshot.
    schema_mapping : SchemaMapping or dict or None
        Confirmed column roles.
    seed : int
        Reproducible entity shuffle seed.
    allow_irregular : bool
        Consent to record-index analysis.
    policy : str or None
        Explicit ``entity_chronological`` partitions each entity in time.
        None preserves the legacy entity/single-series policy.

    Returns
    -------
    dict
        Policy, seed, and original row positions for each split.
    """
    df, profile = _read_source(csv_path, "imputation", schema_mapping, allow_irregular)
    id_col = profile.schema_mapping.sample_id_col
    groups = list(df.groupby(id_col, sort=True)) if id_col else [(0, df)]
    if policy not in (None, "entity", "chronological", "row_order", "entity_chronological"):
        raise ValueError("Unsupported source split policy")
    if policy == "entity_chronological":
        rows = {name: [] for name in ("train", "val", "test")}
        for sid, group in groups:
            n = len(group)
            nv, nt = max(2, int(n * 0.1)), max(2, int(n * 0.2))
            if n - nv - nt < 2:
                raise ValueError(f"Entity {sid!r} too short: each split requires at least two records")
            indices = group.index.tolist()
            for name, part in zip(rows, (indices[: n - nv - nt], indices[n - nv - nt : n - nt], indices[n - nt :])):
                rows[name].extend(part)
    elif len(groups) >= 3 and policy in (None, "entity"):
        ids = split_by_sample_id(df, seed=seed, sample_id_col=id_col)
        rows = {
            name: df.index[df[id_col].isin(selected)].tolist() for name, selected in zip(("train", "val", "test"), ids)
        }
        policy = "entity"
    elif len(groups) == 1 and policy != "entity":
        n = len(df)
        n_val, n_test = max(2, int(n * 0.1)), max(2, int(n * 0.2))
        n_train = n - n_val - n_test
        if n_train < 2:
            raise ValueError("Series too short: each split requires at least two records")
        indices = df.index.tolist()
        rows = {
            "train": indices[:n_train],
            "val": indices[n_train : n_train + n_val],
            "test": indices[n_train + n_val :],
        }
        policy = "chronological" if profile.schema_mapping.timestamp_col else "row_order"
    else:
        raise ValueError("Two entities cannot form three independent splits")
    return {"policy": policy, "seed": seed, "rows": rows}


def validate_source_split(source, roles, split_info, task, row_indices=None, set_type="train", strict=False):
    """Check source-row isolation; only explicit temporal policy shares entities."""
    rows = split_info.get("rows", {})
    assigned = [i for role in ("train", "val", "test") for i in rows.get(role, [])]
    if (
        any(type(i) is not int for i in assigned)
        or len(assigned) != len(set(assigned))
        or not set(assigned).issubset(source.index)
    ):
        raise ValueError("Split rows overlap or leave source bounds")
    temporal = split_info.get("policy") == "entity_chronological"
    if (strict or temporal) and (
        set(assigned) != set(source.index) or any(not rows.get(r) for r in ("train", "val", "test"))
    ):
        raise ValueError("Split must cover source rows exactly once with three nonempty roles")
    if set_type != "full" and set(row_indices if row_indices is not None else source.index) != set(
        rows.get(set_type, [])
    ):
        raise ValueError("Split role/rows mismatch")
    id_col = roles.get("sample_id_col")
    independent = task in ("classification", "clustering", "representation")
    if temporal and independent:
        raise ValueError("Independent samples cannot use entity_chronological")
    groups = [source.loc[source.index.isin(rows.get(r, []))] for r in ("train", "val", "test")]
    multi = bool(id_col and source[id_col].nunique() > 1)
    if not temporal and (independent or multi):
        if not id_col:
            raise ValueError("Independent samples require sample identifiers")
        ids = [set(g[id_col]) for g in groups]
        if any(ids[i] & ids[j] for i in range(3) for j in range(i)):
            raise ValueError("Sample entities overlap between splits")
        return
    entities = source.groupby(id_col, sort=True) if id_col else [(0, source)]
    for sid, group in entities:
        order = {idx: i for i, idx in enumerate(group.index)}
        boundaries = [[order[i] for i in g.index if i in order] for g in groups]
        if any(not b or b != list(range(min(b), max(b) + 1)) for b in boundaries) or any(
            max(boundaries[i]) >= min(boundaries[i + 1]) for i in range(2)
        ):
            raise ValueError(f"Forecast splits must be contiguous and chronological for entity {sid!r}")


def reconstruct_from_predictions(
    predictions: np.ndarray,
    registry: WindowRegistry,
) -> dict[Any, np.ndarray]:
    """Reconstruct original-shape arrays from model predictions using the registry.

    Strips padding from each window and reassembles windows belonging to the
    same original sample in start_step order.

    Parameters
    ----------
    predictions : np.ndarray
        Model output, shape [n_windows, window_size, n_features].
    registry : WindowRegistry
        The registry generated during data preparation.

    Returns
    -------
    dict mapping original_sample_id → np.ndarray of shape [original_length, n_features]
    """
    if predictions.shape != (registry.n_windows, registry.window_size, registry.n_features):
        raise ValueError("Prediction shape does not match registry windows, steps and features")

    result = {}
    for sample_id in registry.original_sample_ids:
        entries = registry.get_entries_for_sample(sample_id)
        segments = []
        for entry in entries:
            window = predictions[entry.window_id]
            # Strip padding: keep only actual_length timesteps
            actual = window[: entry.actual_length]
            segments.append(actual)

        if segments:
            result[sample_id] = np.concatenate(segments, axis=0)

    return result


def reconstruct_semantic_csv(
    predictions: np.ndarray,
    registry: WindowRegistry,
    source_csv: str | Path,
    binding: dict,
    scaler_state: dict | None = None,
) -> pd.DataFrame:
    """Fill only natural feature gaps in the original CSV layout.

    Parameters
    ----------
    predictions : np.ndarray
        Imputation tensor in recorded window/feature order.
    registry : WindowRegistry
        Version 2 registry covering the complete source.
    source_csv : str or Path
        Unmodified source snapshot, including passthrough columns.
    binding : dict
        Source/registry metadata saved with the predictions.
    scaler_state : dict or None
        Required saved state for a standard-v1 registry.

    Returns
    -------
    pd.DataFrame
        Original string contents, with finite estimates inserted in natural gaps.
    """
    registry.validate_binding(predictions, source_csv, binding)
    if registry.preprocessing:
        from .preprocessing import state_sha256, transform_values, validate_scaler

        validate_scaler(scaler_state, registry.feature_columns)
        if state_sha256(scaler_state) != registry.scaler_sha256:
            raise ValueError("Scaler/registry binding mismatch")
        predictions = transform_values(predictions, scaler_state, inverse=True)
    elif scaler_state is not None:
        raise ValueError("Raw-scale registry cannot use a standardized scaler")
    if registry.task_type != "imputation":
        raise ValueError("Semantic CSV export currently supports imputation only")
    original = pd.read_csv(source_csv, dtype=str, keep_default_na=False)
    numeric = pd.read_csv(source_csv)[registry.feature_columns].apply(pd.to_numeric, errors="raise")
    if list(original.columns) != registry.columns or len(original) != registry.source_row_count:
        raise ValueError("Source layout differs from registry")
    for entry in registry.entries:
        for step, row in enumerate(entry.row_indices):
            for feature, column in enumerate(registry.feature_columns):
                if pd.isna(numeric.at[row, column]):
                    value = predictions[entry.window_id, step, feature]
                    if not np.isfinite(value):
                        raise ValueError("Non-finite estimate at a natural missing position")
                    original.at[row, column] = str(float(value))
    return original


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────


def _extract_samples(
    df: pd.DataFrame,
    sample_id_col: str | None,
    feature_cols: list[str],
) -> tuple[list[np.ndarray], list]:
    """Group DataFrame rows by sample ID and extract feature arrays.

    Returns a list of 2D arrays (one per sample) and their sample IDs.
    """
    if sample_id_col and sample_id_col in df.columns:
        groups = df.groupby(sample_id_col, sort=True)
        sample_ids = sorted(df[sample_id_col].unique())
        samples = [groups.get_group(sid)[feature_cols].values.astype(np.float64) for sid in sample_ids]
    else:
        sample_ids = [0]
        samples = [df[feature_cols].values.astype(np.float64)]

    return samples, sample_ids


def _extract_sample_labels(
    df: pd.DataFrame,
    sample_id_col: str | None,
    label_col: str,
) -> dict:
    """Extract one label per sample ID.

    Returns dict mapping sample_id → label value.
    """
    from sklearn.preprocessing import LabelEncoder

    if sample_id_col and sample_id_col in df.columns:
        per_sample = df.groupby(sample_id_col, sort=True)[label_col].first()
        label_values = per_sample.values
        sample_ids = sorted(per_sample.index)
    else:
        label_values = np.array([df[label_col].iloc[0]])
        sample_ids = [0]

    # Encode string labels
    if label_values.dtype == object or label_values.dtype.kind in ("U", "S"):
        le = LabelEncoder()
        label_values = le.fit_transform(label_values)

    return dict(zip(sample_ids, label_values.astype(np.float64)))


def _determine_strategy(
    lengths: list[int],
    is_classification: bool,
    requested_window_size: int | None,
) -> tuple[str, int]:
    """Decide windowing strategy and effective window size.

    Returns
    -------
    tuple of (strategy_name, window_size)
    """
    min_len = min(lengths)
    max_len = max(lengths)
    has_variable = min_len != max_len

    if is_classification:
        # Classification: pad to max_len, never window
        if requested_window_size is not None and requested_window_size < max_len:
            raise ValueError("Classification window cannot truncate samples")
        return "pad_only", requested_window_size or max_len

    if requested_window_size is not None:
        if not isinstance(requested_window_size, int) or requested_window_size < 1:
            raise ValueError("Window size must be a positive integer")
        return ("direct" if max_len <= requested_window_size else "sliding_window"), requested_window_size

    # Auto-determine
    if not has_variable and max_len <= _DIRECT_THRESHOLD:
        return "direct", max_len

    # Need sliding window
    ws = min(max_len, _DIRECT_THRESHOLD) if max_len > _DIRECT_THRESHOLD else max_len
    return "sliding_window", ws


def _transform(
    samples: list[np.ndarray],
    sample_ids: list,
    window_size: int,
    strategy: str,
    is_classification: bool,
) -> tuple[np.ndarray, WindowRegistry]:
    """Apply the chosen windowing strategy to all samples.

    Returns the 3D output array and populated WindowRegistry.
    """
    registry = WindowRegistry(window_size=window_size, strategy=strategy)
    all_windows = []
    window_counter = 0

    for sample_idx, (sample, sid) in enumerate(zip(samples, sample_ids)):
        sample_len = sample.shape[0]
        n_features = sample.shape[1]

        if strategy == "pad_only" or strategy == "direct":
            # One window per sample, pad if needed
            if sample_len >= window_size:
                window = sample[:window_size]
                pad_len = 0
            else:
                window = np.full((window_size, n_features), np.nan, dtype=np.float64)
                window[:sample_len] = sample
                pad_len = window_size - sample_len

            entry = WindowEntry(
                window_id=window_counter,
                original_sample_id=sid,
                start_step=0,
                end_step=sample_len,
                is_padded=pad_len > 0,
                pad_length=pad_len,
            )
            registry.add(entry)
            all_windows.append(window)
            window_counter += 1

        elif strategy == "sliding_window":
            # Non-overlapping sliding window: stride = window_size
            n_full_windows = sample_len // window_size
            remainder = sample_len % window_size

            for w in range(n_full_windows):
                start = w * window_size
                end = start + window_size
                window = sample[start:end]
                entry = WindowEntry(
                    window_id=window_counter,
                    original_sample_id=sid,
                    start_step=start,
                    end_step=end,
                    is_padded=False,
                    pad_length=0,
                )
                registry.add(entry)
                all_windows.append(window)
                window_counter += 1

            # Handle tail: pad to window_size
            if remainder > 0:
                start = n_full_windows * window_size
                end = sample_len
                tail = sample[start:end]
                pad_len = window_size - remainder
                window = np.full((window_size, n_features), np.nan, dtype=np.float64)
                window[:remainder] = tail

                entry = WindowEntry(
                    window_id=window_counter,
                    original_sample_id=sid,
                    start_step=start,
                    end_step=end,
                    is_padded=True,
                    pad_length=pad_len,
                )
                registry.add(entry)
                all_windows.append(window)
                window_counter += 1

    X = np.stack(all_windows, axis=0)
    return X, registry


def _inject_artificial_missing(
    X_ori: np.ndarray,
    rate: float,
    seed: int,
) -> np.ndarray:
    """Inject MCAR artificial missing into observed positions only."""
    rng = np.random.RandomState(seed)
    X = X_ori.copy()
    if rate > 0:
        observed = ~np.isnan(X)
        random_mask = rng.rand(*X.shape) < rate
        artificial = observed & random_mask
        X[artificial] = np.nan
    return X


def _align_labels_to_windows(
    labels: dict,
    sample_ids: list,
    registry: WindowRegistry,
) -> np.ndarray:
    """Create a label array aligned to the registry windows.

    For classification (pad_only), each window maps 1:1 to its sample label.
    For sliding_window, each window inherits its source sample's label.
    """
    y = np.full(registry.n_windows, np.nan, dtype=np.float64)
    for entry in registry.entries:
        sid = entry.original_sample_id
        if sid in labels:
            y[entry.window_id] = labels[sid]
    return y
