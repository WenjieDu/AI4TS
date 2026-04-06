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

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import COL_NAME_SAMPLE_ID, CLAF_TARGET_SYMBOL, RESERVED_COLUMNS
from .profile import DataProfile, analyze_dataset
from .registry import WindowEntry, WindowRegistry

# Samples with uniform length at or below this threshold skip sliding windows
_DIRECT_THRESHOLD = 200


def prepare_for_pypots(
    csv_path: str | Path,
    task: str = "imputation",
    window_size: int | None = None,
    missing_rate: float = 0.1,
    set_type: str = "train",
    seed: int = 2024,
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
        Rate of artificial missing injection for val/test sets.
    set_type : str
        One of "train", "val", "test".
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        - "X": np.ndarray shape [n_windows, window_size, n_features]
        - "X_ori": np.ndarray (same shape, without artificial missing)
        - "y": np.ndarray [n_windows] (if classification labels exist)
        - "registry": WindowRegistry
        - "profile": DataProfile
    """
    profile = analyze_dataset(csv_path, task_type=task)
    df = pd.read_csv(csv_path)

    schema = profile.schema_mapping
    sample_id_col = schema.sample_id_col
    label_col = schema.label_col
    feature_cols = schema.feature_cols

    if not feature_cols:
        raise ValueError(f"No numeric feature columns found in {csv_path}")

    # Extract per-sample data arrays
    samples, sample_ids = _extract_samples(df, sample_id_col, feature_cols)

    # Extract per-sample labels if present
    labels = None
    if label_col:
        labels = _extract_sample_labels(df, sample_id_col, label_col)

    # Determine windowing strategy and window_size
    lengths = [s.shape[0] for s in samples]
    is_classification = task == "classification"
    strategy, effective_window_size = _determine_strategy(
        lengths, is_classification, window_size
    )

    # Apply transformation
    X_ori, registry = _transform(
        samples, sample_ids, effective_window_size, strategy, is_classification
    )
    registry.n_features = len(feature_cols)
    registry.task_type = task

    n_windows = X_ori.shape[0]

    # Build X (with artificial missing for val/test)
    if set_type == "train":
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
    if y is not None:
        result["y"] = y

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
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:n_train + n_val]
    test_ids = unique_ids[n_train + n_val:]

    return list(train_ids), list(val_ids), list(test_ids)


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
    if predictions.shape[0] != registry.n_windows:
        raise ValueError(
            f"Predictions have {predictions.shape[0]} windows but registry has "
            f"{registry.n_windows} entries."
        )

    result = {}
    for sample_id in registry.original_sample_ids:
        entries = registry.get_entries_for_sample(sample_id)
        segments = []
        for entry in entries:
            window = predictions[entry.window_id]
            # Strip padding: keep only actual_length timesteps
            actual = window[:entry.actual_length]
            segments.append(actual)

        if segments:
            result[sample_id] = np.concatenate(segments, axis=0)

    return result


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
        samples = [groups.get_group(sid)[feature_cols].values.astype(np.float64)
                    for sid in sample_ids]
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
        return "pad_only", max_len

    if requested_window_size is not None:
        # User explicitly requested a window size
        if max_len <= requested_window_size and not has_variable:
            return "direct", max_len
        else:
            return "sliding_window", requested_window_size

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
