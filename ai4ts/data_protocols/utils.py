"""
Low-level utility functions for time series data processing.
"""

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

from .constants import (
    COL_NAME_SAMPLE_ID,
    COL_NAME_TIMESTAMP,
    CLAF_TARGET_SYMBOL,
    AT_LEAST_SAMPLE_NUM,
    DEFAULT_TIMESTAMP_START,
    DEFAULT_TIMESTAMP_FREQ,
    RESERVED_COLUMNS,
)
from .enums import FeatureType


def separate_features(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Separate DataFrame columns by type and role.

    Identifies numeric, string, and integer feature columns, plus classification
    target columns (identified by CLAF_TARGET symbol in column name).
    Reserved columns (SAMPLE_ID, TIMESTAMP, etc.) are excluded from features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.

    Returns
    -------
    tuple of (numeric_features, string_features, integer_features, classification_targets)
        Each element is a list of column name strings.
    """
    all_columns = df.columns.tolist()

    classification_targets = [c for c in all_columns if CLAF_TARGET_SYMBOL in c]
    excluded = set(classification_targets) | RESERVED_COLUMNS
    feature_cols = [c for c in all_columns if c not in excluded]

    numeric_features = [c for c in feature_cols if pd_types.is_numeric_dtype(df[c])]
    string_features = [c for c in feature_cols if pd_types.is_string_dtype(df[c])]
    integer_features = [c for c in feature_cols if pd_types.is_integer_dtype(df[c])]

    # Only use the first classification target if multiple are found
    if len(classification_targets) > 1:
        classification_targets = [classification_targets[0]]

    return numeric_features, string_features, integer_features, classification_targets


def get_feature_type(series: pd.Series) -> FeatureType:
    """Determine the FeatureType of a pandas Series."""
    if pd_types.is_integer_dtype(series):
        return FeatureType.INTEGER
    elif pd_types.is_numeric_dtype(series):
        return FeatureType.NUMERIC
    else:
        return FeatureType.STRING


def compute_sliding_stride(
    data_length: int,
    ts_context_len: int,
    min_samples: int | None = AT_LEAST_SAMPLE_NUM,
) -> int:
    """
    Compute the sliding window stride for single-sample datasets.

    Ensures at least `min_samples` windows can be generated from the data.

    Parameters
    ----------
    data_length : int
        Total number of timesteps in the dataset.
    ts_context_len : int
        Length of each context window.
    min_samples : int or None
        Minimum number of windows desired. If None, stride=1.

    Returns
    -------
    int
        Sliding window stride (>= 1).
    """
    if min_samples is None or data_length <= min_samples:
        return 1
    stride = min(data_length // min_samples, ts_context_len)
    return max(stride, 1)


def compute_n_samples(data_length: int, ts_context_len: int, stride: int) -> int:
    """
    Compute number of samples from sliding window parameters.

    Parameters
    ----------
    data_length : int
        Total number of timesteps.
    ts_context_len : int
        Context window length.
    stride : int
        Sliding window stride.

    Returns
    -------
    int
        Number of samples that can be extracted.
    """
    if data_length < ts_context_len:
        return 1  # at least one (will be padded)
    return (data_length - ts_context_len) // stride + 1


def forward_fill(data: np.ndarray) -> np.ndarray:
    """
    Forward-fill NaN values along the time axis (axis=0).

    For each column, propagate the last observed value forward.
    Leading NaNs are filled with 0 (no prior value to propagate).

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (timesteps, features) potentially containing NaN.

    Returns
    -------
    np.ndarray
        Array with NaN values forward-filled.
    """
    filled = data.copy()
    for col in range(filled.shape[1]):
        col_data = filled[:, col]
        mask = np.isnan(col_data)
        if not mask.any():
            continue
        # Forward fill
        indices = np.where(~mask, np.arange(len(col_data)), 0)
        np.maximum.accumulate(indices, out=indices)
        filled[:, col] = col_data[indices]
        # Fill leading NaN with 0
        first_valid = np.argmax(~mask)
        if mask[0]:
            filled[:first_valid, col] = 0.0
    return filled


def create_missing_mask(data: np.ndarray) -> np.ndarray:
    """
    Create a binary mask indicating observed values.

    Parameters
    ----------
    data : np.ndarray
        Input array potentially containing NaN.

    Returns
    -------
    np.ndarray
        Binary mask: 1 where data is observed, 0 where NaN.
    """
    return (~np.isnan(data)).astype(np.float32)


def fill_and_get_mask(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward-fill NaN values and create the corresponding missing mask.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (timesteps, features).

    Returns
    -------
    tuple of (filled_data, missing_mask)
        filled_data: NaN values forward-filled.
        missing_mask: 1 where originally observed, 0 where originally NaN.
    """
    mask = create_missing_mask(data)
    filled = forward_fill(data)
    return filled, mask


def insert_anchor_points(data: np.ndarray, n_anchors: int = 2) -> np.ndarray:
    """
    Prepend anchor points (rows of ones) to the beginning of a sequence.

    Anchor points serve as reference values for the model's attention mechanism.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (timesteps, features).
    n_anchors : int
        Number of anchor rows to prepend.

    Returns
    -------
    np.ndarray
        Array of shape (n_anchors + timesteps, features).
    """
    if n_anchors <= 0:
        return data
    n_features = data.shape[1]
    anchors = np.ones((n_anchors, n_features), dtype=data.dtype)
    return np.concatenate([anchors, data], axis=0)


def pad_sequence(
    data: np.ndarray,
    max_len: int,
    pad_value: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad a sequence to a fixed length along the time axis.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (timesteps, features).
    max_len : int
        Desired total length.
    pad_value : float
        Value used for padding.

    Returns
    -------
    tuple of (padded_data, valid_mask)
        padded_data: Shape (max_len, features).
        valid_mask: 1D bool array of length max_len, True where data is real.
    """
    actual_len = data.shape[0]
    n_features = data.shape[1]
    valid_mask = np.zeros(max_len, dtype=bool)

    if actual_len >= max_len:
        return data[:max_len].copy(), np.ones(max_len, dtype=bool)

    padded = np.full((max_len, n_features), pad_value, dtype=data.dtype)
    padded[:actual_len] = data
    valid_mask[:actual_len] = True
    return padded, valid_mask


def create_padding_mask(actual_len: int, max_len: int) -> np.ndarray:
    """
    Create a 2D attention-style padding mask.

    Parameters
    ----------
    actual_len : int
        Number of real timesteps (including anchors).
    max_len : int
        Total padded length.

    Returns
    -------
    np.ndarray
        2D bool mask of shape (max_len, max_len).
        True where both row and column positions are valid (real data).
    """
    valid = np.zeros(max_len, dtype=bool)
    valid[:actual_len] = True
    return np.outer(valid, valid)


def inject_artificial_missing(
    data: np.ndarray,
    rate: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Inject missing values (NaN) at random positions (MCAR pattern).

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (timesteps, features).
    rate : float
        Probability of each value becoming NaN.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Copy of data with artificial NaN values injected.
    """
    if rng is None:
        rng = np.random.default_rng()
    result = data.copy()
    mask = rng.random(data.shape) < rate
    result[mask] = np.nan
    return result


def ensure_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add SAMPLE_ID column with value 0 if not present."""
    if COL_NAME_SAMPLE_ID not in df.columns:
        df = df.copy()
        df[COL_NAME_SAMPLE_ID] = 0
    return df


def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Add TIMESTAMP column with hourly sequence if not present."""
    if COL_NAME_TIMESTAMP not in df.columns:
        df = df.copy()
        df[COL_NAME_TIMESTAMP] = pd.date_range(
            start=DEFAULT_TIMESTAMP_START,
            periods=len(df),
            freq=DEFAULT_TIMESTAMP_FREQ,
        )
    return df
