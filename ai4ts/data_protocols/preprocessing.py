"""Finite, portable StandardScaler state for the opt-in standard-v1 bridge."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def state_sha256(state: dict) -> str:
    """Hash a finite JSON state independently of file formatting."""
    return hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()


def training_sha256(values, feature_columns) -> str:
    """Identify the ordered, unmasked training observations and feature order."""
    values = np.asarray(values, dtype="<f8").copy()
    values[np.isnan(values)] = np.nan
    digest = hashlib.sha256(json.dumps(list(feature_columns)).encode())
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def validate_scaler(state: dict, feature_columns=None, training_sha=None) -> dict:
    """Reject malformed, nonfinite or incorrectly bound standard-v1 state.

    Parameters
    ----------
    state : dict
        Numeric StandardScaler state, never a serialized Python object.
    feature_columns : list or None
        Expected feature order, if supplied.
    training_sha : str or None
        Expected fingerprint of the unmasked training observations.

    Returns
    -------
    dict
        The validated state.
    """
    if not isinstance(state, dict) or state.get("profile") != "standard-v1" or state.get("version") != 1:
        raise ValueError("Unsupported scaler profile/version")
    columns = state.get("feature_columns")
    if not isinstance(columns, list) or not columns or any(not isinstance(c, (str, int)) for c in columns):
        raise ValueError("Invalid scaler feature order")
    if len(set(columns)) != len(columns) or (feature_columns is not None and columns != list(feature_columns)):
        raise ValueError("Scaler feature order mismatch")
    for key in ("mean", "scale", "variance", "n_samples_seen"):
        try:
            values = np.asarray(state[key], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid scaler {key}") from error
        if values.shape != (len(columns),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid scaler {key} length or finite values")
        if key in ("scale", "n_samples_seen") and (values <= 0).any():
            raise ValueError(f"Scaler {key} must be positive")
        if key == "variance" and (values < 0).any():
            raise ValueError("Scaler variance must be nonnegative")
    fingerprint = state.get("training_sha256", "")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(c not in "0123456789abcdef" for c in fingerprint)
    ):
        raise ValueError("Invalid scaler training fingerprint")
    if training_sha is not None and fingerprint != training_sha:
        raise ValueError("Scaler training source mismatch")
    state_sha256(state)
    return state


def scaler_state(scaler, feature_columns, training_sha: str) -> dict:
    """Extract finite numerical state from an already fitted StandardScaler."""
    columns = list(feature_columns)
    state = {
        "profile": "standard-v1",
        "version": 1,
        "feature_columns": columns,
        "training_sha256": training_sha,
        "mean": np.asarray(scaler.mean_).tolist(),
        "scale": np.asarray(scaler.scale_).tolist(),
        "variance": np.asarray(scaler.var_).tolist(),
        "n_samples_seen": np.broadcast_to(scaler.n_samples_seen_, (len(columns),)).tolist(),
    }
    return validate_scaler(state)


def fit_scaler(values, feature_columns) -> dict:
    """Fit only finite, observed training values; reject empty features."""
    from sklearn.preprocessing import StandardScaler

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(feature_columns) or np.isinf(values).any():
        raise ValueError("Invalid training values for scaler")
    if not np.isfinite(values).any(axis=0).all():
        raise ValueError("Training features must contain observed values")
    return scaler_state(StandardScaler().fit(values), feature_columns, training_sha256(values, feature_columns))


def transform_values(values, state: dict, inverse=False):
    """Transform arrays in recorded feature order, preserving unknown values."""
    validate_scaler(state)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != len(state["feature_columns"]):
        raise ValueError("Scaler/input feature length mismatch")
    mean, scale = np.asarray(state["mean"]), np.asarray(state["scale"])
    return values * scale + mean if inverse else (values - mean) / scale


def transform_targets(values, state: dict, target_features, inverse=False):
    """Transform a confirmed ordered target subset with training parameters."""
    validate_scaler(state)
    if (
        not target_features
        or len(set(target_features)) != len(target_features)
        or not set(target_features).issubset(state["feature_columns"])
    ):
        raise ValueError("Invalid scaler target feature order")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != len(target_features):
        raise ValueError("Target/scaler dimensions mismatch")
    indices = [state["feature_columns"].index(c) for c in target_features]
    mean, scale = np.asarray(state["mean"])[indices], np.asarray(state["scale"])[indices]
    return values * scale + mean if inverse else (values - mean) / scale


def save_scaler(state: dict, path: str | Path) -> None:
    """Save validated numerical state as finite JSON."""
    validate_scaler(state)
    Path(path).write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_scaler(path: str | Path, feature_columns=None, training_sha=None) -> dict:
    """Load and validate a saved numerical state and optional bindings."""
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_scaler(state, feature_columns, training_sha)
