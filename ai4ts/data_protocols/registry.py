"""
Window registry for tracking how original time series samples are sliced into
fixed-length windows during data preparation.

The registry enables exact reconstruction: after model inference on fixed-length
windows, the registry maps each window back to its original sample and position,
allowing padding to be stripped and windows to be reassembled in order.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class WindowEntry:
    """A single window's mapping back to its source sample.

    Attributes
    ----------
    window_id : int
        Index of this window in the output 3D array (axis 0).
    original_sample_id : Any
        The SAMPLE_ID value from the original CSV data.
    start_step : int
        The starting timestep index within the original sample.
    end_step : int
        The ending timestep index (exclusive) within the original sample.
    is_padded : bool
        Whether this window required NaN padding at the end.
    pad_length : int
        Number of NaN-padded timesteps at the tail of this window.
    """

    window_id: int
    original_sample_id: Any
    start_step: int
    end_step: int
    is_padded: bool = False
    pad_length: int = 0
    row_indices: list[int] = field(default_factory=list)
    target_row_indices: list[int] = field(default_factory=list)
    forecast_origin: int | None = None

    @property
    def actual_length(self) -> int:
        """Number of real (non-padded) timesteps in this window."""
        return self.end_step - self.start_step


@dataclass
class WindowRegistry:
    """Collection of window-to-sample mappings for a prepared dataset.

    The registry is the key to reversing the windowing transformation.
    It records exactly how each fixed-length window was carved from
    the original variable-length samples.

    Attributes
    ----------
    entries : list[WindowEntry]
        One entry per window in the output tensor.
    window_size : int
        The fixed window size (n_steps in the output tensor).
    n_features : int
        Number of features per timestep.
    task_type : str
        The task type used during preparation.
    strategy : str
        The windowing strategy: "pad_only" (classification) or
        "sliding_window" (non-classification with long/variable data) or
        "direct" (uniform length ≤ threshold).
    """

    entries: list[WindowEntry] = field(default_factory=list)
    window_size: int = 0
    n_features: int = 0
    task_type: str = ""
    strategy: str = "direct"
    format_version: int = 1
    source_sha256: str = ""
    source_row_count: int = 0
    columns: list[str] = field(default_factory=list)
    column_roles: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    ordering: str = "row_order"
    split_info: dict[str, Any] = field(default_factory=dict)
    preprocessing: str | None = None
    scaler_sha256: str = ""
    task_state_sha256: str = ""
    target_features: list[str] = field(default_factory=list)
    n_pred_steps: int = 0
    feature_semantics: dict | None = None
    feature_semantics_sha256: str = ""

    def add(self, entry: WindowEntry) -> None:
        """Add a window entry to the registry."""
        self.entries.append(entry)

    @property
    def n_windows(self) -> int:
        """Total number of windows in the registry."""
        return len(self.entries)

    @property
    def original_sample_ids(self) -> list:
        """Unique original sample IDs in order of first appearance."""
        seen = set()
        result = []
        for e in self.entries:
            if e.original_sample_id not in seen:
                seen.add(e.original_sample_id)
                result.append(e.original_sample_id)
        return result

    def get_entries_for_sample(self, sample_id: Any) -> list[WindowEntry]:
        """Get all window entries belonging to a given original sample, sorted by start_step."""
        entries = [e for e in self.entries if e.original_sample_id == sample_id]
        entries.sort(key=lambda e: e.start_step)
        return entries

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        result = {
            "window_size": self.window_size,
            "n_features": self.n_features,
            "task_type": self.task_type,
            "strategy": self.strategy,
            "format_version": self.format_version,
            "source_sha256": self.source_sha256,
            "source_row_count": self.source_row_count,
            "columns": self.columns,
            "column_roles": self.column_roles,
            "feature_columns": self.feature_columns,
            "ordering": self.ordering,
            "split_info": self.split_info,
            "entries": [
                {
                    k: v
                    for k, v in asdict(e).items()
                    if self.task_type == "forecasting" or k not in ("target_row_indices", "forecast_origin")
                }
                for e in self.entries
            ],
        }
        if self.preprocessing:
            result.update(preprocessing=self.preprocessing, scaler_sha256=self.scaler_sha256)
        if self.task_state_sha256:
            result["task_state_sha256"] = self.task_state_sha256
        if self.task_type == "forecasting":
            result.update(target_features=self.target_features, n_pred_steps=self.n_pred_steps)
        self.validate_semantics()
        if self.feature_semantics is not None:
            result.update(feature_semantics=self.feature_semantics,
                          feature_semantics_sha256=self.feature_semantics_sha256)
        return result

    def to_json(self, path: str | Path) -> None:
        """Save registry to a JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_serializer)

    @classmethod
    def from_dict(cls, d: dict) -> "WindowRegistry":
        """Reconstruct from a plain dict."""
        reg = cls(
            window_size=d["window_size"],
            n_features=d["n_features"],
            task_type=d.get("task_type", ""),
            strategy=d.get("strategy", "direct"),
            **{
                k: d[k]
                for k in (
                    "format_version",
                    "source_sha256",
                    "source_row_count",
                    "columns",
                    "column_roles",
                    "feature_columns",
                    "ordering",
                    "split_info",
                    "preprocessing",
                    "scaler_sha256",
                    "task_state_sha256",
                    "target_features",
                    "n_pred_steps",
                    "feature_semantics",
                    "feature_semantics_sha256",
                )
                if k in d
            },
        )
        if reg.format_version not in (1, 2):
            raise ValueError(f"Unsupported registry format: {reg.format_version}")
        if reg.preprocessing not in (None, "standard-v1") or bool(reg.preprocessing) != bool(reg.scaler_sha256):
            raise ValueError("Invalid registry preprocessing binding")
        if "feature_semantics" in d or "feature_semantics_sha256" in d:
            from .semantics import validate_semantic_binding

            if d.get("feature_semantics") is None or not d.get("feature_semantics_sha256"):
                raise ValueError("Incomplete registry feature semantics")
            validate_semantic_binding(d.get("feature_semantics"), d.get("feature_semantics_sha256"),
                                      reg.feature_columns, reg.source_sha256)
        for entry_dict in d.get("entries", []):
            reg.add(WindowEntry(**entry_dict))
        reg.validate_semantics()
        return reg

    @classmethod
    def from_json(cls, path: str | Path) -> "WindowRegistry":
        """Load registry from a JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    def fingerprint(self) -> str:
        """Return a stable SHA-256 of the registry's semantic contents."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=_json_serializer).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def prediction_binding(self) -> dict:
        """Return metadata required to bind a prediction to this registry."""
        result = {
            "registry_sha256": self.fingerprint(),
            "source_sha256": self.source_sha256,
            "feature_columns": self.feature_columns,
        }
        if self.preprocessing:
            result.update(preprocessing=self.preprocessing, scaler_sha256=self.scaler_sha256)
        if self.task_state_sha256:
            result.update(task=self.task_type, task_state_sha256=self.task_state_sha256)
        if self.task_type == "forecasting":
            result.update(target_features=self.target_features, n_pred_steps=self.n_pred_steps)
        if self.feature_semantics is not None:
            result["feature_semantics_sha256"] = self.feature_semantics_sha256
        return result

    def validate_semantics(self) -> None:
        """Check optional units while preserving legacy registry identities."""
        from .semantics import validate_semantic_binding

        validate_semantic_binding(self.feature_semantics, self.feature_semantics_sha256 or None,
                                  self.feature_columns, self.source_sha256)

    def validate_task_binding(self, source_csv, binding, task_state) -> None:
        """Check split-aware source, training state, and task-specific positions."""
        from .task_data import validate_task_state
        from .preprocessing import state_sha256

        validate_task_state(task_state, self.task_type)
        if (
            self.format_version != 2
            or file_sha256(source_csv) != self.source_sha256
            or binding != self.prediction_binding()
            or state_sha256(task_state) != self.task_state_sha256
            or task_state["roles"] != self.column_roles
            or task_state["window_size"] != self.window_size
            or self.n_features != len(self.feature_columns)
            or self.feature_columns != task_state["roles"]["feature_cols"]
        ):
            raise ValueError("Task prediction/source/state/registry binding mismatch")
        if self.preprocessing and state_sha256(task_state["scaler_state"]) != self.scaler_sha256:
            raise ValueError("Task scaler/registry binding mismatch")
        if task_state.get("feature_semantics") is not None:
            from .semantics import prepare_feature_semantics

            if self.feature_semantics is None:
                raise ValueError("Registry is missing declared training units")
            prepare_feature_semantics(self.feature_semantics["units"], self.feature_columns,
                                      self.source_sha256, task_state["feature_semantics"])
        selected = set(self.split_info.get("rows", {}).get(self.split_info.get("role"), range(self.source_row_count)))
        from .pipeline import _read_source

        source, _ = _read_source(source_csv, self.task_type, self.column_roles, True)
        id_col = self.column_roles["sample_id_col"]
        original_groups = (
            {sid: list(g.index) for sid, g in source.groupby(id_col, sort=True)} if id_col else {0: list(source.index)}
        )
        source = source.loc[source.index.isin(selected)]
        groups = (
            {sid: list(g.index) for sid, g in source.groupby(id_col, sort=True)} if id_col else {0: list(source.index)}
        )
        seen, samples = set(), set()
        for i, e in enumerate(self.entries):
            rows = groups.get(e.original_sample_id, [])
            if (
                e.window_id != i
                or not 0 < e.actual_length <= self.window_size
                or len(e.row_indices) != e.actual_length
                or len(set(e.row_indices)) != len(e.row_indices)
                or e.pad_length != self.window_size - e.actual_length
                or e.is_padded != bool(e.pad_length)
                or not set(e.row_indices).issubset(selected)
            ):
                raise ValueError("Invalid task input window coverage")
            if e.row_indices != rows[e.start_step : e.end_step]:
                raise ValueError("Task window/sample/source position mismatch")
            if self.task_type in ("classification", "clustering", "anomaly_detection", "representation"):
                if (
                    self.task_type != "anomaly_detection" and samples.intersection([e.original_sample_id])
                ) or seen.intersection(e.row_indices):
                    raise ValueError("Duplicate classification sample/rows")
                samples.add(e.original_sample_id)
                seen.update(e.row_indices)
            else:
                if (
                    self.n_pred_steps != task_state["n_pred_steps"]
                    or self.target_features != task_state["target_features"]
                    or len(e.target_row_indices) not in (0, self.n_pred_steps)
                    or not set(e.target_row_indices).issubset(selected)
                    or set(e.target_row_indices).intersection(e.row_indices)
                    or e.forecast_origin != original_groups[e.original_sample_id].index(e.row_indices[-1])
                    or e.target_row_indices != rows[e.end_step : e.end_step + self.n_pred_steps]
                ):
                    raise ValueError("Invalid future target coverage")
        if self.task_type in ("classification", "clustering", "anomaly_detection", "representation") and seen != selected:
            raise ValueError("Classification rows must be covered exactly once")

    def validate_binding(self, predictions: np.ndarray, source_csv: str | Path, binding: dict) -> None:
        """Validate a complete semantic prediction before any CSV is written.

        Parameters
        ----------
        predictions : np.ndarray
            Output tensor in registry window and feature order.
        source_csv : str or Path
            Unmodified source snapshot.
        binding : dict
            Metadata saved with the predictions.
        """
        if self.format_version != 2 or not self.source_sha256 or not self.columns:
            raise ValueError("Semantic export requires a version 2 source-aware registry")
        if file_sha256(source_csv) != self.source_sha256 or binding != self.prediction_binding():
            raise ValueError("Prediction/source/registry binding mismatch")
        if predictions.shape != (self.n_windows, self.window_size, self.n_features):
            raise ValueError("Prediction shape does not match registry")
        if len(self.feature_columns) != self.n_features or len(set(self.feature_columns)) != self.n_features:
            raise ValueError("Invalid registry feature order")
        rows = []
        for i, entry in enumerate(self.entries):
            if (
                entry.window_id != i
                or not 0 < entry.actual_length <= self.window_size
                or entry.pad_length != self.window_size - entry.actual_length
                or entry.is_padded != bool(entry.pad_length)
                or len(entry.row_indices) != entry.actual_length
            ):
                raise ValueError("Invalid window coverage or padding")
            rows.extend(entry.row_indices)
        if sorted(rows) != list(range(self.source_row_count)):
            raise ValueError("Source rows must be covered exactly once")


def file_sha256(path: str | Path) -> str:
    """Hash a local file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_serializer(obj: Any) -> Any:
    """JSON serializer for numpy types and other non-standard types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
