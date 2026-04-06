"""
Window registry for tracking how original time series samples are sliced into
fixed-length windows during data preparation.

The registry enables exact reconstruction: after model inference on fixed-length
windows, the registry maps each window back to its original sample and position,
allowing padding to be stripped and windows to be reassembled in order.
"""

from __future__ import annotations

import json
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
        return {
            "window_size": self.window_size,
            "n_features": self.n_features,
            "task_type": self.task_type,
            "strategy": self.strategy,
            "entries": [asdict(e) for e in self.entries],
        }

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
        )
        for entry_dict in d.get("entries", []):
            reg.add(WindowEntry(**entry_dict))
        return reg

    @classmethod
    def from_json(cls, path: str | Path) -> "WindowRegistry":
        """Load registry from a JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)


def _json_serializer(obj: Any) -> Any:
    """JSON serializer for numpy types and other non-standard types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
