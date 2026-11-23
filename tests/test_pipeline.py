"""
Tests for ai4ts.data_protocols.profile, registry, and pipeline modules.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pandas as pd
import pytest

from ai4ts.data_protocols.profile import (
    DataProfile,
    analyze_dataset,
)
from ai4ts.data_protocols.registry import WindowEntry, WindowRegistry
from ai4ts.data_protocols.pipeline import (
    prepare_for_pypots,
    split_by_sample_id,
    reconstruct_from_predictions,
    _determine_strategy,
    _DIRECT_THRESHOLD,
)

# ──────────────────────────────────────────────────────────────
# Fixtures: reusable CSV files
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def uniform_csv(tmp_path):
    """CSV with uniform-length samples (10 samples × 8 steps × 3 features)."""
    rows = []
    rng = np.random.RandomState(42)
    for sid in range(10):
        for step in range(8):
            row = {"SAMPLE_ID": sid}
            for f in range(3):
                val = rng.randn()
                row[f"feat_{f}"] = val if rng.rand() > 0.15 else np.nan
            rows.append(row)
    path = tmp_path / "uniform.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def classification_csv(tmp_path):
    """CSV with variable-length classification samples."""
    rows = []
    rng = np.random.RandomState(42)
    lengths = [10, 15, 10, 20, 12, 15, 10, 18, 14, 10]
    for sid, length in enumerate(lengths):
        for step in range(length):
            row = {"SAMPLE_ID": sid}
            row["feat_0"] = rng.randn()
            row["feat_1"] = rng.randn()
            row["CLAF_TARGET"] = sid % 3
            rows.append(row)
    path = tmp_path / "classification.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def long_single_csv(tmp_path):
    """CSV with a single long sample (500 steps × 2 features)."""
    rng = np.random.RandomState(42)
    n_steps = 500
    rows = []
    for step in range(n_steps):
        rows.append(
            {
                "SAMPLE_ID": 0,
                "feat_0": rng.randn(),
                "feat_1": rng.randn(),
            }
        )
    path = tmp_path / "long_single.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def timestamp_csv(tmp_path):
    """CSV with TIMESTAMP column, regular frequency."""
    rng = np.random.RandomState(42)
    timestamps = pd.date_range("2024-01-01", periods=20, freq="h")
    rows = []
    for sid in range(2):
        for i, ts in enumerate(timestamps):
            rows.append(
                {
                    "SAMPLE_ID": sid,
                    "TIMESTAMP": ts,
                    "feat_0": rng.randn(),
                }
            )
    path = tmp_path / "timestamp.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


# ──────────────────────────────────────────────────────────────
# Profile Tests
# ──────────────────────────────────────────────────────────────


class TestAnalyzeDataset:
    def test_uniform_csv_stats(self, uniform_csv):
        profile = analyze_dataset(uniform_csv)
        assert profile.dataset_stats.n_samples == 10
        assert profile.dataset_stats.n_features == 3
        assert profile.dataset_stats.total_rows == 80
        assert profile.dataset_stats.min_sample_length == 8
        assert profile.dataset_stats.max_sample_length == 8
        assert profile.dataset_stats.has_variable_length is False

    def test_classification_detection(self, classification_csv):
        profile = analyze_dataset(classification_csv)
        assert profile.task_type == "classification"
        assert profile.schema_mapping.label_col == "CLAF_TARGET"
        assert profile.schema_mapping.n_classes == 3

    def test_explicit_task_type_override(self, uniform_csv):
        profile = analyze_dataset(uniform_csv, task_type="forecasting")
        assert profile.task_type == "forecasting"

    def test_variable_length_detection(self, classification_csv):
        profile = analyze_dataset(classification_csv)
        assert profile.dataset_stats.has_variable_length is True
        assert profile.dataset_stats.min_sample_length == 10
        assert profile.dataset_stats.max_sample_length == 20

    def test_missing_rate(self, uniform_csv):
        profile = analyze_dataset(uniform_csv)
        assert 0.0 < profile.dataset_stats.missing_rate < 1.0

    def test_timestamp_regular(self, timestamp_csv):
        profile = analyze_dataset(timestamp_csv)
        assert profile.timestamp_info.has_timestamp is True
        assert profile.timestamp_info.is_regular is True
        assert profile.timestamp_info.warning is None

    def test_profile_serialization_roundtrip(self, uniform_csv):
        profile = analyze_dataset(uniform_csv)
        d = profile.to_dict()
        assert isinstance(d, dict)
        restored = DataProfile.from_dict(d)
        assert restored.dataset_stats.n_samples == profile.dataset_stats.n_samples
        assert restored.schema_mapping.feature_cols == profile.schema_mapping.feature_cols

    def test_no_sample_id_column(self, tmp_path):
        """Single-sample CSV without explicit SAMPLE_ID."""
        df = pd.DataFrame({"feat_0": [1, 2, 3], "feat_1": [4, 5, 6]})
        path = tmp_path / "no_id.csv"
        df.to_csv(path, index=False)
        profile = analyze_dataset(str(path))
        assert profile.dataset_stats.n_samples == 1
        assert profile.dataset_stats.total_rows == 3


# ──────────────────────────────────────────────────────────────
# Registry Tests
# ──────────────────────────────────────────────────────────────


class TestWindowRegistry:
    def test_add_and_count(self):
        reg = WindowRegistry(window_size=10, strategy="direct")
        assert reg.n_windows == 0
        reg.add(WindowEntry(0, "s0", 0, 10, False, 0))
        reg.add(WindowEntry(1, "s1", 0, 8, True, 2))
        assert reg.n_windows == 2

    def test_original_sample_ids(self):
        reg = WindowRegistry(window_size=10, strategy="sliding_window")
        reg.add(WindowEntry(0, "A", 0, 10))
        reg.add(WindowEntry(1, "A", 10, 20))
        reg.add(WindowEntry(2, "B", 0, 10))
        assert reg.original_sample_ids == ["A", "B"]

    def test_get_entries_for_sample_sorted(self):
        reg = WindowRegistry(window_size=10, strategy="sliding_window")
        reg.add(WindowEntry(0, "A", 10, 20))
        reg.add(WindowEntry(1, "A", 0, 10))
        entries = reg.get_entries_for_sample("A")
        assert [e.start_step for e in entries] == [0, 10]

    def test_json_roundtrip(self, tmp_path):
        reg = WindowRegistry(window_size=50, n_features=3, task_type="imputation", strategy="sliding_window")
        reg.add(WindowEntry(0, 0, 0, 50, False, 0))
        reg.add(WindowEntry(1, 0, 50, 80, True, 20))

        path = tmp_path / "reg.json"
        reg.to_json(str(path))

        loaded = WindowRegistry.from_json(str(path))
        assert loaded.window_size == 50
        assert loaded.n_features == 3
        assert loaded.strategy == "sliding_window"
        assert loaded.n_windows == 2
        assert loaded.entries[1].is_padded is True
        assert loaded.entries[1].pad_length == 20

    def test_numpy_int_serialization(self, tmp_path):
        """Ensure numpy int types serialize to JSON without error."""
        reg = WindowRegistry(window_size=10)
        reg.add(WindowEntry(np.int64(0), np.int64(5), np.int64(0), np.int64(10)))
        path = tmp_path / "numpy_reg.json"
        reg.to_json(str(path))

        with open(path) as f:
            data = json.load(f)
        assert data["entries"][0]["window_id"] == 0


class TestWindowEntry:
    def test_actual_length(self):
        entry = WindowEntry(0, "s0", 5, 15, False, 0)
        assert entry.actual_length == 10

    def test_padded_actual_length(self):
        entry = WindowEntry(0, "s0", 0, 8, True, 2)
        assert entry.actual_length == 8


# ──────────────────────────────────────────────────────────────
# Strategy Tests
# ──────────────────────────────────────────────────────────────


class TestDetermineStrategy:
    def test_classification_always_pad_only(self):
        strategy, ws = _determine_strategy([10, 20, 15], True, None)
        assert strategy == "pad_only"
        assert ws == 20

    def test_uniform_short_is_direct(self):
        strategy, ws = _determine_strategy([50, 50, 50], False, None)
        assert strategy == "direct"
        assert ws == 50

    def test_uniform_long_is_sliding_window(self):
        strategy, ws = _determine_strategy([300, 300], False, None)
        assert strategy == "sliding_window"
        assert ws == _DIRECT_THRESHOLD

    def test_variable_length_is_sliding_window(self):
        strategy, ws = _determine_strategy([50, 100], False, None)
        assert strategy == "sliding_window"

    def test_explicit_window_size(self):
        strategy, ws = _determine_strategy([500, 500], False, 100)
        assert strategy == "sliding_window"
        assert ws == 100

    def test_explicit_window_size_small_data(self):
        """If data fits in one window and uniform, still direct."""
        strategy, ws = _determine_strategy([50, 50], False, 200)
        assert strategy == "direct"
        assert ws == 200


# ──────────────────────────────────────────────────────────────
# Pipeline Tests
# ──────────────────────────────────────────────────────────────


class TestPrepareForPyPOTS:
    def test_uniform_imputation(self, uniform_csv):
        result = prepare_for_pypots(uniform_csv, task="imputation", seed=42)
        X = result["X"]
        X_ori = result["X_ori"]
        reg = result["registry"]
        assert X.shape == (10, 8, 3)
        assert X_ori.shape == (10, 8, 3)
        assert reg.strategy == "direct"
        assert reg.n_windows == 10
        # train set: X == X_ori (no artificial missing)
        np.testing.assert_array_equal(X, X_ori)

    def test_val_set_has_more_missing(self, uniform_csv):
        result = prepare_for_pypots(uniform_csv, task="imputation", set_type="val", seed=42)
        X = result["X"]
        X_ori = result["X_ori"]
        x_nan = np.isnan(X).sum()
        xori_nan = np.isnan(X_ori).sum()
        assert x_nan >= xori_nan

    def test_classification_pad_only(self, classification_csv):
        result = prepare_for_pypots(classification_csv, task="classification", seed=42)
        X = result["X"]
        reg = result["registry"]
        y = result["y"]
        assert reg.strategy == "pad_only"
        assert X.shape[0] == 10
        assert X.shape[1] == 20  # max length
        assert y.shape == (10,)
        assert len(np.unique(y[~np.isnan(y)])) == 3

    def test_long_single_sliding_window(self, long_single_csv):
        result = prepare_for_pypots(long_single_csv, task="imputation", seed=42)
        X = result["X"]
        reg = result["registry"]
        assert reg.strategy == "sliding_window"
        assert X.shape[1] == _DIRECT_THRESHOLD
        # 500 / 200 = 2 full windows + 1 tail (100 steps padded to 200)
        assert reg.n_windows == 3

    def test_explicit_window_size(self, long_single_csv):
        result = prepare_for_pypots(long_single_csv, task="imputation", window_size=100, seed=42)
        X = result["X"]
        reg = result["registry"]
        assert X.shape[1] == 100
        assert reg.n_windows == 5  # 500 / 100 = 5 exact windows

    def test_profile_included(self, uniform_csv):
        result = prepare_for_pypots(uniform_csv, task="imputation", seed=42)
        profile = result["profile"]
        assert isinstance(profile, DataProfile)
        assert profile.dataset_stats.n_samples == 10


class TestSplitBySampleId:
    def test_basic_split(self, uniform_csv):
        df = pd.read_csv(uniform_csv)
        train_ids, val_ids, test_ids = split_by_sample_id(df, 0.7, 0.1, 0.2, seed=42)
        all_ids = train_ids + val_ids + test_ids
        assert len(set(all_ids)) == 10
        assert len(train_ids) == 7
        assert len(val_ids) == 1
        assert len(test_ids) == 2

    def test_no_overlap(self, uniform_csv):
        df = pd.read_csv(uniform_csv)
        train_ids, val_ids, test_ids = split_by_sample_id(df, 0.6, 0.2, 0.2, seed=42)
        assert len(set(train_ids) & set(val_ids)) == 0
        assert len(set(train_ids) & set(test_ids)) == 0
        assert len(set(val_ids) & set(test_ids)) == 0

    def test_bad_ratios_raises(self, uniform_csv):
        df = pd.read_csv(uniform_csv)
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            split_by_sample_id(df, 0.5, 0.5, 0.5, seed=42)


# ──────────────────────────────────────────────────────────────
# Reconstruction Tests
# ──────────────────────────────────────────────────────────────


class TestReconstructFromPredictions:
    def test_direct_exact_roundtrip(self, uniform_csv):
        """Direct strategy: reconstruction should perfectly match original."""
        result = prepare_for_pypots(uniform_csv, task="imputation", seed=42)
        X_ori = result["X_ori"]
        registry = result["registry"]

        reconstructed = reconstruct_from_predictions(X_ori, registry)
        assert len(reconstructed) == 10
        for sid in range(10):
            assert reconstructed[str(sid)].shape == (8, 3)

    def test_classification_strips_padding(self, classification_csv):
        """Pad-only: reconstruction must strip padding and recover original lengths."""
        result = prepare_for_pypots(classification_csv, task="classification", seed=42)
        X_ori = result["X_ori"]
        registry = result["registry"]

        reconstructed = reconstruct_from_predictions(X_ori, registry)
        expected_lengths = [10, 15, 10, 20, 12, 15, 10, 18, 14, 10]
        for sid, expected_len in enumerate(expected_lengths):
            assert (
                reconstructed[str(sid)].shape[0] == expected_len
            ), f"Sample {sid}: expected {expected_len} but got {reconstructed[str(sid)].shape[0]}"

    def test_sliding_window_reconstruction(self, long_single_csv):
        """Sliding window: reconstruction should exactly match original length."""
        result = prepare_for_pypots(long_single_csv, task="imputation", seed=42)
        X_ori = result["X_ori"]
        registry = result["registry"]

        reconstructed = reconstruct_from_predictions(X_ori, registry)
        assert "0" in reconstructed
        assert reconstructed["0"].shape[0] == 500

    def test_shape_mismatch_raises(self):
        reg = WindowRegistry(window_size=10)
        reg.add(WindowEntry(0, "A", 0, 10))
        fake_pred = np.zeros((5, 10, 3))  # 5 windows but registry has 1
        with pytest.raises(ValueError, match="Prediction shape does not match"):
            reconstruct_from_predictions(fake_pred, reg)

    def test_sliding_window_values_match(self, long_single_csv):
        """Verify actual values are preserved through prepare → reconstruct."""
        df = pd.read_csv(long_single_csv)
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        original = df[feature_cols].values

        result = prepare_for_pypots(long_single_csv, task="imputation", seed=42)
        reconstructed = reconstruct_from_predictions(result["X_ori"], result["registry"])

        np.testing.assert_allclose(
            reconstructed["0"], original, rtol=1e-10, err_msg="Reconstructed values don't match original"
        )
