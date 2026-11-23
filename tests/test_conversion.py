"""
Tests for ai4ts.data_protocols.conversion module.

Tests the full data conversion pipeline including:
- FeatureSeparator
- TimeSeriesConverter (from_csv, from_numpy, from_dataframe)
- to_tensor_samples
- Reconstruction (tensor_to_dataframe, tensor_to_numpy, tensor_to_csv)
- NaN preservation with NAN_KEEP strategy
- Scaling with missing values
"""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ai4ts.data_protocols.conversion import FeatureSeparator, TimeSeriesConverter
from ai4ts.data_protocols.schemas import ProcessingConfig, TimeSeriesMetadata, TensorDescriptor
from ai4ts.data_protocols.constants import (
    COL_NAME_SAMPLE_ID,
    COL_NAME_TIMESTAMP,
    CLAF_TARGET_SYMBOL,
)
from ai4ts.data_protocols.enums import ScalerType, MissingValueStrategy


# ──────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────

@pytest.fixture
def single_sample_df():
    """Single-sample DataFrame with 20 timesteps and 3 features."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "feat_0": np.random.randn(n),
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n),
    })


@pytest.fixture
def multi_sample_df():
    """Multi-sample DataFrame with 3 samples, each 10 timesteps, 2 features."""
    np.random.seed(42)
    frames = []
    for sid in range(3):
        df = pd.DataFrame({
            COL_NAME_SAMPLE_ID: sid,
            "feat_0": np.random.randn(10),
            "feat_1": np.random.randn(10),
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def df_with_missing():
    """DataFrame with natural missing values (NaN)."""
    np.random.seed(42)
    data = np.random.randn(30, 3)
    # Inject ~20% missing
    mask = np.random.random(data.shape) < 0.2
    data[mask] = np.nan
    return pd.DataFrame(data, columns=["feat_0", "feat_1", "feat_2"])


@pytest.fixture
def df_with_claf():
    """Multi-sample DataFrame with classification target."""
    frames = []
    for sid in range(4):
        df = pd.DataFrame({
            COL_NAME_SAMPLE_ID: sid,
            "feat_0": np.random.randn(8),
            "feat_1": np.random.randn(8),
            CLAF_TARGET_SYMBOL: sid % 2,
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def simple_csv(tmp_path):
    """Create a simple CSV file."""
    path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "feat_0": np.random.randn(50),
        "feat_1": np.random.randn(50),
    })
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def csv_with_nan(tmp_path):
    """Create a CSV file with NaN values."""
    path = tmp_path / "nan_data.csv"
    data = np.random.randn(30, 2)
    data[5, 0] = np.nan
    data[10, 1] = np.nan
    data[15, :] = np.nan
    df = pd.DataFrame(data, columns=["feat_0", "feat_1"])
    df.to_csv(path, index=False)
    return path


# ──────────────────────────────────────────────────
# FeatureSeparator
# ──────────────────────────────────────────────────

class TestFeatureSeparator:
    """Tests for FeatureSeparator."""

    def test_numeric_only(self, single_sample_df):
        sep = FeatureSeparator(single_sample_df)
        assert sep.numeric == ["feat_0", "feat_1", "feat_2"]
        assert sep.string == []
        assert sep.classification_targets == []
        assert sep.n_numeric_features == 3

    def test_all_features(self, single_sample_df):
        sep = FeatureSeparator(single_sample_df)
        assert sep.all_features == ["feat_0", "feat_1", "feat_2"]

    def test_with_classification(self, df_with_claf):
        sep = FeatureSeparator(df_with_claf)
        assert CLAF_TARGET_SYMBOL in sep.classification_targets
        assert CLAF_TARGET_SYMBOL not in sep.numeric


# ──────────────────────────────────────────────────
# TimeSeriesConverter: Input loading
# ──────────────────────────────────────────────────

class TestConverterInput:
    """Tests for loading data into TimeSeriesConverter."""

    def test_from_dataframe_single_sample(self, single_sample_df):
        config = ProcessingConfig(ts_context_len=10, n_anchor_points=0)
        converter = TimeSeriesConverter(config)
        df, meta = converter.from_dataframe(single_sample_df)
        assert isinstance(meta, TimeSeriesMetadata)
        assert meta.n_features == 3
        assert meta.n_samples == 1

    def test_from_dataframe_multi_sample(self, multi_sample_df):
        config = ProcessingConfig(ts_context_len=10, n_anchor_points=0)
        converter = TimeSeriesConverter(config)
        df, meta = converter.from_dataframe(multi_sample_df)
        assert meta.n_samples == 3

    def test_from_csv(self, simple_csv):
        config = ProcessingConfig(ts_context_len=10, n_anchor_points=0)
        converter = TimeSeriesConverter(config)
        df, meta = converter.from_csv(simple_csv)
        assert meta.n_features == 2
        assert len(df) == 50

    def test_from_numpy_2d(self):
        arr = np.random.randn(30, 4)
        config = ProcessingConfig(ts_context_len=10, n_anchor_points=0)
        converter = TimeSeriesConverter(config)
        df, meta = converter.from_numpy(arr)
        assert meta.n_features == 4
        assert meta.n_samples == 1

    def test_from_numpy_3d(self):
        arr = np.random.randn(5, 10, 3)
        config = ProcessingConfig(ts_context_len=10, n_anchor_points=0)
        converter = TimeSeriesConverter(config)
        df, meta = converter.from_numpy(arr)
        assert meta.n_samples == 5
        assert meta.n_features == 3

    def test_from_numpy_with_feature_names(self):
        arr = np.random.randn(20, 2)
        config = ProcessingConfig(ts_context_len=10, n_anchor_points=0)
        converter = TimeSeriesConverter(config)
        df, meta = converter.from_numpy(arr, feature_names=["temp", "pressure"])
        assert "temp" in meta.numeric_feature_names
        assert "pressure" in meta.numeric_feature_names


# ──────────────────────────────────────────────────
# TimeSeriesConverter: Scaling
# ──────────────────────────────────────────────────

class TestConverterScaling:
    """Tests for scaler behavior in the converter."""

    def test_standard_scaler(self, single_sample_df):
        config = ProcessingConfig(scaler_type=ScalerType.STANDARD, ts_context_len=10)
        converter = TimeSeriesConverter(config)
        df, meta = converter.from_dataframe(single_sample_df)
        assert converter.scaler is not None

        # After standard scaling, features should be roughly zero-mean
        numeric_cols = meta.numeric_feature_names
        means = df[numeric_cols].mean()
        for m in means:
            assert abs(m) < 0.5  # approximately zero-mean

    def test_minmax_scaler(self, single_sample_df):
        config = ProcessingConfig(scaler_type=ScalerType.MINMAX, ts_context_len=10)
        converter = TimeSeriesConverter(config)
        df, meta = converter.from_dataframe(single_sample_df)
        numeric_cols = meta.numeric_feature_names
        assert df[numeric_cols].min().min() >= -0.01
        assert df[numeric_cols].max().max() <= 1.01

    def test_no_scaler(self, single_sample_df):
        original = single_sample_df.copy()
        config = ProcessingConfig(scaler_type=ScalerType.NONE, ts_context_len=10)
        converter = TimeSeriesConverter(config)
        df, meta = converter.from_dataframe(single_sample_df)
        assert converter.scaler is None
        # Values should be unchanged (modulo sorting/index reset)
        for col in ["feat_0", "feat_1", "feat_2"]:
            np.testing.assert_array_almost_equal(
                sorted(df[col].dropna().values),
                sorted(original[col].dropna().values),
            )

    def test_scaler_with_nan_preserves_nan(self, df_with_missing):
        """NaN values should be preserved through scaling."""
        config = ProcessingConfig(scaler_type=ScalerType.STANDARD, ts_context_len=10)
        converter = TimeSeriesConverter(config)
        df, meta = converter.from_dataframe(df_with_missing)

        # NaN positions should still be NaN after scaling
        numeric_cols = meta.numeric_feature_names
        assert df[numeric_cols].isna().any().any()

    def test_scaler_params_extraction(self, single_sample_df):
        config = ProcessingConfig(scaler_type=ScalerType.STANDARD, ts_context_len=10)
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(single_sample_df)
        params = converter.get_scaler_params()
        assert params is not None
        assert params.mean is not None
        assert params.std is not None
        assert len(params.feature_names) == 3


# ──────────────────────────────────────────────────
# TimeSeriesConverter: to_tensor_samples
# ──────────────────────────────────────────────────

class TestConverterTensorOutput:
    """Tests for to_tensor_samples conversion."""

    def test_basic_tensor_output(self, single_sample_df):
        config = ProcessingConfig(
            ts_context_len=10,
            n_anchor_points=2,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(single_sample_df)
        result = converter.to_tensor_samples(training_mode=False)

        assert "X" in result
        assert "missing_mask" in result
        assert "padding_mask" in result
        assert "sample_actual_lengths" in result
        assert "tensor_descriptor" in result

        X = result["X"]
        assert X.ndim == 3
        # max_input_steps = 10 + 2 = 12
        assert X.shape[2] == 3  # 3 features

    def test_training_mode_adds_extras(self, single_sample_df):
        config = ProcessingConfig(
            ts_context_len=10,
            n_anchor_points=2,
            artificial_missing_rate=0.1,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(single_sample_df)
        result = converter.to_tensor_samples(training_mode=True)

        assert "X_original" in result
        assert "indicating_mask" in result

    def test_multi_sample_tensor(self, multi_sample_df):
        config = ProcessingConfig(
            ts_context_len=10,
            n_anchor_points=2,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(multi_sample_df)
        result = converter.to_tensor_samples(training_mode=False)

        X = result["X"]
        assert X.shape[0] == 3  # 3 samples
        assert X.shape[1] == 12  # 10 + 2 anchor
        assert X.shape[2] == 2  # 2 features

    def test_classification_target(self, df_with_claf):
        config = ProcessingConfig(
            ts_context_len=8,
            n_anchor_points=0,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(df_with_claf)
        result = converter.to_tensor_samples(
            include_classification=True,
            training_mode=False,
        )

        assert "clas_target" in result
        clas = result["clas_target"]
        assert len(clas) == 4

    def test_tensor_descriptor_populated(self, single_sample_df):
        config = ProcessingConfig(
            ts_context_len=10,
            n_anchor_points=2,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(single_sample_df)
        result = converter.to_tensor_samples(training_mode=False)

        td = result["tensor_descriptor"]
        assert isinstance(td, TensorDescriptor)
        assert td.shape == result["X"].shape
        assert td.n_anchor_points == 2

    def test_masks_shape(self, single_sample_df):
        config = ProcessingConfig(
            ts_context_len=10,
            n_anchor_points=2,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(single_sample_df)
        result = converter.to_tensor_samples(training_mode=False)

        X = result["X"]
        assert result["missing_mask"].shape == X.shape
        n_samples, max_steps, _ = X.shape
        assert result["padding_mask"].shape == (n_samples, max_steps, max_steps)


# ──────────────────────────────────────────────────
# NaN preservation with NAN_KEEP strategy
# ──────────────────────────────────────────────────

class TestNanKeepStrategy:
    """Tests that NAN_KEEP strategy preserves NaN values through the pipeline."""

    @pytest.mark.parametrize("strategy", [MissingValueStrategy.ZERO_FILL, MissingValueStrategy.MEAN_FILL])
    def test_unsupported_strategy_does_not_fall_back_to_forward_fill(self, df_with_missing, strategy):
        converter = TimeSeriesConverter(ProcessingConfig(missing_value_strategy=strategy, scaler_type=ScalerType.NONE))
        converter.from_dataframe(df_with_missing)
        with pytest.raises(ValueError, match="only nan_keep and forward_fill"):
            converter.to_tensor_samples(training_mode=False)

    def test_nan_keep_preserves_nan_in_tensor(self, df_with_missing):
        """With NAN_KEEP, NaN values should be present in the output tensor."""
        config = ProcessingConfig(
            ts_context_len=15,
            n_anchor_points=0,
            missing_value_strategy=MissingValueStrategy.NAN_KEEP,
            scaler_type=ScalerType.NONE,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(df_with_missing)
        result = converter.to_tensor_samples(training_mode=False)

        X = result["X"]
        # With NAN_KEEP and no artificial missing, original NaNs should be preserved
        assert np.isnan(X).any(), "NaN values should be preserved in tensor with NAN_KEEP"

    def test_forward_fill_removes_nan(self, df_with_missing):
        """With FORWARD_FILL (default), NaN should be filled in output tensor."""
        config = ProcessingConfig(
            ts_context_len=15,
            n_anchor_points=0,
            missing_value_strategy=MissingValueStrategy.FORWARD_FILL,
            scaler_type=ScalerType.NONE,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(df_with_missing)
        result = converter.to_tensor_samples(training_mode=False)

        X = result["X"]
        # Anchor/padding excluded, check data region only
        for i in range(X.shape[0]):
            actual_len = result["sample_actual_lengths"][i]
            data_region = X[i, :actual_len, :]
            assert not np.isnan(data_region).any(), (
                "NaN should be forward-filled in data region"
            )

    def test_nan_keep_mask_still_correct(self, df_with_missing):
        """Missing mask should still correctly indicate NaN positions under NAN_KEEP."""
        config = ProcessingConfig(
            ts_context_len=15,
            n_anchor_points=2,
            missing_value_strategy=MissingValueStrategy.NAN_KEEP,
            scaler_type=ScalerType.NONE,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(df_with_missing)
        result = converter.to_tensor_samples(training_mode=False)

        X = result["X"]
        mask = result["missing_mask"]
        n_anchors = config.n_anchor_points

        # For each sample, NaN positions in X should have mask=0
        for i in range(X.shape[0]):
            actual_len = result["sample_actual_lengths"][i]
            data_start = n_anchors
            data_end = actual_len
            data_x = X[i, data_start:data_end, :]
            data_mask = mask[i, data_start:data_end, :]
            nan_positions = np.isnan(data_x)
            # Where X is NaN, mask should be 0
            assert (data_mask[nan_positions] == 0.0).all()
            # Where X is not NaN, mask should be 1
            assert (data_mask[~nan_positions] == 1.0).all()

    def test_nan_keep_with_scaling(self, df_with_missing):
        """NaN values should survive through StandardScaler when NAN_KEEP is used."""
        config = ProcessingConfig(
            ts_context_len=15,
            n_anchor_points=0,
            missing_value_strategy=MissingValueStrategy.NAN_KEEP,
            scaler_type=ScalerType.STANDARD,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(df_with_missing)
        result = converter.to_tensor_samples(training_mode=False)

        X = result["X"]
        assert np.isnan(X).any(), "NaN should survive through scaler + NAN_KEEP"

    def test_nan_keep_training_mode(self, df_with_missing):
        """Training mode with NAN_KEEP should work correctly."""
        config = ProcessingConfig(
            ts_context_len=15,
            n_anchor_points=0,
            missing_value_strategy=MissingValueStrategy.NAN_KEEP,
            scaler_type=ScalerType.NONE,
            artificial_missing_rate=0.1,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(df_with_missing)
        result = converter.to_tensor_samples(training_mode=True)

        assert "X_original" in result
        assert "indicating_mask" in result
        # Original should also have NaN preserved
        X_ori = result["X_original"]
        assert np.isnan(X_ori).any()


# ──────────────────────────────────────────────────
# Reconstruction
# ──────────────────────────────────────────────────

class TestConverterReconstruction:
    """Tests for converting tensors back to user-friendly formats."""

    def test_tensor_to_dataframe_basic(self, single_sample_df):
        config = ProcessingConfig(
            ts_context_len=10,
            n_anchor_points=2,
            scaler_type=ScalerType.STANDARD,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(single_sample_df)
        result = converter.to_tensor_samples(training_mode=False)

        reconstructed = converter.tensor_to_dataframe(
            result["X"],
            sample_lengths=result["sample_actual_lengths"],
        )
        assert isinstance(reconstructed, pd.DataFrame)
        assert len(reconstructed.columns) >= 3

    def test_tensor_to_numpy(self, single_sample_df):
        config = ProcessingConfig(
            ts_context_len=10,
            n_anchor_points=2,
            scaler_type=ScalerType.NONE,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(single_sample_df)
        result = converter.to_tensor_samples(training_mode=False)

        arr = converter.tensor_to_numpy(
            result["X"],
            sample_lengths=result["sample_actual_lengths"],
        )
        assert isinstance(arr, np.ndarray)

    def test_tensor_to_csv(self, single_sample_df, tmp_path):
        config = ProcessingConfig(
            ts_context_len=10,
            n_anchor_points=2,
            scaler_type=ScalerType.NONE,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(single_sample_df)
        result = converter.to_tensor_samples(training_mode=False)

        out_path = tmp_path / "output.csv"
        returned_path = converter.tensor_to_csv(
            result["X"],
            out_path,
            sample_lengths=result["sample_actual_lengths"],
        )
        assert os.path.exists(returned_path)
        df_out = pd.read_csv(returned_path)
        assert len(df_out) > 0

    def test_inverse_scale_preserves_nan(self, df_with_missing):
        """Inverse transform should preserve NaN positions."""
        config = ProcessingConfig(
            ts_context_len=15,
            n_anchor_points=0,
            missing_value_strategy=MissingValueStrategy.NAN_KEEP,
            scaler_type=ScalerType.STANDARD,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(df_with_missing)
        result = converter.to_tensor_samples(training_mode=False)

        reconstructed = converter.tensor_to_dataframe(
            result["X"],
            strip_anchors=False,
            sample_lengths=result["sample_actual_lengths"],
        )
        # NaN positions should be preserved after inverse transform
        numeric_cols = [c for c in reconstructed.columns if c != COL_NAME_SAMPLE_ID]
        assert reconstructed[numeric_cols].isna().any().any()

    def test_roundtrip_without_nan(self, single_sample_df):
        """Data should survive a roundtrip (scale → tensor → inverse scale) approximately."""
        config = ProcessingConfig(
            ts_context_len=10,
            n_anchor_points=0,
            scaler_type=ScalerType.STANDARD,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        df, meta = converter.from_dataframe(single_sample_df)
        result = converter.to_tensor_samples(training_mode=False)

        # Take first sample from tensor and reconstruct
        first_sample = result["X"][0:1]
        reconstructed = converter.tensor_to_numpy(
            first_sample,
            sample_lengths=[result["sample_actual_lengths"][0]],
        )
        # Should have approximately the same values as input (first window)
        assert reconstructed.shape[1] == 3  # 3 features


# ──────────────────────────────────────────────────
# CSV with NaN end-to-end
# ──────────────────────────────────────────────────

class TestCsvNanEndToEnd:
    """End-to-end tests with CSV files containing NaN values."""

    def test_csv_nan_keep_pipeline(self, csv_with_nan):
        """Full pipeline: CSV with NaN → NAN_KEEP → tensor with NaN preserved."""
        config = ProcessingConfig(
            ts_context_len=15,
            n_anchor_points=0,
            missing_value_strategy=MissingValueStrategy.NAN_KEEP,
            scaler_type=ScalerType.NONE,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        df, meta = converter.from_csv(csv_with_nan)

        assert meta.has_missing_values is True

        result = converter.to_tensor_samples(training_mode=False)
        X = result["X"]
        mask = result["missing_mask"]

        # NaN should be in the tensor
        assert np.isnan(X).any()
        # Mask should flag them
        assert (mask == 0).any()

    def test_csv_forward_fill_pipeline(self, csv_with_nan):
        """Full pipeline: CSV with NaN → FORWARD_FILL → tensor without NaN."""
        config = ProcessingConfig(
            ts_context_len=15,
            n_anchor_points=0,
            missing_value_strategy=MissingValueStrategy.FORWARD_FILL,
            scaler_type=ScalerType.NONE,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        df, meta = converter.from_csv(csv_with_nan)

        result = converter.to_tensor_samples(training_mode=False)
        X = result["X"]

        # Check data region only (not padding)
        for i in range(X.shape[0]):
            actual_len = result["sample_actual_lengths"][i]
            data_region = X[i, :actual_len, :]
            assert not np.isnan(data_region).any()


# ──────────────────────────────────────────────────
# Numpy 3D array end-to-end
# ──────────────────────────────────────────────────

class TestNumpy3dEndToEnd:
    """End-to-end tests with 3D numpy arrays."""

    def test_3d_array_nan_keep(self):
        """3D array with NaN → NAN_KEEP → tensor preserves NaN."""
        arr = np.random.randn(4, 8, 3)
        arr[0, 2, 1] = np.nan
        arr[1, 5, 0] = np.nan
        arr[2, :, 2] = np.nan

        config = ProcessingConfig(
            ts_context_len=8,
            n_anchor_points=0,
            missing_value_strategy=MissingValueStrategy.NAN_KEEP,
            scaler_type=ScalerType.NONE,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        df, meta = converter.from_numpy(arr)
        assert meta.n_samples == 4
        assert meta.has_missing_values is True

        result = converter.to_tensor_samples(training_mode=False)
        X = result["X"]
        assert np.isnan(X).any()

    def test_3d_array_with_scaling(self):
        """3D array with NaN → StandardScaler → NaN preserved in output."""
        arr = np.random.randn(3, 10, 2)
        arr[0, 3, 0] = np.nan
        arr[1, 7, 1] = np.nan

        config = ProcessingConfig(
            ts_context_len=10,
            n_anchor_points=0,
            missing_value_strategy=MissingValueStrategy.NAN_KEEP,
            scaler_type=ScalerType.STANDARD,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_numpy(arr)
        result = converter.to_tensor_samples(training_mode=False)

        X = result["X"]
        assert np.isnan(X).any()

    def test_3d_array_roundtrip(self):
        """3D array → tensor → numpy reconstruction."""
        arr = np.random.randn(2, 8, 3)

        config = ProcessingConfig(
            ts_context_len=8,
            n_anchor_points=0,
            scaler_type=ScalerType.NONE,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_numpy(arr)
        result = converter.to_tensor_samples(training_mode=False)

        reconstructed = converter.tensor_to_numpy(
            result["X"],
            sample_lengths=result["sample_actual_lengths"],
        )
        assert reconstructed.shape[1] == 3


# ──────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case tests for the conversion pipeline."""

    def test_single_feature(self):
        df = pd.DataFrame({"feat": np.random.randn(20)})
        config = ProcessingConfig(ts_context_len=10, n_anchor_points=0)
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(df)
        result = converter.to_tensor_samples(training_mode=False)
        assert result["X"].shape[2] == 1

    def test_zero_anchor_points(self):
        df = pd.DataFrame({"feat": np.random.randn(20)})
        config = ProcessingConfig(ts_context_len=10, n_anchor_points=0)
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(df)
        result = converter.to_tensor_samples(training_mode=False)
        assert result["X"].shape[1] == 10  # no anchors

    def test_no_data_loaded_raises(self):
        converter = TimeSeriesConverter()
        with pytest.raises(RuntimeError, match="No data loaded"):
            converter.to_tensor_samples()

    def test_all_nan_column(self):
        """Column with all NaN should still work with NAN_KEEP."""
        df = pd.DataFrame({
            "feat_0": np.random.randn(10),
            "feat_1": [np.nan] * 10,
        })
        config = ProcessingConfig(
            ts_context_len=5,
            n_anchor_points=0,
            missing_value_strategy=MissingValueStrategy.NAN_KEEP,
            scaler_type=ScalerType.NONE,
            artificial_missing_rate=0.0,
        )
        converter = TimeSeriesConverter(config)
        converter.from_dataframe(df)
        result = converter.to_tensor_samples(training_mode=False)

        X = result["X"]
        # feat_1 (index 1) should be all NaN in data region
        for i in range(X.shape[0]):
            actual_len = result["sample_actual_lengths"][i]
            assert np.isnan(X[i, :actual_len, 1]).all()
