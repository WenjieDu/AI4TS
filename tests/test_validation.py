"""
Tests for ai4ts.data_protocols.validation module.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ai4ts.data_protocols.validation import (
    ValidationError,
    validate_csv_file,
    validate_numpy_array,
    validate_dataframe,
    validate_processing_config,
)
from ai4ts.data_protocols.schemas import ProcessingConfig, TimeSeriesMetadata
from ai4ts.data_protocols.constants import (
    COL_NAME_SAMPLE_ID,
    COL_NAME_TIMESTAMP,
    CLAF_TARGET_SYMBOL,
)
from ai4ts.data_protocols.enums import ColumnRole, MissingValueStrategy


# ──────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────

@pytest.fixture
def simple_csv(tmp_path):
    """Create a simple CSV file for testing."""
    path = tmp_path / "test.csv"
    df = pd.DataFrame({
        "feat_0": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat_1": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def csv_with_missing(tmp_path):
    """Create a CSV file with missing values."""
    path = tmp_path / "missing.csv"
    df = pd.DataFrame({
        "feat_0": [1.0, np.nan, 3.0, np.nan, 5.0],
        "feat_1": [np.nan, 20.0, np.nan, 40.0, 50.0],
    })
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def csv_with_sample_id(tmp_path):
    """Create a CSV with SAMPLE_ID column."""
    path = tmp_path / "multi_sample.csv"
    df = pd.DataFrame({
        COL_NAME_SAMPLE_ID: [0, 0, 0, 1, 1, 1],
        "feat_0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feat_1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def csv_with_claf(tmp_path):
    """Create a CSV with classification target."""
    path = tmp_path / "claf.csv"
    df = pd.DataFrame({
        COL_NAME_SAMPLE_ID: [0, 0, 1, 1],
        "feat_0": [1.0, 2.0, 3.0, 4.0],
        CLAF_TARGET_SYMBOL: [0, 0, 1, 1],
    })
    df.to_csv(path, index=False)
    return path


# ──────────────────────────────────────────────────
# validate_csv_file
# ──────────────────────────────────────────────────

class TestValidateCsvFile:
    """Tests for validate_csv_file."""

    def test_valid_csv(self, simple_csv):
        df = validate_csv_file(simple_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "feat_0" in df.columns

    def test_nonexistent_file(self, tmp_path):
        with pytest.raises(ValidationError, match="File not found"):
            validate_csv_file(tmp_path / "nonexistent.csv")

    def test_wrong_extension(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text("{}")
        with pytest.raises(ValidationError, match="Unsupported file extension"):
            validate_csv_file(path)

    def test_empty_csv(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("a,b\n")
        with pytest.raises(ValidationError, match="no data rows"):
            validate_csv_file(path)

    def test_csv_with_missing_values(self, csv_with_missing):
        df = validate_csv_file(csv_with_missing)
        assert df["feat_0"].isna().sum() == 2
        assert df["feat_1"].isna().sum() == 2


# ──────────────────────────────────────────────────
# validate_numpy_array
# ──────────────────────────────────────────────────

class TestValidateNumpyArray:
    """Tests for validate_numpy_array."""

    def test_valid_2d_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        df, names = validate_numpy_array(arr)
        assert isinstance(df, pd.DataFrame)
        assert names == ["feat_0", "feat_1"]
        assert len(df) == 3

    def test_custom_feature_names(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        df, names = validate_numpy_array(arr, feature_names=["x", "y"])
        assert names == ["x", "y"]
        assert list(df.columns) == ["x", "y"]

    def test_wrong_feature_names_length(self):
        arr = np.array([[1.0, 2.0]])
        with pytest.raises(ValidationError, match="feature_names length"):
            validate_numpy_array(arr, feature_names=["a", "b", "c"])

    def test_1d_array_rejected(self):
        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValidationError, match="2D.*or 3D"):
            validate_numpy_array(arr)

    def test_empty_array(self):
        arr = np.empty((0, 3))
        with pytest.raises(ValidationError, match="must not be empty"):
            validate_numpy_array(arr)

    def test_non_numeric_array(self):
        arr = np.array([["a", "b"], ["c", "d"]])
        with pytest.raises(ValidationError, match="must be numeric"):
            validate_numpy_array(arr)

    def test_3d_array(self):
        arr = np.random.randn(5, 10, 3)
        df, names = validate_numpy_array(arr)
        assert names == ["feat_0", "feat_1", "feat_2"]
        assert len(df) == 50  # 5 samples * 10 timesteps
        assert COL_NAME_SAMPLE_ID in df.columns
        assert df[COL_NAME_SAMPLE_ID].nunique() == 5

    def test_3d_array_with_sample_ids(self):
        arr = np.random.randn(3, 8, 2)
        df, names = validate_numpy_array(arr, sample_ids=["a", "b", "c"])
        assert set(df[COL_NAME_SAMPLE_ID].unique()) == {"a", "b", "c"}

    def test_3d_wrong_sample_ids_length(self):
        arr = np.random.randn(3, 8, 2)
        with pytest.raises(ValidationError, match="sample_ids length"):
            validate_numpy_array(arr, sample_ids=[0, 1])

    def test_3d_empty_array(self):
        arr = np.empty((0, 10, 3))
        with pytest.raises(ValidationError, match="must not be empty"):
            validate_numpy_array(arr)

    def test_2d_nan_preserved(self):
        arr = np.array([[1.0, np.nan], [np.nan, 4.0]])
        df, _ = validate_numpy_array(arr)
        assert df.isna().sum().sum() == 2

    def test_3d_nan_preserved(self):
        arr = np.array([[[1.0, np.nan], [np.nan, 4.0]]])
        df, _ = validate_numpy_array(arr)
        assert df["feat_0"].isna().sum() == 1
        assert df["feat_1"].isna().sum() == 1


# ──────────────────────────────────────────────────
# validate_dataframe
# ──────────────────────────────────────────────────

class TestValidateDataframe:
    """Tests for validate_dataframe."""

    def test_basic_dataframe(self):
        df = pd.DataFrame({"feat_0": [1.0, 2.0, 3.0], "feat_1": [4.0, 5.0, 6.0]})
        meta = validate_dataframe(df)
        assert isinstance(meta, TimeSeriesMetadata)
        assert meta.n_samples == 1  # auto-assigned single SAMPLE_ID
        assert meta.n_features == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            validate_dataframe(df)

    def test_no_numeric_features(self):
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        with pytest.raises(ValidationError, match="No numeric feature"):
            validate_dataframe(df)

    def test_multi_sample_metadata(self):
        df = pd.DataFrame({
            COL_NAME_SAMPLE_ID: [0, 0, 0, 1, 1],
            "feat": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        meta = validate_dataframe(df)
        assert meta.n_samples == 2
        assert meta.n_timesteps_per_sample == [3, 2]

    def test_missing_value_detection(self):
        df = pd.DataFrame({
            "feat_0": [1.0, np.nan, 3.0],
            "feat_1": [np.nan, 2.0, np.nan],
        })
        meta = validate_dataframe(df)
        assert meta.has_missing_values is True
        assert meta.overall_missing_rate == pytest.approx(0.5)

    def test_classification_target_detected(self):
        df = pd.DataFrame({
            COL_NAME_SAMPLE_ID: [0, 0],
            "feat": [1.0, 2.0],
            CLAF_TARGET_SYMBOL: [0, 1],
        })
        meta = validate_dataframe(df)
        assert meta.has_classification_target is True
        assert CLAF_TARGET_SYMBOL in meta.classification_target_names

    def test_column_descriptors_populated(self):
        df = pd.DataFrame({
            "feat_0": [1.0, np.nan, 3.0],
            "feat_1": [4.0, 5.0, 6.0],
        })
        meta = validate_dataframe(df)
        feat_cols = [c for c in meta.columns if c.role == ColumnRole.FEATURE]
        assert len(feat_cols) == 2
        missing_col = [c for c in feat_cols if c.name == "feat_0"][0]
        assert missing_col.has_missing is True
        assert missing_col.missing_count == 1

    def test_config_validation_rejects_oversized_samples(self):
        df = pd.DataFrame({
            COL_NAME_SAMPLE_ID: [0] * 200 + [1] * 200,
            "feat": list(range(400)),
        })
        config = ProcessingConfig(ts_context_len=100)
        with pytest.raises(ValidationError, match="exceeds ts_context_len"):
            validate_dataframe(df, config)


# ──────────────────────────────────────────────────
# validate_processing_config
# ──────────────────────────────────────────────────

class TestValidateProcessingConfig:
    """Tests for validate_processing_config."""

    def test_valid_config(self):
        config = ProcessingConfig()
        meta = TimeSeriesMetadata(
            n_samples=1,
            n_timesteps_per_sample=[100],
            n_features=5,
            n_total_columns=7,
            numeric_feature_names=["a", "b", "c", "d", "e"],
        )
        result = validate_processing_config(config, meta)
        assert result is config

    def test_unknown_forecasting_features(self):
        config = ProcessingConfig(forecasting_features=["nonexistent"])
        meta = TimeSeriesMetadata(
            n_samples=1,
            n_timesteps_per_sample=[100],
            n_features=2,
            n_total_columns=4,
            numeric_feature_names=["a", "b"],
        )
        with pytest.raises(ValidationError, match="Unknown forecasting features"):
            validate_processing_config(config, meta)
