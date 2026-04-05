"""
Tests for ai4ts.data_protocols.utils module.
"""

import pytest
import numpy as np
import pandas as pd

from ai4ts.data_protocols.utils import (
    separate_features,
    get_feature_type,
    compute_sliding_stride,
    compute_n_samples,
    forward_fill,
    create_missing_mask,
    fill_and_get_mask,
    insert_anchor_points,
    pad_sequence,
    create_padding_mask,
    inject_artificial_missing,
    ensure_sample_id,
    ensure_timestamp,
)
from ai4ts.data_protocols.enums import FeatureType
from ai4ts.data_protocols.constants import COL_NAME_SAMPLE_ID, COL_NAME_TIMESTAMP


class TestSeparateFeatures:
    """Tests for the separate_features utility."""

    def test_numeric_only(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        numeric, string, integer, claf = separate_features(df)
        assert numeric == ["a", "b"]
        assert string == []
        assert claf == []

    def test_mixed_types(self):
        df = pd.DataFrame({
            "num": [1.0, 2.0],
            "str_col": ["x", "y"],
            "int_col": pd.array([1, 2], dtype="int64"),
        })
        numeric, string, integer, claf = separate_features(df)
        assert "num" in numeric
        assert "int_col" in numeric  # integer is also numeric
        assert "str_col" in string
        assert "int_col" in integer

    def test_reserved_columns_excluded(self):
        df = pd.DataFrame({
            "SAMPLE_ID": [0, 0],
            "TIMESTAMP": ["2024-01-01", "2024-01-02"],
            "feat": [1.0, 2.0],
        })
        numeric, string, integer, claf = separate_features(df)
        assert "SAMPLE_ID" not in numeric
        assert "TIMESTAMP" not in string
        assert "feat" in numeric

    def test_classification_target_detected(self):
        df = pd.DataFrame({
            "feat": [1.0, 2.0],
            "CLAF_TARGET": [0, 1],
        })
        numeric, string, integer, claf = separate_features(df)
        assert "CLAF_TARGET" in claf
        assert "CLAF_TARGET" not in numeric

    def test_multiple_claf_targets_uses_first(self):
        df = pd.DataFrame({
            "feat": [1.0, 2.0],
            "CLAF_TARGET_A": [0, 1],
            "CLAF_TARGET_B": [1, 0],
        })
        _, _, _, claf = separate_features(df)
        assert len(claf) == 1
        assert claf[0] == "CLAF_TARGET_A"


class TestGetFeatureType:
    """Tests for get_feature_type."""

    def test_numeric(self):
        assert get_feature_type(pd.Series([1.0, 2.5])) == FeatureType.NUMERIC

    def test_integer(self):
        assert get_feature_type(pd.Series([1, 2, 3], dtype="int64")) == FeatureType.INTEGER

    def test_string(self):
        assert get_feature_type(pd.Series(["a", "b"])) == FeatureType.STRING


class TestComputeSlidingStride:
    """Tests for compute_sliding_stride."""

    def test_small_data(self):
        assert compute_sliding_stride(100, 50, min_samples=300) == 1

    def test_large_data(self):
        stride = compute_sliding_stride(10000, 128, min_samples=300)
        assert stride >= 1
        assert stride <= 128

    def test_none_min_samples(self):
        assert compute_sliding_stride(10000, 128, min_samples=None) == 1

    def test_stride_at_least_one(self):
        assert compute_sliding_stride(1, 128, min_samples=300) >= 1


class TestComputeNSamples:
    """Tests for compute_n_samples."""

    def test_exact_fit(self):
        assert compute_n_samples(128, 128, 1) == 1

    def test_multiple_windows(self):
        assert compute_n_samples(256, 128, 1) == 129

    def test_with_stride(self):
        assert compute_n_samples(256, 128, 2) == 65

    def test_short_data(self):
        assert compute_n_samples(50, 128, 1) == 1  # at least 1


class TestForwardFill:
    """Tests for forward_fill."""

    def test_no_nan(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = forward_fill(data)
        np.testing.assert_array_equal(result, data)

    def test_middle_nan(self):
        data = np.array([[1.0], [np.nan], [3.0]])
        result = forward_fill(data)
        expected = np.array([[1.0], [1.0], [3.0]])
        np.testing.assert_array_equal(result, expected)

    def test_leading_nan_filled_with_zero(self):
        data = np.array([[np.nan], [2.0], [3.0]])
        result = forward_fill(data)
        assert result[0, 0] == 0.0
        assert result[1, 0] == 2.0

    def test_consecutive_nan(self):
        data = np.array([[1.0], [np.nan], [np.nan], [4.0]])
        result = forward_fill(data)
        expected = np.array([[1.0], [1.0], [1.0], [4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_all_nan(self):
        data = np.array([[np.nan], [np.nan]])
        result = forward_fill(data)
        # When all values are NaN, there is no valid value to propagate,
        # leading NaN fill sets to 0 but only up to first_valid which is 0
        # for a column of all NaN, argmax returns 0 and mask[0] is True,
        # so no leading fill occurs. The column stays NaN.
        assert np.isnan(result).all()


class TestCreateMissingMask:
    """Tests for create_missing_mask."""

    def test_no_missing(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = create_missing_mask(data)
        np.testing.assert_array_equal(mask, np.ones_like(data, dtype=np.float32))

    def test_with_missing(self):
        data = np.array([[1.0, np.nan], [np.nan, 4.0]])
        mask = create_missing_mask(data)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_equal(mask, expected)

    def test_all_missing(self):
        data = np.full((2, 3), np.nan)
        mask = create_missing_mask(data)
        np.testing.assert_array_equal(mask, np.zeros((2, 3), dtype=np.float32))


class TestFillAndGetMask:
    """Tests for fill_and_get_mask."""

    def test_returns_tuple(self):
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        filled, mask = fill_and_get_mask(data)
        assert filled.shape == data.shape
        assert mask.shape == data.shape

    def test_filled_has_no_nan(self):
        data = np.array([[1.0, np.nan], [np.nan, 4.0]])
        filled, mask = fill_and_get_mask(data)
        assert not np.isnan(filled).any()

    def test_mask_tracks_originals(self):
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        _, mask = fill_and_get_mask(data)
        assert mask[0, 0] == 1.0  # observed
        assert mask[0, 1] == 0.0  # was NaN
        assert mask[1, 0] == 1.0
        assert mask[1, 1] == 1.0


class TestInsertAnchorPoints:
    """Tests for insert_anchor_points."""

    def test_basic(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = insert_anchor_points(data, n_anchors=2)
        assert result.shape == (4, 2)
        np.testing.assert_array_equal(result[:2], np.ones((2, 2)))
        np.testing.assert_array_equal(result[2:], data)

    def test_zero_anchors(self):
        data = np.array([[1.0], [2.0]])
        result = insert_anchor_points(data, n_anchors=0)
        np.testing.assert_array_equal(result, data)

    def test_preserves_dtype(self):
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        result = insert_anchor_points(data, n_anchors=1)
        assert result.dtype == np.float32


class TestPadSequence:
    """Tests for pad_sequence."""

    def test_padding_needed(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        padded, valid = pad_sequence(data, max_len=5)
        assert padded.shape == (5, 2)
        assert valid[:2].all()
        assert not valid[2:].any()

    def test_no_padding_needed(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        padded, valid = pad_sequence(data, max_len=3)
        np.testing.assert_array_equal(padded, data)
        assert valid.all()

    def test_truncation(self):
        data = np.array([[1.0], [2.0], [3.0], [4.0]])
        padded, valid = pad_sequence(data, max_len=2)
        assert padded.shape == (2, 1)
        np.testing.assert_array_equal(padded, data[:2])

    def test_custom_pad_value(self):
        data = np.array([[1.0]])
        padded, _ = pad_sequence(data, max_len=3, pad_value=0.0)
        assert padded[1, 0] == 0.0
        assert padded[2, 0] == 0.0


class TestCreatePaddingMask:
    """Tests for create_padding_mask."""

    def test_full_valid(self):
        mask = create_padding_mask(5, 5)
        assert mask.shape == (5, 5)
        assert mask.all()

    def test_partial_valid(self):
        mask = create_padding_mask(3, 5)
        assert mask[:3, :3].all()  # valid region
        assert not mask[3:, :].any()  # padded rows
        assert not mask[:, 3:].any()  # padded cols


class TestInjectArtificialMissing:
    """Tests for inject_artificial_missing."""

    def test_produces_nan(self):
        data = np.ones((100, 5))
        rng = np.random.default_rng(42)
        result = inject_artificial_missing(data, rate=0.3, rng=rng)
        assert np.isnan(result).any()

    def test_does_not_modify_original(self):
        data = np.ones((10, 3))
        original = data.copy()
        inject_artificial_missing(data, rate=0.5)
        np.testing.assert_array_equal(data, original)

    def test_rate_zero(self):
        data = np.ones((10, 3))
        result = inject_artificial_missing(data, rate=0.0)
        assert not np.isnan(result).any()

    def test_reproducibility(self):
        data = np.ones((50, 5))
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        r1 = inject_artificial_missing(data, rate=0.2, rng=rng1)
        r2 = inject_artificial_missing(data, rate=0.2, rng=rng2)
        np.testing.assert_array_equal(np.isnan(r1), np.isnan(r2))


class TestEnsureSampleId:
    """Tests for ensure_sample_id."""

    def test_adds_column_when_missing(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = ensure_sample_id(df)
        assert COL_NAME_SAMPLE_ID in result.columns
        assert (result[COL_NAME_SAMPLE_ID] == 0).all()

    def test_preserves_existing(self):
        df = pd.DataFrame({COL_NAME_SAMPLE_ID: [1, 2], "a": [3, 4]})
        result = ensure_sample_id(df)
        assert list(result[COL_NAME_SAMPLE_ID]) == [1, 2]

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = ensure_sample_id(df)
        assert COL_NAME_SAMPLE_ID not in df.columns


class TestEnsureTimestamp:
    """Tests for ensure_timestamp."""

    def test_adds_column_when_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = ensure_timestamp(df)
        assert COL_NAME_TIMESTAMP in result.columns
        assert len(result[COL_NAME_TIMESTAMP]) == 3

    def test_preserves_existing(self):
        df = pd.DataFrame({
            COL_NAME_TIMESTAMP: ["2024-01-01", "2024-01-02"],
            "a": [1, 2],
        })
        result = ensure_timestamp(df)
        assert list(result[COL_NAME_TIMESTAMP]) == ["2024-01-01", "2024-01-02"]

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = ensure_timestamp(df)
        assert COL_NAME_TIMESTAMP not in df.columns
