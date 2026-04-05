"""
Tests for ai4ts.data_protocols.constants module.
"""

import pytest

from ai4ts.data_protocols.constants import (
    COL_NAME_SAMPLE_ID,
    COL_NAME_SAMPLE_LABEL,
    COL_NAME_STEP_LABEL,
    COL_NAME_TIMESTAMP,
    CLAF_TARGET_SYMBOL,
    RESERVED_COLUMNS,
    DEFAULT_TS_CONTEXT_LEN,
    DEFAULT_FORECASTING_HORIZON,
    DEFAULT_N_ANCHOR_POINTS,
    DEFAULT_ARTIFICIAL_MISSING_RATE,
    DEFAULT_CAST_DTYPE,
    AT_LEAST_SAMPLE_NUM,
    DEFAULT_MAX_INPUT_STEPS,
    SUPPORTED_FILE_TYPES,
    SUPPORTED_FILE_EXTENSIONS,
    TEXT_RESPONSE_PREFIX,
    JSON_RESPONSE_PREFIX,
    DEFAULT_TIMESTAMP_START,
    DEFAULT_TIMESTAMP_FREQ,
)


class TestReservedColumnNames:
    """Tests for reserved column name constants."""

    def test_sample_id_column_name(self):
        assert COL_NAME_SAMPLE_ID == "SAMPLE_ID"

    def test_timestamp_column_name(self):
        assert COL_NAME_TIMESTAMP == "TIMESTAMP"

    def test_sample_label_column_name(self):
        assert COL_NAME_SAMPLE_LABEL == "SAMPLE_LABEL"

    def test_step_label_column_name(self):
        assert COL_NAME_STEP_LABEL == "STEP_LABEL"

    def test_reserved_columns_is_set(self):
        assert isinstance(RESERVED_COLUMNS, set)

    def test_reserved_columns_contains_all_reserved(self):
        assert COL_NAME_SAMPLE_ID in RESERVED_COLUMNS
        assert COL_NAME_TIMESTAMP in RESERVED_COLUMNS
        assert COL_NAME_SAMPLE_LABEL in RESERVED_COLUMNS
        assert COL_NAME_STEP_LABEL in RESERVED_COLUMNS

    def test_classification_target_symbol(self):
        assert CLAF_TARGET_SYMBOL == "CLAF_TARGET"


class TestDefaultValues:
    """Tests for default processing parameter constants."""

    def test_default_context_len(self):
        assert DEFAULT_TS_CONTEXT_LEN == 128
        assert isinstance(DEFAULT_TS_CONTEXT_LEN, int)

    def test_default_forecasting_horizon(self):
        assert DEFAULT_FORECASTING_HORIZON == 10
        assert isinstance(DEFAULT_FORECASTING_HORIZON, int)

    def test_default_n_anchor_points(self):
        assert DEFAULT_N_ANCHOR_POINTS == 2
        assert isinstance(DEFAULT_N_ANCHOR_POINTS, int)

    def test_default_artificial_missing_rate(self):
        assert DEFAULT_ARTIFICIAL_MISSING_RATE == 0.1
        assert 0.0 <= DEFAULT_ARTIFICIAL_MISSING_RATE <= 1.0

    def test_default_cast_dtype(self):
        assert DEFAULT_CAST_DTYPE == "float32"

    def test_at_least_sample_num(self):
        assert AT_LEAST_SAMPLE_NUM == 300

    def test_max_input_steps_computed_correctly(self):
        assert DEFAULT_MAX_INPUT_STEPS == DEFAULT_TS_CONTEXT_LEN + DEFAULT_N_ANCHOR_POINTS


class TestSupportedFormats:
    """Tests for supported file format constants."""

    def test_supported_file_types(self):
        assert "text/csv" in SUPPORTED_FILE_TYPES

    def test_supported_extensions(self):
        assert "csv" in SUPPORTED_FILE_EXTENSIONS
        assert "json" in SUPPORTED_FILE_EXTENSIONS

    def test_response_prefixes(self):
        assert TEXT_RESPONSE_PREFIX == "TEXT:"
        assert JSON_RESPONSE_PREFIX == "JSON:"

    def test_timestamp_defaults(self):
        assert DEFAULT_TIMESTAMP_START == "2024-01-01"
        assert DEFAULT_TIMESTAMP_FREQ == "h"
