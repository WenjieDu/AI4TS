"""
Standard constants for the TimeSeries AI data protocol.

These constants define the reserved column names, symbols, and default values
used across the entire TimeSeries AI platform (SDK, backend, model library).
"""

# Reserved column names
COL_NAME_SAMPLE_ID = "SAMPLE_ID"
COL_NAME_SAMPLE_LABEL = "SAMPLE_LABEL"
COL_NAME_STEP_LABEL = "STEP_LABEL"
COL_NAME_TIMESTAMP = "TIMESTAMP"

# All reserved column names that are not features
RESERVED_COLUMNS = {
    COL_NAME_SAMPLE_ID,
    COL_NAME_SAMPLE_LABEL,
    COL_NAME_STEP_LABEL,
    COL_NAME_TIMESTAMP,
}

# Target column identification symbols
# Columns whose names contain this symbol are treated as classification targets
CLAF_TARGET_SYMBOL = "CLAF_TARGET"

# Default processing parameters
DEFAULT_TS_CONTEXT_LEN = 128
DEFAULT_FORECASTING_HORIZON = 10
DEFAULT_N_ANCHOR_POINTS = 2
DEFAULT_ARTIFICIAL_MISSING_RATE = 0.1
DEFAULT_CAST_DTYPE = "float32"
AT_LEAST_SAMPLE_NUM = 300

# Computed defaults
DEFAULT_MAX_INPUT_STEPS = DEFAULT_TS_CONTEXT_LEN + DEFAULT_N_ANCHOR_POINTS  # 130

# Supported file formats for upload
SUPPORTED_FILE_TYPES = {"text/csv", "text/json", "application/json"}
SUPPORTED_FILE_EXTENSIONS = {"csv", "json"}

# API response protocol prefixes
TEXT_RESPONSE_PREFIX = "TEXT:"
JSON_RESPONSE_PREFIX = "JSON:"
RESPONSE_LINE_END = "\n"

# Default timestamp settings (for auto-generation)
DEFAULT_TIMESTAMP_START = "2024-01-01"
DEFAULT_TIMESTAMP_FREQ = "h"  # hourly
