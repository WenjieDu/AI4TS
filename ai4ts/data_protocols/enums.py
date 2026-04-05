"""
Enumerations for the TimeSeries AI data protocol.
"""

from enum import Enum


class TaskType(str, Enum):
    """Supported time series analysis tasks."""
    IMPUTATION = "imputation"
    FORECASTING = "forecasting"
    CLASSIFICATION = "classification"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"


class FeatureType(str, Enum):
    """Data types for time series features."""
    NUMERIC = "numeric"
    STRING = "string"
    INTEGER = "integer"


class ColumnRole(str, Enum):
    """Roles a column can play in the dataset."""
    FEATURE = "feature"
    SAMPLE_ID = "sample_id"
    TIMESTAMP = "timestamp"
    CLASSIFICATION_TARGET = "classification_target"
    SAMPLE_LABEL = "sample_label"
    STEP_LABEL = "step_label"


class ScalerType(str, Enum):
    """Available scaling/normalization methods."""
    STANDARD = "standard"    # z-score: (x - mean) / std
    MINMAX = "minmax"        # min-max: (x - min) / (max - min)
    NONE = "none"            # no scaling


class MissingValueStrategy(str, Enum):
    """Strategies for handling missing values."""
    FORWARD_FILL = "forward_fill"
    ZERO_FILL = "zero_fill"
    MEAN_FILL = "mean_fill"
    NAN_KEEP = "nan_keep"  # keep NaN as-is (for downstream handling)


class InputFormat(str, Enum):
    """Supported input data formats."""
    CSV = "csv"
    NUMPY = "numpy"
    DATAFRAME = "dataframe"
