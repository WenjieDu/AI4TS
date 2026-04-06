"""

"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import os

import pandas as pd

from .logging import logger


def check_file_size(
    file_path: str,
    max_size_in_mb: int,
) -> bool:
    """Check if the file size exceeds the maximum size limit.

    Parameters
    ----------
    file_path:
        The path to the file.

    max_size_in_mb:
        The maximum size limit in MB


    Returns
    -------
    bool
        True if the file size is within the limit, False otherwise.

    """
    file_size = os.path.getsize(file_path)
    max_bytes = max_size_in_mb * 1024 * 1024  # convert MB to bytes
    # TODO: check if data file aligns with our protocols

    if file_size > max_bytes:
        logger.error(f"❌ File exceeds {max_size_in_mb}MB limit")
        return False
    return True


def check_data_protocols(
    file_path: str,
) -> bool:
    """Check if a CSV file follows the TimeSeries AI data protocol.

    Only check necessary and fatal protocols on the client end.
    Leave the warning-level checks for the server to handle.

    Parameters
    ----------
    file_path:
        The path to the CSV file.

    Returns
    -------
    bool
        True if the file follows the data protocol, False otherwise.
    """
    from ..data_protocols.validation import validate_csv_file, validate_dataframe, ValidationError

    try:
        df = validate_csv_file(file_path)
        validate_dataframe(df)
        return True
    except ValidationError as e:
        logger.warning(f"Data protocol check failed: {e}")
        return False
