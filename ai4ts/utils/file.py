"""

"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import os

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

    if file_size > max_bytes:
        logger.error(f"❌ File exceeds {max_size_in_mb}MB limit")
        return False
    return True
