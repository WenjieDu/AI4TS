"""
Utility functions for the ai4ts package.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import logging
import os

logger: logging.Logger = logging.getLogger(__name__)


def determine_api_key(api_key: str = None) -> str:
    """Determine the API key from the environment variable or the local config if not given.

    Parameters
    ----------
    api_key:
        The passed-in API key to access TimeSeries AI. If not provided,
        the function will try to load the API key from the environment variable or the local config.

    Returns
    -------
    api_key: str
        The API key to access TimeSeries AI

    """
    # load the API key from the environment variable if not provided
    if api_key is None:
        api_key = os.getenv("TIMESERIESAI_API_KEY", None)
        if api_key is not None:
            logger.info("API key loaded from the environment variable")

    # load the API key from the local config if still None
    if api_key is None:
        # api_key = load_api_key_from_local_config()
        if api_key is not None:
            logger.info("API key loaded from the local config")

    # raise an error if the API key is not provided by all means
    if api_key is None:
        raise ValueError("API key is required to access TimeSeriesAI APIs")

    return api_key
