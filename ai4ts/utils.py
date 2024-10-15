"""
Utility functions for the ai4ts package.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0


import logging
import os

import requests

LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


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
        else:
            # load the API key from the local config if still None
            # api_key = load_api_key_from_local_config()
            if api_key is not None:
                logger.info("API key loaded from the local config")
    # raise an error if the API key is not provided by all means
    if api_key is None:
        raise ValueError(
            "‼️API key is required to access TimeSeriesAI APIs, "
            "please pass it by the argument `api_key` or the environment variable `TIMESERIESAI_API_KEY`"
        )
    return api_key


def check_response_code(response: requests.Response, success_print: str) -> None:
    """Check the response status code and print the corresponding message.

    Parameters
    ----------
    response:
        The response object from the API request.

    success_print:
        The success message to be printed if the response status code is 200.

    Returns
    -------
    None

    """

    if response.status_code == 200:
        logger.info(success_print)
    elif response.status_code == 401:
        logger.error("‼️Unauthorized access. Please check your API key.")
    elif response.status_code == 415:
        logger.error(f"❌{response.json()['detail']}")
    elif response.status_code == 521:
        logger.error("🙇Server is not available. Please try again later.")
    else:
        logger.error(f"Response status code: {response.status_code}. Response body: {response.text}")


class Logger:
    def __init__(
        self,
        name: str = "AI4TS running log",
        logging_level: str = "debug",
        logging_format: str = "%(asctime)s [%(levelname)s]: %(message)s",
    ):
        """
        Parameters
        ----------
        name :
            The name for the logger to be initialized.

        logging_level :
            The logging level of the logger, should be debug/info/warning/error.

        logging_format :
            Logging format of the logger.

        """

        assert logging_level in LEVELS.keys(), f"logging_level should be {list(LEVELS.keys())}, but got {logging_level}"

        self.logger = logging.getLogger(name)
        self.logging_level = LEVELS[logging_level]

        self.stream_handler = logging.StreamHandler()
        self.formatter = None
        self.file_handler = None

        self.set_level(logging_level)
        self.set_logging_format(logging_format)
        self.logger.propagate = False

    def set_logging_format(self, logging_format: str) -> None:
        self.formatter = logging.Formatter(logging_format, datefmt="%Y-%m-%d %H:%M:%S")
        self.stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)
        if self.file_handler is not None:
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)

    def set_saving_path(self, saving_dir: str, name: str, mode: str = "a") -> None:
        """Set the logger's saving path. This function will enable saving logs to the specified path.

        Parameters
        ----------
        saving_dir :
            The path to the directory for logging file saving.

        name :
            The name of the logging file to be saved.

        mode :
            Logging file writing mode.

        """
        if not os.path.exists(saving_dir):
            self.logger.warning(f"{saving_dir} does not exist. Creating it now...")
            os.makedirs(saving_dir)
        path = os.path.join(saving_dir, name)
        self.file_handler = logging.FileHandler(path, mode=mode)
        self.file_handler.setLevel(self.logging_level)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.info(f"Log will be saved to {path}")

    def set_level(self, level: str) -> None:
        """Set the logger's logging level.

        Parameters
        ----------
        level :
            The logging level of the logger, should be debug/info/warning/error.

        """
        self.logging_level = LEVELS[level]
        self.logger.setLevel(self.logging_level)
        if self.stream_handler is not None:
            self.stream_handler.setLevel(self.logging_level)
        if self.file_handler is not None:
            self.file_handler.setLevel(self.logging_level)


# initialize a logger for logging
logger_creator = Logger()
logger = logger_creator.logger
