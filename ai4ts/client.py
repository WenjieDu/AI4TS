"""
The client module for interacting with the Time Series AI API.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import os
import time

import numpy as np
import requests

from .config import (
    INIT_ENDPOINT,
    LEARNING_ENDPOINT,
    IMPUTATION_ENDPOINT,
    FORECASTING_ENDPOINT,
    CLASSIFICATION_ENDPOINT,
    CLUSTERING_ENDPOINT,
    ANOMALY_DETECTION_ENDPOINT,
    CLEAN_ENDPOINT,
)
from .utils import (
    determine_api_key,
    response_handler,
    check_file_size,
)

# the list of supported AI models
MODEL_LIST = [
    "gungnir_v1",
]


class TimeSeriesAI:
    """The client for interacting with the Time Series AI API.

    Parameters
    ----------
    api_key:
        The API key to access the Time Series AI API.
        If not provided, the function will try to load the API key from the environment variable or the local config

    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gungnir_v1",
    ):
        self.api_key = determine_api_key(api_key)
        self.authorization = f"Bearer {api_key}"
        self.http_session = requests.session()

        assert model in MODEL_LIST, f"Model {model} is not supported. Please choose from {MODEL_LIST}"
        self.model = model

        # initialize the chat session
        session_config = {
            "chat": {
                "models": [self.model],
                "timestamp": time.time(),
            }
        }
        with self.http_session.post(
            url=INIT_ENDPOINT,
            headers={
                "authorization": self.authorization,
                "Accept": "application/json",
            },
            json=session_config,
            stream=True,
        ) as response:
            result = response_handler(response)

        self.chat_session_id = result["chat_session_id"] if response.status_code == 200 else None
        self.max_file_size_in_mb = result["max_file_size_in_mb"] if response.status_code == 200 else None

    def _post_data(
        self,
        endpoint: str,
        data: str,
    ):
        result = None

        if check_file_size(data, self.max_file_size_in_mb):  # check the file size
            # post data to the server
            with self.http_session.post(
                url=endpoint,
                headers={
                    "authorization": self.authorization,
                    "chat_session_id": self.chat_session_id,
                },
                files={
                    "file": (os.path.basename(data), open(data, "rb"), "text/csv"),
                },
                stream=True,
            ) as response:
                result = response_handler(response)

        return result

    def learn(self, data: str) -> None:
        """Feed the data into AI model and let it learn from the context.
        This operation can be repeated multiple times to improve the performance,
        and such online learning process assumes the fed data is IID (independent and identically distributed).

        Returns
        -------
        None

        """
        return self._post_data(LEARNING_ENDPOINT, data)

    def impute(self, data):
        """Impute the missing values in the data based on the learned AI model.

        data:
            The incomplete time series data to be imputed.

        Returns
        -------
        np.ndarray
            The imputed data.

        """
        return self._post_data(IMPUTATION_ENDPOINT, data)

    def forecast(self, data):
        """Forecast the future values based on the learned AI model.

        Parameters
        ----------
        data:
            The historic time series data to be used for forecasting.

        Returns
        -------
        np.ndarray
            The forecasting result.

        """
        return self._post_data(FORECASTING_ENDPOINT, data)

    def classify(self, data):
        """Classify the data based on the learned AI model.

        Parameters
        ----------
        data:
            The time series samples to be classified.

        Returns
        -------
        np.ndarray
            The classification result.

        """
        return self._post_data(CLASSIFICATION_ENDPOINT, data)

    def detect(self, data):
        """Detect the anomalies in the data based on the learned AI model

        Returns
        -------
        np.ndarray
            The anomaly detection result.

        """
        return self._post_data(ANOMALY_DETECTION_ENDPOINT, data)

    def cluster(self, data):
        """Cluster the data based on the learned AI model.

        Parameters
        ----------
        data:
            The time series data to be clustered.

        Returns
        -------
        np.ndarray
            The clustering result.

        """
        return self._post_data(CLUSTERING_ENDPOINT, data)

    def clean(self, data):
        """Clean the given data based on the learned AI model.
        Remove the noise and outliers from the data, and reconstruct it.

        Returns
        -------
        np.ndarray
            The cleaned data.

        """
        return self._post_data(CLEAN_ENDPOINT, data)

    def generate(self) -> np.ndarray:
        """Generate synthetic data based on the learned AI model.
        The generated data will bear similar statistical properties as the original data fed into the AI model previously.

        Returns
        -------
        np.ndarray
            The generated synthetic data.

        """
        pass

    def persist(self) -> str:
        """Persist the session of the AI model learned context.

        Returns
        -------
        session_id:
            The session ID to be used for restoring the AI model context.
        """
        pass

    def restore(self) -> None:
        """Restore the AI model context from the persisted session.

        Returns
        -------
        None

        """

        pass
