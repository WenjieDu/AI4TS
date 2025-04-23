"""
The client module for interacting with the Time Series AI API.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import os
import time
from typing import Optional

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
    PERSIST_ENDPOINT,
    RESTORE_ENDPOINT,
)
from .utils import (
    determine_api_key,
    response_handler,
    check_file_size,
)
from .utils.logging import logger

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

        self.chat_session_id = None
        self.max_file_size_in_mb = None

    def _post_to_endpoint(
        self,
        endpoint: str,
        data: str,
        **kwargs: Optional[dict],
    ):
        result = None
        if self.chat_session_id is None:
            logger.error("❌ Chat session is not initialized. Call `learn` to feed your data into the AI model first.")
            return None

        if data is not None:
            if not os.path.exists(data):
                logger.error(f"❌ File {data} does not exist")
                return None
            if not check_file_size(data, self.max_file_size_in_mb):  # check the file size
                logger.error(f"❌ File {data} exceeds the maximum size limit of {self.max_file_size_in_mb}MB")
                return None

            # post data to the server
            with self.http_session.post(
                url=endpoint,
                headers={
                    "authorization": self.authorization,
                    "chat_session_id": self.chat_session_id,
                    **kwargs,
                },
                files={
                    "file": (os.path.basename(data), open(data, "rb"), "text/csv"),
                },
                stream=True,
            ) as response:
                result = response_handler(response)
        else:
            with self.http_session.post(
                url=endpoint,
                headers={
                    "authorization": self.authorization,
                    "chat_session_id": self.chat_session_id,
                    **kwargs,
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
        # initialize the chat session first
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

        return self._post_to_endpoint(LEARNING_ENDPOINT, data)

    def impute(self, data: Optional[str] = None):
        """Impute the missing values in the data based on the learned AI model.

        data:
            The incomplete time series data to be imputed.

        Returns
        -------
        np.ndarray
            The imputed data.

        """
        result = self._post_to_endpoint(IMPUTATION_ENDPOINT, data)
        return result

    def forecast(
        self,
        data: Optional[str] = None,
        n_forecast_steps: Optional[int] = None,
    ):
        """Forecast the future values based on the learned AI model.

        Parameters
        ----------
        data:
            The historic time series data to be used for forecasting.

        n_forecast_steps:
            The number of future steps to forecast for each sample.

        Returns
        -------
        np.ndarray
            The forecasting result.

        """
        kwargs = {
            "n_forecast_steps": n_forecast_steps,
        }
        return self._post_to_endpoint(FORECASTING_ENDPOINT, data, **kwargs)

    def classify(self, data: Optional[str] = None):
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
        return self._post_to_endpoint(CLASSIFICATION_ENDPOINT, data)

    def detect(
        self,
        data: Optional[str] = None,
        anomaly_rate: float = 0.01,
    ):
        """Detect the anomalies in the data based on the learned AI model

        Parameters
        ----------
        data:
            The time series data to be clustered.

        anomaly_rate:
            The expected anomaly rate in the data.
            This is used to adjust the sensitivity of the anomaly detection algorithm.
            A higher value means more anomalies will be detected,
            while a lower value means fewer anomalies will be detected.

        Returns
        -------
        np.ndarray
            The anomaly detection result.

        """
        kwargs = {
            "anomaly_rate": anomaly_rate,
        }
        return self._post_to_endpoint(
            ANOMALY_DETECTION_ENDPOINT,
            data,
            **kwargs,
        )

    def cluster(
        self,
        data: Optional[str] = None,
        n_clusters: Optional[int] = None,
    ):
        """Cluster the data based on the learned AI model.

        Parameters
        ----------
        data:
            The time series data to be clustered.

        n_clusters:
            Number of clusters to form.

        Returns
        -------
        np.ndarray
            The clustering result.

        """
        kwargs = {
            "n_clusters": n_clusters,
        }
        return self._post_to_endpoint(
            CLUSTERING_ENDPOINT,
            data,
            **kwargs,
        )

    def clean(
        self,
        data: Optional[str] = None,
        anomaly_rate: float = 0,
    ):
        """Clean the given data based on the learned AI model.
        Remove the noise and outliers from the data and reconstruct it.

        Parameters
        ----------
        data:
            The time series data to be clustered.

        anomaly_rate:
            The expected anomaly rate in the data.
            This is used to adjust the sensitivity of the anomaly detection algorithm.
            A higher value means more anomalies will be detected,
            while a lower value means fewer anomalies will be detected.

        Returns
        -------
        np.ndarray
            The cleaned data.

        """
        kwargs = {
            "anomaly_rate": anomaly_rate,
        }
        return self._post_to_endpoint(
            CLEAN_ENDPOINT,
            data,
            **kwargs,
        )

    def generate(self) -> np.ndarray:
        """Generate synthetic data based on the learned AI model.
        The generated data will bear similar statistical properties as the original data fed into the AI model previously.

        Returns
        -------
        np.ndarray
            The generated synthetic data.

        """
        pass

    def persist(
        self,
        alias: Optional[str] = None,
    ) -> None:
        """Persist the current session of the AI model learned context.

        Returns
        -------
        session_id:
            The session ID to be used for restoring the AI model context.
        """
        # post data to the server
        with self.http_session.post(
            url=PERSIST_ENDPOINT,
            headers={
                "authorization": self.authorization,
                "chat_session_id": self.chat_session_id,
                "alias": alias or "",
            },
            stream=True,
        ) as response:
            response_handler(response)

    def restore(self, session_id: str) -> None:
        """Restore the AI model context from the persisted session with the given session ID.

        Returns
        -------
        None

        """
        with self.http_session.post(
            url=RESTORE_ENDPOINT,
            headers={
                "authorization": self.authorization,
                "chat_session_id": session_id,
            },
            stream=True,
        ) as response:
            response_handler(response)

        if session_id != self.chat_session_id:
            self.chat_session_id = session_id
            logger.info(f"Switched session to {self.chat_session_id}")
