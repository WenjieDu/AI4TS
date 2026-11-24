"""
The client module for interacting with the Time Series AI API.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import Optional

import numpy as np
import requests

from .config import API_PATH, DEFAULT_BASE_URL
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
        Defaults to the TIMESERIESAI_API_KEY environment variable.
    model:
        Name of the remote model.
    base_url:
        API server URL. Defaults to TIMESERIESAI_BASE_URL or the public service.

    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gungnir_v1",
        *,
        base_url: Optional[str] = None,
    ):
        self.api_key = determine_api_key(api_key)
        self.authorization = f"Bearer {self.api_key}"
        self.base_url = (base_url or os.getenv("TIMESERIESAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.http_session = requests.session()

        assert model in MODEL_LIST, f"Model {model} is not supported. Please choose from {MODEL_LIST}"
        self.model = model

        self.chat_session_id = None
        self.max_file_size_in_mb = None

    def _post(self, route, *, headers, **kwargs):
        with self.http_session.post(
            url=f"{self.base_url}{API_PATH}/{route}",
            headers={"authorization": self.authorization, **headers},
            stream=True,
            **kwargs,
        ) as response:
            response.raise_for_status()
            if not 200 <= response.status_code < 300:
                raise RuntimeError(f"Unexpected service response (HTTP {response.status_code})")
            return response_handler(response)

    def _require_session(self):
        if self.chat_session_id is None:
            raise ValueError("Chat session is not initialized. Call learn or restore first.")

    def _post_to_endpoint(self, route, data, **options):
        self._require_session()
        headers = {
            "chat_session_id": self.chat_session_id,
            **{key: str(value) for key, value in options.items() if value is not None},
        }
        if data is None:
            return self._post(route, headers=headers)

        # A resumed session may not carry a local copy of the service's upload limit.
        if self.max_file_size_in_mb is not None and not check_file_size(data, self.max_file_size_in_mb):
            raise ValueError(f"File exceeds the maximum size limit of {self.max_file_size_in_mb} MB")
        with open(data, "rb") as stream:
            return self._post(route, headers=headers, files={"file": (os.path.basename(data), stream, "text/csv")})

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
        result = self._post("init_chat", headers={"Accept": "application/json"}, json=session_config)
        session_id = result.get("chat_session_id") if isinstance(result, dict) else None
        if not isinstance(session_id, str) or not session_id.strip():
            raise RuntimeError("Service did not return a session ID")
        self.chat_session_id = session_id
        self.max_file_size_in_mb = result.get("max_file_size_in_mb")
        return self._post_to_endpoint("learn", data)

    def impute(self, data: Optional[str] = None):
        """Impute the missing values in the data based on the learned AI model.

        data:
            The incomplete time series data to be imputed.

        Returns
        -------
        np.ndarray
            The imputed data.

        """
        result = self._post_to_endpoint("impute", data)
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
        return self._post_to_endpoint("forecast", data, **kwargs)

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
        return self._post_to_endpoint("classify", data)

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
            "detect",
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
            "cluster",
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
            "clean",
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
        self._require_session()
        self._post("persist", headers={"chat_session_id": self.chat_session_id, "alias": alias or ""})

    def restore(self, session_id: str) -> None:
        """Restore the AI model context from the persisted session with the given session ID.

        Returns
        -------
        None

        """
        result = self._post("restore", headers={"chat_session_id": session_id})
        self.max_file_size_in_mb = result.get("max_file_size_in_mb") if isinstance(result, dict) else None
        if session_id != self.chat_session_id:
            self.chat_session_id = session_id
            logger.info(f"Switched session to {self.chat_session_id}")
