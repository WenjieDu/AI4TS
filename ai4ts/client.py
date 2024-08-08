"""
The client module for interacting with the Time Series AI API.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import logging
import time

import numpy as np
import requests

from .config import (
    LEARNING_SESSION_INIT_ENDPOINT,
    LEARNING_ENDPOINT,
)
from .utils import determine_api_key

logger: logging.Logger = logging.getLogger(__name__)


class TimeSeriesAI:
    def __init__(
        self,
        api_key: str = None,
    ):
        self.api_key = determine_api_key(api_key)
        self.authorization = f"Bearer {api_key}"
        self.learning_session_id = None

        LEARNING_SESSION = {
            "chat": {
                "models": ["Gungnir"],
                "timestamp": time.time(),
            }
        }
        response = requests.post(
            url=LEARNING_SESSION_INIT_ENDPOINT,
            headers={
                "authorization": self.authorization,
                "Accept": "application/json",
            },
            json=LEARNING_SESSION,
        )

        if response.status_code == 200:
            self.learning_session_id = response.json()["id"]
        else:
            raise Exception(response.text)

    def learn(self, data: str) -> None:
        """Feed the data into AI model and let it learn from the context.
        This operation can be repeated multiple times to improve the performance,
        and such online learning process assumes the fed data is IID (independent and identically distributed).

        Returns
        -------
        None

        """
        requests.post(
            url=LEARNING_ENDPOINT,
            headers={
                "authorization": self.authorization,
            },
            files={"file": ("file.csv", open(data, "rb"), "text/csv")},
        )

    def impute(self, data):
        """Impute the missing values in the data based on the learned AI model.

        data:
            The incomplete time series data to be imputed.

        Returns
        -------
        np.ndarray
            The imputed data.

        """
        pass

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
        pass

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
        pass

    def detect(self, data):
        """Detect the anomalies in the data based on the learned AI model

        Returns
        -------
        np.ndarray
            The anomaly detection result.

        """
        pass

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
        pass

    def generate(self) -> np.ndarray:
        """Generate synthetic data based on the learned AI model.
        The generated data will bear similar statistical properties as the original data fed into the AI model previously.

        Returns
        -------
        np.ndarray
            The generated synthetic data.

        """
        pass

    def clean(self, data) -> np.ndarray:
        """Clean the given data based on the learned AI model.
        Remove the noise and outliers from the data, and reconstruct it.

        Returns
        -------
        np.ndarray
            The cleaned data.

        """
        pass

    def persist(self) -> str:
        """Persist the session of the AI model learned context.

        Returns
        -------
        session_id:
            The session ID to be used for restoring the AI model context.
        """
        session_id: str = None
        return session_id

    def restore(self) -> None:
        """Restore the AI model context from the persisted session.

        Returns
        -------
        None

        """

        pass
