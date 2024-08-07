"""
Configurations for the ai4ts package.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import os

BASE_URL = os.getenv(
    "TIMESERIESAI_BASE_URL",
    "https://api.time-series.ai",
)

LEARNING_SESSION_INIT_ENDPOINT = f"{BASE_URL}/api/v1/chats/new"
LEARNING_ENDPOINT = f"{BASE_URL}/timeseriesai/api/learn"
IMPUTATION_ENDPOINT = f"{BASE_URL}/timeseriesai/api/impute"
FORECASTING_ENDPOINT = f"{BASE_URL}/timeseriesai/api/forecast"
CLASSIFICATION_ENDPOINT = f"{BASE_URL}/timeseriesai/api/classify"
CLUSTERING_ENDPOINT = f"{BASE_URL}/timeseriesai/api/cluster"
ANOMALY_DETECTION_ENDPOINT = f"{BASE_URL}/timeseriesai/api/detect"
GENERATION_ENDPOINT = f"{BASE_URL}/timeseriesai/api/generate"
CLEAN_ENDPOINT = f"{BASE_URL}/timeseriesai/api/clean"
