"""
Configurations for the ai4ts package.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import os

# beginnings of text and json content in the server response
TEXT_RESP_BEG = "TEXT:"
JSON_RESP_BEG = "JSON:"
RESP_BEG_LEN = len(TEXT_RESP_BEG)

BASE_URL = os.getenv(
    "TIMESERIESAI_BASE_URL",
    "https://dash.time-series.ai",
)

# API endpoints for different functionalities
INIT_ENDPOINT = f"{BASE_URL}/timeseriesai/api/init_chat"
LEARNING_ENDPOINT = f"{BASE_URL}/timeseriesai/api/learn"
IMPUTATION_ENDPOINT = f"{BASE_URL}/timeseriesai/api/impute"
FORECASTING_ENDPOINT = f"{BASE_URL}/timeseriesai/api/forecast"
CLASSIFICATION_ENDPOINT = f"{BASE_URL}/timeseriesai/api/classify"
CLUSTERING_ENDPOINT = f"{BASE_URL}/timeseriesai/api/cluster"
ANOMALY_DETECTION_ENDPOINT = f"{BASE_URL}/timeseriesai/api/detect"
GENERATION_ENDPOINT = f"{BASE_URL}/timeseriesai/api/generate"
CLEAN_ENDPOINT = f"{BASE_URL}/timeseriesai/api/clean"
PERSIST_ENDPOINT = f"{BASE_URL}/timeseriesai/api/persist"
RESTORE_ENDPOINT = f"{BASE_URL}/timeseriesai/api/restore"
