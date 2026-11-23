"""
Configurations for the ai4ts package.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import os

# protocols (imported from data_protocols for backward compatibility)
from .data_protocols.constants import CLAF_TARGET_SYMBOL, COL_NAME_SAMPLE_ID

# beginnings of text and json content in the server response
TEXT_RESP_BEG = "TEXT:"
JSON_RESP_BEG = "JSON:"
RESP_BEG_LEN = len(TEXT_RESP_BEG)

DEFAULT_BASE_URL = "https://dash.time-series.ai"
BASE_URL = os.getenv("TIMESERIESAI_BASE_URL") or DEFAULT_BASE_URL
API_PATH = "/timeseriesai/api"

# API endpoints for different functionalities
INIT_ENDPOINT = f"{BASE_URL}{API_PATH}/init_chat"
LEARNING_ENDPOINT = f"{BASE_URL}{API_PATH}/learn"
IMPUTATION_ENDPOINT = f"{BASE_URL}{API_PATH}/impute"
FORECASTING_ENDPOINT = f"{BASE_URL}{API_PATH}/forecast"
CLASSIFICATION_ENDPOINT = f"{BASE_URL}{API_PATH}/classify"
CLUSTERING_ENDPOINT = f"{BASE_URL}{API_PATH}/cluster"
ANOMALY_DETECTION_ENDPOINT = f"{BASE_URL}{API_PATH}/detect"
GENERATION_ENDPOINT = f"{BASE_URL}{API_PATH}/generate"
CLEAN_ENDPOINT = f"{BASE_URL}{API_PATH}/clean"
PERSIST_ENDPOINT = f"{BASE_URL}{API_PATH}/persist"
RESTORE_ENDPOINT = f"{BASE_URL}{API_PATH}/restore"
