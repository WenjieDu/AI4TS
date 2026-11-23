"""
Configuration utilities for the AI4TS CLI.

Manages reading and writing CLI configuration stored at ~/.ai4ts/config.json.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path.home() / ".ai4ts"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_BASE_URL = "https://dash.time-series.ai"

DEFAULT_CONFIG = {
    "api_key": "",
    "base_url": DEFAULT_BASE_URL,
}


def load_config() -> dict:
    """Load configuration from disk, creating defaults if needed."""
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)

    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        config = dict(DEFAULT_CONFIG)
        save_config(config)

    # ensure all default keys exist
    for key, value in DEFAULT_CONFIG.items():
        config.setdefault(key, value)

    return config


def save_config(config: dict) -> None:
    """Write configuration to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_api_key() -> Optional[str]:
    """Return the API key from config file or TIMESERIESAI_API_KEY env var.

    Priority: env var > config file.
    """
    env_key = os.getenv("TIMESERIESAI_API_KEY")
    if env_key:
        return env_key

    config = load_config()
    key = config.get("api_key", "")
    return key if key else None


def get_base_url() -> str:
    """Return the base URL from config file, env var, or default.

    Priority: env var > config file > default.
    """
    env_url = os.getenv("TIMESERIESAI_BASE_URL")
    if env_url:
        return env_url

    config = load_config()
    url = config.get("base_url", "")
    return url if url else DEFAULT_BASE_URL
