"""
Utility functions for the ai4ts package.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import json
import sys
import threading
import time
from typing import Optional

import requests

from .logging import logger
from ..config import JSON_RESP_BEG, RESP_BEG_LEN, TEXT_RESP_BEG


class SpinningCursor(threading.Thread):
    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()

    def run(self):
        spinner = ["⣾", "⣷", "⣯", "⣟", "⡿", "⢿", "⣻", "⣽"]
        idx = 0
        while not self._stop_event.is_set():
            sys.stdout.write(spinner[idx % len(spinner)])  # write the current character
            sys.stdout.flush()  # ensure immediate display
            time.sleep(0.1)
            sys.stdout.write("\b")  # backspace the cursor
            idx += 1

    def stop(self):
        self._stop_event.set()
        self.join()


def check_response_code(response: requests.Response) -> Optional[json]:
    """Check the response status code and print the corresponding message.

    Parameters
    ----------
    response:
        The response object from the API request.

    Returns
    -------
    None

    """
    spinning_cursor = SpinningCursor()
    spinning_cursor.start()
    try:
        if response.status_code == 200:
            # print the response content line by line for streaming response
            buffer = ""
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    buffer += chunk.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line:
                            continue

                        if line.startswith(TEXT_RESP_BEG):
                            sys.stdout.write("\b")
                            logger.info(line[RESP_BEG_LEN:])
                        elif line.startswith(JSON_RESP_BEG):
                            return json.loads(line[RESP_BEG_LEN:])

        elif response.status_code == 401:
            # unauthorized access
            logger.error("‼️Unauthorized access. Please check your API key.")

        elif response.status_code == 415:
            # unsupported media type
            logger.error(f"❌{response.json()['detail']}")

        elif response.status_code == 521:
            # server is down
            logger.error("🙇Server is not available now. Please try again later.")

        else:
            logger.error(f"Response status code: {response.status_code}. Response body: {response.text}")
    finally:
        spinning_cursor.stop()
