"""
Utility functions for the ai4ts package.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import io
import json
import sys
import threading
import time

import pandas as pd
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


def bytes2df_handler(response):
    file_obj = io.BytesIO(response.content)
    dataframe = pd.read_parquet(file_obj)
    return dataframe


def response_handler(response: requests.Response):
    """Check the response status code and print the corresponding message.

    Parameters
    ----------
    response:
        The response object from the API request.

    Returns
    -------
    dict
        The JSON response content if the response status

    """
    spinning_cursor = SpinningCursor()
    spinning_cursor.start()
    try:
        if response.status_code == 200:
            if response.headers.get("Content-Type") == "application/octet-stream":
                # process binary data
                df = bytes2df_handler(response)
                return df
            else:
                # print the response content line by line for streaming response
                buffer = ""
                for chunk in response.iter_content(chunk_size=1):
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
            sys.stdout.write("\b")
            logger.error("‼️Unauthorized access. Please check your API key.")

        elif response.status_code == 415:
            # unsupported media type
            sys.stdout.write("\b")
            logger.error(f"❌{response.json()['detail']}")

        elif str(response.status_code).startswith("5"):  # notify users server is down when status code starts with 5
            # server is down
            sys.stdout.write("\b")
            logger.error("🙇Server is not available now. Please try again later.")
            logger.debug(f"Response status code: {response.status_code}. Response body:\n{response.text}")
        elif response.status_code in [
            404,
        ]:
            # for specific errors like 404, directly print the error message from the server
            logger.error(response.text)
        else:
            # log info in the response for other status codes
            sys.stdout.write("\b")
            logger.error(f"Response status code: {response.status_code}. Response body:\n{response.text}")

    finally:
        spinning_cursor.stop()
