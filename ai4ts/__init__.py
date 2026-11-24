"""AI4TS package metadata and lazy data protocol imports."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import sys

from .version import __version__

__all__ = [
    "__version__",
]


def __getattr__(name):
    if name == "data_protocols":
        from importlib import import_module

        module = import_module(".data_protocols", __name__)
        globals()[name] = module
        return module
    raise AttributeError(name)
