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

# stdout_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
# if stdout_encoding.lower() == "utf-8":
#     # for Unix-like systems (macOS and Linux) that can decode chars like ═ and ╝
#     logo = f"""\u001b[34m
# ████████╗██╗███╗   ███╗███████╗    ███████╗███████╗██████╗ ██╗███████╗███████╗    █████╗ ██╗
# ╚══██╔══╝██║████╗ ████║██╔════╝    ██╔════╝██╔════╝██╔══██╗██║██╔════╝██╔════╝   ██╔══██╗██║
#    ██║   ██║██╔████╔██║█████╗█████╗███████╗█████╗  ██████╔╝██║█████╗  ███████╗   ███████║██║
#    ██║   ██║██║╚██╔╝██║██╔══╝╚════╝╚════██║██╔══╝  ██╔══██╗██║██╔══╝  ╚════██║   ██╔══██║██║
#    ██║   ██║██║ ╚═╝ ██║███████╗    ███████║███████╗██║  ██║██║███████╗███████║██╗██║  ██║██║
#    ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝╚═╝
# ai4ts v{__version__} - building AI for unified time-series analysis, https://time-series.ai \u001b[0m
# """
# else:
#     # for Windows platform
#     logo = f"""\u001b[34m
# ████████ ██ ███    ███ ███████       ███████ ███████ ██████  ██ ███████ ███████     █████  ██
#    ██    ██ ████  ████ ██            ██      ██      ██   ██ ██ ██      ██         ██   ██ ██
#    ██    ██ ██ ████ ██ █████   █████ ███████ █████   ██████  ██ █████   ███████    ███████ ██
#    ██    ██ ██  ██  ██ ██                 ██ ██      ██   ██ ██ ██           ██    ██   ██ ██
#    ██    ██ ██      ██ ███████       ███████ ███████ ██   ██ ██ ███████ ███████ ██ ██   ██ ██
# ai4ts v{__version__} - building AI for unified time-series analysis, https://time-series.ai \u001b[0m
# """
#     logo = logo.encode("utf-8").decode(stdout_encoding, errors="ignore")
#
# print(logo)
