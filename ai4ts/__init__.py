"""

"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import sys

from .version import __version__

__all__ = [
    "__version__",
]

if sys.stdout.encoding == "utf-8":
    # for Unix-like systems (macOS and Linux) that can decode chars like ═ and ╝
    logo = f"""\u001b[34m
████████╗██╗███╗   ███╗███████╗    ███████╗███████╗██████╗ ██╗███████╗███████╗    █████╗ ██╗
╚══██╔══╝██║████╗ ████║██╔════╝    ██╔════╝██╔════╝██╔══██╗██║██╔════╝██╔════╝   ██╔══██╗██║
   ██║   ██║██╔████╔██║█████╗█████╗███████╗█████╗  ██████╔╝██║█████╗  ███████╗   ███████║██║
   ██║   ██║██║╚██╔╝██║██╔══╝╚════╝╚════██║██╔══╝  ██╔══██╗██║██╔══╝  ╚════██║   ██╔══██║██║
   ██║   ██║██║ ╚═╝ ██║███████╗    ███████║███████╗██║  ██║██║███████╗███████║██╗██║  ██║██║
   ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝╚═╝
v{0.1} - building AI for unified time-series analysis, https://time-series.ai
\u001b[0m
"""
else:
    # for Windows platform
    logo = f"""\u001b[34m
████████ ██ ███    ███ ███████       ███████ ███████ ██████  ██ ███████ ███████     █████  ██
   ██    ██ ████  ████ ██            ██      ██      ██   ██ ██ ██      ██         ██   ██ ██
   ██    ██ ██ ████ ██ █████   █████ ███████ █████   ██████  ██ █████   ███████    ███████ ██
   ██    ██ ██  ██  ██ ██                 ██ ██      ██   ██ ██ ██           ██    ██   ██ ██
   ██    ██ ██      ██ ███████       ███████ ███████ ██   ██ ██ ███████ ███████ ██ ██   ██ ██   
v{0.1} - building AI for unified time-series analysis, https://time-series.ai
\u001b[0m
"""

print(logo.encode("utf8").decode(sys.stdout.encoding, errors="ignore"))
