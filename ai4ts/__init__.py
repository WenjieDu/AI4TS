"""

"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import sys

from .version import __version__

__all__ = [
    "__version__",
]

if sys.stdout.encoding.lower() == "utf-8":
    # for Unix-like systems (macOS and Linux) that can decode chars like ═ and ╝
    logo = f"""\u001b[34m
████████╗██╗███╗   ███╗███████╗    ███████╗███████╗██████╗ ██╗███████╗███████╗    █████╗ ██╗
╚══██╔══╝██║████╗ ████║██╔════╝    ██╔════╝██╔════╝██╔══██╗██║██╔════╝██╔════╝   ██╔══██╗██║
   ██║   ██║██╔████╔██║█████╗█████╗███████╗█████╗  ██████╔╝██║█████╗  ███████╗   ███████║██║
   ██║   ██║██║╚██╔╝██║██╔══╝╚════╝╚════██║██╔══╝  ██╔══██╗██║██╔══╝  ╚════██║   ██╔══██║██║
   ██║   ██║██║ ╚═╝ ██║███████╗    ███████║███████╗██║  ██║██║███████╗███████║██╗██║  ██║██║
   ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝╚═╝
ai4ts v{__version__} - building AI for unified time-series analysis, https://time-series.ai \u001b[0m
"""
else:
    # for Windows platform
    logo = f"""\u001b[34m
████████ ██ ███    ███ ███████       ███████ ███████ ██████  ██ ███████ ███████     █████  ██
   ██    ██ ████  ████ ██            ██      ██      ██   ██ ██ ██      ██         ██   ██ ██
   ██    ██ ██ ████ ██ █████   █████ ███████ █████   ██████  ██ █████   ███████    ███████ ██
   ██    ██ ██  ██  ██ ██                 ██ ██      ██   ██ ██ ██           ██    ██   ██ ██
   ██    ██ ██      ██ ███████       ███████ ███████ ██   ██ ██ ███████ ███████ ██ ██   ██ ██   
ai4ts v{__version__} - building AI for unified time-series analysis, https://time-series.ai \u001b[0m
"""
    logo = logo.encode("utf-8").decode(sys.stdout.encoding, errors="ignore")

print(logo)
