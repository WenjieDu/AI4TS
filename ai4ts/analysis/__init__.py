"""Source-bound general time-series analysis with optional statistical backends."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from .capabilities import available_methods

__all__ = ["available_methods", "diagnose", "compare_baselines", "inspect_analysis"]


def __getattr__(name):
    if name in ("diagnose", "compare_baselines"):
        from . import artifacts

        return getattr(artifacts, name)
    if name == "inspect_analysis":
        from .verify import inspect_analysis

        return inspect_analysis
    raise AttributeError(name)
