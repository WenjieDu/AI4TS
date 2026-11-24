"""
Main CLI group for the AI4TS package.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import click

from .analyze import analyze

from .config_cmd import config
from .data_cmd import data
from .services import service
from .version_cmd import version


@click.group()
def cli():
    """AI4TS - AI for Time Series Analysis.

    Command-line interface for the TimeSeries AI platform.
    Validate, convert, and analyze time series data.
    """
    pass


cli.add_command(config)
cli.add_command(data)
cli.add_command(analyze)
cli.add_command(service)
cli.add_command(version)
