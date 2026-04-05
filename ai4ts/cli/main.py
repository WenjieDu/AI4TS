"""
Main CLI group for the AI4TS package.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import click

from .config_cmd import config
from .data_cmd import data
from .task_cmd import learn, impute, forecast, classify, detect, cluster, clean, persist, restore
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
cli.add_command(learn)
cli.add_command(impute)
cli.add_command(forecast)
cli.add_command(classify)
cli.add_command(detect)
cli.add_command(cluster)
cli.add_command(clean)
cli.add_command(persist)
cli.add_command(restore)
cli.add_command(version)
