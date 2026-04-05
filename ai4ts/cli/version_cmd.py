"""
Version command for the AI4TS CLI.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import click

from ..version import __version__


@click.command()
def version():
    """Display the AI4TS version."""
    click.echo(f"ai4ts v{__version__}")
