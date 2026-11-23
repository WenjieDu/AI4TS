"""
Configuration management commands for the AI4TS CLI.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import click

from .config_utils import load_config, save_config, CONFIG_FILE


@click.group()
def config():
    """Manage AI4TS configuration.

    Configuration is stored at ~/.ai4ts/config.json.
    Environment variables take precedence over config file values.
    """
    pass


@config.command("show")
def config_show():
    """Display the current configuration."""
    cfg = load_config()

    api_key = cfg.get("api_key", "")
    if api_key:
        masked = api_key[:4] + "****" + api_key[-4:] if len(api_key) > 8 else "****"
    else:
        masked = click.style("(not set)", fg="yellow")

    base_url = cfg.get("base_url", "")
    if not base_url:
        base_url = click.style("(default)", fg="yellow")

    click.echo(click.style("AI4TS Configuration", fg="cyan", bold=True))
    click.echo(f"  Config file : {CONFIG_FILE}")
    click.echo(f"  API key     : {masked}")
    click.echo(f"  Base URL    : {base_url}")
    click.echo()
    click.echo(
        click.style("Tip: ", fg="green")
        + "Environment variables TIMESERIESAI_API_KEY and TIMESERIESAI_BASE_URL override config file values."
    )


@config.command("set-key")
@click.argument("key")
def config_set_key(key):
    """Set the API key for authentication.

    KEY is the API key obtained from https://time-series.ai.

    You can also set the TIMESERIESAI_API_KEY environment variable instead.
    """
    cfg = load_config()
    cfg["api_key"] = key
    save_config(cfg)
    click.echo(click.style("✓", fg="green") + " API key saved to config.")
    click.echo(
        click.style("Tip: ", fg="green")
        + "You can also set the TIMESERIESAI_API_KEY environment variable."
    )


@config.command("set-url")
@click.argument("url")
def config_set_url(url):
    """Set the base URL for the API server.

    URL is the base URL of the TimeSeries AI platform (e.g. https://dash.time-series.ai).

    You can also set the TIMESERIESAI_BASE_URL environment variable instead.
    """
    cfg = load_config()
    cfg["base_url"] = url
    save_config(cfg)
    click.echo(click.style("✓", fg="green") + f" Base URL set to {url}")
    click.echo(
        click.style("Tip: ", fg="green")
        + "You can also set the TIMESERIESAI_BASE_URL environment variable."
    )
