"""
Remote TimeSeries.AI service commands for the AI4TS CLI.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import click

from .config_utils import get_api_key, get_base_url


@click.group()
@click.option("--session-id", help="Use the session ID printed by service learn or service restore.")
def service(session_id):
    """Run remote TimeSeries.AI tasks and manage their sessions."""


_CSV_FILE = click.Path(exists=True, dir_okay=False)


def _get_client(*, require_session=False):
    """Create a client for the selected service and session."""
    ctx = click.get_current_context()
    session_id = ctx.parent.params["session_id"] if require_session else None
    if require_session and not (session_id and session_id.strip()):
        raise click.UsageError("Supply --session-id before the task, using the ID from service learn or service restore.")

    api_key = get_api_key()
    if not api_key:
        raise click.ClickException(
            "API key not configured. Use ai4ts-cli config set-key or set TIMESERIESAI_API_KEY."
        )

    from ..client import TimeSeriesAI

    client = TimeSeriesAI(api_key=api_key, base_url=get_base_url())
    client.chat_session_id = session_id
    ctx.call_on_close(client.http_session.close)
    return client


def _call(method, **kwargs):
    try:
        return method(**kwargs)
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error


def _display_result(result, label):
    """Display a task result in a readable format."""
    import numpy as np

    click.echo(click.style(f"{label}:", fg="cyan", bold=True))
    if isinstance(result, np.ndarray):
        click.echo(f"  Shape: {result.shape}")
        click.echo(f"  Dtype: {result.dtype}")
        if result.size <= 50:
            click.echo(f"  Data:\n{result}")
        else:
            click.echo(f"  Preview (first 5 rows):\n{result[:5]}")
    elif isinstance(result, dict):
        for key, value in result.items():
            click.echo(f"  {key}: {value}")
    else:
        click.echo(f"  {result}")


def _run_task(operation, label, file, **kwargs):
    client = _get_client(require_session=True)
    result = _call(getattr(client, operation), data=file, **kwargs)
    if result is None:
        raise click.ClickException(f"The service returned no {label.lower()} result.")
    _display_result(result, label)


@service.command()
@click.argument("file", type=_CSV_FILE)
def learn(file):
    """Upload training CSV data and print the new session ID."""
    client = _get_client()
    result = _call(client.learn, data=file)
    if result is not None:
        _display_result(result, "Learning result")
    click.echo(f"Session ID: {client.chat_session_id}")


@service.command()
@click.argument("file", type=_CSV_FILE, required=False)
def impute(file):
    """Impute missing values in a CSV or in the session data."""
    _run_task("impute", "Imputed data", file)


@service.command()
@click.argument("file", type=_CSV_FILE, required=False)
@click.option("--steps", "-n", type=int, help="Number of future steps to forecast.")
def forecast(file, steps):
    """Forecast future values from a CSV or the session data."""
    _run_task("forecast", "Forecast", file, n_forecast_steps=steps)


@service.command()
@click.argument("file", type=_CSV_FILE, required=False)
def classify(file):
    """Classify time-series samples from a CSV or the session data."""
    _run_task("classify", "Classification", file)


@service.command()
@click.argument("file", type=_CSV_FILE, required=False)
@click.option("--anomaly-rate", "-r", type=float, default=0.01, help="Expected anomaly rate (0.0-1.0).")
def detect(file, anomaly_rate):
    """Detect anomalies in a CSV or the session data."""
    _run_task("detect", "Detection", file, anomaly_rate=anomaly_rate)


@service.command()
@click.argument("file", type=_CSV_FILE, required=False)
@click.option("--n-clusters", "-k", type=int, help="Number of clusters to form.")
def cluster(file, n_clusters):
    """Cluster time series from a CSV or the session data."""
    _run_task("cluster", "Clustering", file, n_clusters=n_clusters)


@service.command()
@click.argument("file", type=_CSV_FILE, required=False)
@click.option("--anomaly-rate", "-r", type=float, default=0.0, help="Expected anomaly rate (0.0-1.0).")
def clean(file, anomaly_rate):
    """Clean and denoise a CSV or the session data."""
    _run_task("clean", "Cleaned data", file, anomaly_rate=anomaly_rate)


@service.command()
@click.option("--alias", "-a", help="Friendly alias for the saved session.")
def persist(alias):
    """Persist a service session for later restoration."""
    client = _get_client(require_session=True)
    _call(client.persist, alias=alias)
    click.echo(f"Session persisted: {client.chat_session_id}")


@service.command()
@click.argument("session_id")
def restore(session_id):
    """Restore a persisted service session and print its session ID."""
    client = _get_client()
    _call(client.restore, session_id=session_id)
    click.echo(f"Session ID: {client.chat_session_id}")
