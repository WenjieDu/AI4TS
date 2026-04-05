"""
Task commands for the AI4TS CLI.

Wraps the TimeSeriesAI client to provide CLI access to all analysis tasks.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import sys
from typing import Optional

import click

from .config_utils import get_api_key, get_base_url


def _get_client():
    """Create and return a configured TimeSeriesAI client."""
    api_key = get_api_key()
    if not api_key:
        click.echo(
            click.style("Error: ", fg="red")
            + "API key not configured. Set it with:\n"
            "  ai4ts config set-key YOUR_KEY\n"
            "  or export TIMESERIESAI_API_KEY=YOUR_KEY"
        )
        sys.exit(1)

    try:
        from ..client import TimeSeriesAI
    except ImportError as e:
        click.echo(click.style("Error: ", fg="red") + f"Failed to import client: {e}")
        sys.exit(1)

    import os
    base_url = get_base_url()
    os.environ["TIMESERIESAI_BASE_URL"] = base_url

    try:
        client = TimeSeriesAI(api_key=api_key)
    except Exception as e:
        click.echo(click.style("Error: ", fg="red") + f"Failed to initialize client: {e}")
        sys.exit(1)

    return client


def _display_result(result, label="Result"):
    """Display a task result in a readable format."""
    if result is None:
        click.echo(click.style("⚠ No result returned.", fg="yellow"))
        return

    try:
        import numpy as np
        if isinstance(result, np.ndarray):
            click.echo(click.style(f"{label}:", fg="cyan", bold=True))
            click.echo(f"  Shape: {result.shape}")
            click.echo(f"  Dtype: {result.dtype}")
            if result.size <= 50:
                click.echo(f"  Data:\n{result}")
            else:
                click.echo(f"  Preview (first 5 rows):\n{result[:5]}")
            return
    except ImportError:
        pass

    if isinstance(result, dict):
        click.echo(click.style(f"{label}:", fg="cyan", bold=True))
        for k, v in result.items():
            click.echo(f"  {k}: {v}")
    else:
        click.echo(click.style(f"{label}:", fg="cyan", bold=True))
        click.echo(f"  {result}")


# --- learn ---

@click.command()
@click.argument("file", type=click.Path(exists=True))
def learn(file):
    """Feed data to the AI model for learning.

    FILE is the path to a CSV file containing training data.
    """
    client = _get_client()
    click.echo(f"Feeding data from {click.style(file, bold=True)} to AI model ...")

    try:
        result = client.learn(file)
        click.echo(click.style("✓ Learning complete", fg="green", bold=True))
        _display_result(result, "Learning result")
    except Exception as e:
        click.echo(click.style("✗ Learning failed", fg="red", bold=True))
        click.echo(f"  {e}")
        sys.exit(1)


# --- impute ---

@click.command()
@click.argument("file", type=click.Path(exists=True), required=False, default=None)
def impute(file):
    """Impute missing values in time-series data.

    FILE is an optional path to a CSV file with missing values.
    If omitted, uses the data from the current session.
    """
    client = _get_client()
    click.echo("Running imputation ...")

    try:
        result = client.impute(data=file)
        click.echo(click.style("✓ Imputation complete", fg="green", bold=True))
        _display_result(result, "Imputed data")
    except Exception as e:
        click.echo(click.style("✗ Imputation failed", fg="red", bold=True))
        click.echo(f"  {e}")
        sys.exit(1)


# --- forecast ---

@click.command()
@click.argument("file", type=click.Path(exists=True), required=False, default=None)
@click.option("--steps", "-n", type=int, default=None, help="Number of future steps to forecast.")
def forecast(file, steps):
    """Forecast future values for time-series data.

    FILE is an optional path to a CSV file with historical data.
    """
    client = _get_client()
    click.echo("Running forecasting ...")

    try:
        result = client.forecast(data=file, n_forecast_steps=steps)
        click.echo(click.style("✓ Forecasting complete", fg="green", bold=True))
        _display_result(result, "Forecast")
    except Exception as e:
        click.echo(click.style("✗ Forecasting failed", fg="red", bold=True))
        click.echo(f"  {e}")
        sys.exit(1)


# --- classify ---

@click.command()
@click.argument("file", type=click.Path(exists=True), required=False, default=None)
def classify(file):
    """Classify time-series data.

    FILE is an optional path to a CSV file with time-series samples.
    """
    client = _get_client()
    click.echo("Running classification ...")

    try:
        result = client.classify(data=file)
        click.echo(click.style("✓ Classification complete", fg="green", bold=True))
        _display_result(result, "Classification result")
    except Exception as e:
        click.echo(click.style("✗ Classification failed", fg="red", bold=True))
        click.echo(f"  {e}")
        sys.exit(1)


# --- detect ---

@click.command()
@click.argument("file", type=click.Path(exists=True), required=False, default=None)
@click.option("--anomaly-rate", "-r", type=float, default=0.01, help="Expected anomaly rate (0.0-1.0).")
def detect(file, anomaly_rate):
    """Detect anomalies in time-series data.

    FILE is an optional path to a CSV file with time-series data.
    """
    client = _get_client()
    click.echo("Running anomaly detection ...")

    try:
        result = client.detect(data=file, anomaly_rate=anomaly_rate)
        click.echo(click.style("✓ Anomaly detection complete", fg="green", bold=True))
        _display_result(result, "Detection result")
    except Exception as e:
        click.echo(click.style("✗ Anomaly detection failed", fg="red", bold=True))
        click.echo(f"  {e}")
        sys.exit(1)


# --- cluster ---

@click.command()
@click.argument("file", type=click.Path(exists=True), required=False, default=None)
@click.option("--n-clusters", "-k", type=int, default=None, help="Number of clusters to form.")
def cluster(file, n_clusters):
    """Cluster time-series data.

    FILE is an optional path to a CSV file with time-series data.
    """
    client = _get_client()
    click.echo("Running clustering ...")

    try:
        result = client.cluster(data=file, n_clusters=n_clusters)
        click.echo(click.style("✓ Clustering complete", fg="green", bold=True))
        _display_result(result, "Clustering result")
    except Exception as e:
        click.echo(click.style("✗ Clustering failed", fg="red", bold=True))
        click.echo(f"  {e}")
        sys.exit(1)


# --- clean ---

@click.command()
@click.argument("file", type=click.Path(exists=True), required=False, default=None)
@click.option("--anomaly-rate", "-r", type=float, default=0.0, help="Expected anomaly rate (0.0-1.0).")
def clean(file, anomaly_rate):
    """Clean and denoise time-series data.

    FILE is an optional path to a CSV file with noisy time-series data.
    """
    client = _get_client()
    click.echo("Running data cleaning ...")

    try:
        result = client.clean(data=file, anomaly_rate=anomaly_rate)
        click.echo(click.style("✓ Cleaning complete", fg="green", bold=True))
        _display_result(result, "Cleaned data")
    except Exception as e:
        click.echo(click.style("✗ Cleaning failed", fg="red", bold=True))
        click.echo(f"  {e}")
        sys.exit(1)


# --- persist ---

@click.command()
@click.option("--alias", "-a", type=str, default=None, help="Friendly alias for the saved session.")
def persist(alias):
    """Save the current AI model session.

    Persists the learned context so it can be restored later.
    """
    client = _get_client()
    click.echo("Persisting session ...")

    try:
        client.persist(alias=alias)
        click.echo(click.style("✓ Session persisted", fg="green", bold=True))
        if client.chat_session_id:
            click.echo(f"  Session ID: {client.chat_session_id}")
    except Exception as e:
        click.echo(click.style("✗ Persist failed", fg="red", bold=True))
        click.echo(f"  {e}")
        sys.exit(1)


# --- restore ---

@click.command()
@click.argument("session_id", type=str)
def restore(session_id):
    """Restore a previously saved AI model session.

    SESSION_ID is the identifier of the session to restore.
    """
    client = _get_client()
    click.echo(f"Restoring session {click.style(session_id, bold=True)} ...")

    try:
        client.restore(session_id=session_id)
        click.echo(click.style("✓ Session restored", fg="green", bold=True))
    except Exception as e:
        click.echo(click.style("✗ Restore failed", fg="red", bold=True))
        click.echo(f"  {e}")
        sys.exit(1)
