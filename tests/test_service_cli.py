"""Mocked tests for remote TimeSeries.AI CLI commands."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import requests
from click.testing import CliRunner

from ai4ts.cli import cli


@pytest.fixture
def service(monkeypatch, tmp_path):
    from ai4ts.cli import config_utils

    monkeypatch.delenv("TIMESERIESAI_BASE_URL", raising=False)
    monkeypatch.delenv("TIMESERIESAI_API_KEY", raising=False)
    monkeypatch.setattr(config_utils, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_utils, "CONFIG_FILE", tmp_path / "config.json")
    runner = CliRunner()
    assert runner.invoke(cli, ["config", "set-key", "test-key"]).exit_code == 0
    assert runner.invoke(cli, ["config", "set-url", "https://service.example/prefix/"]).exit_code == 0
    calls = []
    replies = []

    def post(_session, url, **kwargs):
        base_url = os.getenv("TIMESERIESAI_BASE_URL", "https://service.example/prefix")
        assert url.startswith(f"{base_url}/timeseriesai/api/")
        assert kwargs["headers"]["authorization"] == "Bearer test-key"
        calls.append((url.rsplit("/", 1)[-1], kwargs))
        status, payload = replies.pop(0)
        response = requests.Response()
        response.url = url
        response.status_code = status
        response._content = b"" if payload is None else f"JSON:{json.dumps(payload)}\n".encode()
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "post", post)
    return lambda *args: runner.invoke(cli, ["service", *args]), calls, replies


def test_service_session_across_invocations(tmp_path, service):
    run, calls, replies = service
    source = tmp_path / "data.csv"
    source.write_text("value\n1\n")
    replies.extend(
        [
            (200, {"chat_session_id": "learned", "max_file_size_in_mb": 1}),
            (200, None),
            (200, {"prediction": 5}),
            (200, None),
            (200, None),
            (200, {"prediction": 6}),
        ]
    )

    learned = run("learn", str(source))
    assert learned.exit_code == 0, learned.output
    assert "Session ID: learned" in learned.output
    forecast = run("--session-id", "learned", "forecast", "--steps", "4")
    assert forecast.exit_code == 0, forecast.output
    assert "prediction: 5" in forecast.output
    assert run("--session-id", "learned", "persist", "--alias", "fixture").exit_code == 0
    restored = run("restore", "saved")
    assert restored.exit_code == 0, restored.output
    assert "Session ID: saved" in restored.output
    forecast = run("--session-id", "saved", "forecast", str(source))
    assert forecast.exit_code == 0, forecast.output
    assert [route for route, _ in calls] == ["init_chat", "learn", "forecast", "persist", "restore", "forecast"]
    assert calls[2][1]["headers"]["chat_session_id"] == "learned"
    assert calls[2][1]["headers"]["n_forecast_steps"] == "4"
    assert calls[-1][1]["headers"]["chat_session_id"] == "saved"
    assert calls[-1][1]["files"]["file"][0] == source.name
    assert calls[-1][1]["files"]["file"][1].closed


def test_service_requires_session_before_network(service):
    run, calls, _ = service
    for command in ("forecast", "impute", "persist"):
        result = run(command)
        assert result.exit_code != 0
        assert "--session-id" in result.output
    assert calls == []


@pytest.mark.parametrize(
    ("command", "args", "headers"),
    [
        ("impute", [], {}),
        ("classify", [], {}),
        ("detect", ["--anomaly-rate", "0.05"], {"anomaly_rate": "0.05"}),
        ("cluster", ["--n-clusters", "4"], {"n_clusters": "4"}),
        ("clean", ["--anomaly-rate", "0.02"], {"anomaly_rate": "0.02"}),
    ],
)
def test_service_task_routes_and_url_precedence(service, monkeypatch, command, args, headers):
    run, calls, replies = service
    monkeypatch.setenv("TIMESERIESAI_BASE_URL", "https://override.example")
    replies.append((200, {"values": [1, 2]}))
    result = run("--session-id", "known", command, *args)
    assert result.exit_code == 0, result.output
    assert [route for route, _ in calls] == [command]
    assert calls[0][1]["headers"] == {"authorization": "Bearer test-key", "chat_session_id": "known", **headers}


@pytest.mark.parametrize(
    ("command", "status", "args"),
    [("forecast", 200, []), ("forecast", 503, []), ("persist", 401, []), ("restore", 404, ["unknown"])],
)
def test_service_failures_have_nonzero_exit(service, command, status, args):
    run, calls, replies = service
    replies.append((status, None))
    result = run("--session-id", "known", command, *args)
    assert result.exit_code != 0
    assert "Error:" in result.output
    assert "✓" not in result.output
    assert len(calls) == 1


def test_forecast_is_under_service():
    runner = CliRunner()
    assert runner.invoke(cli, ["service", "forecast", "--help"]).exit_code == 0
    assert runner.invoke(cli, ["forecast", "--help"]).exit_code != 0
