"""Temporal CLI plan, run, inspect, and JSON errors."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json

from click.testing import CliRunner

from ai4ts.cli import cli
from ai4ts.data_protocols.temporal_preprocessing import create_temporal_plan


def test_temporal_json_plan_run_inspect(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("id,time,x\n001,2024-01-01T00:00:00Z,1\n001,2024-01-01T00:10:00Z,3\n")
    runner = CliRunner()
    missing = runner.invoke(
        cli,
        [
            "data",
            "temporal",
            "plan",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "missing"),
            "--mode",
            "regularize",
        ],
    )
    assert missing.exit_code == 2 and json.loads(missing.stdout)["status"] == "needs_clarification"
    plan_dir = tmp_path / "plan"
    result = runner.invoke(
        cli,
        [
            "data",
            "temporal",
            "plan",
            "--input",
            str(source),
            "--output",
            str(plan_dir),
            "--mode",
            "regularize",
            "--frequency",
            "5min",
            "--sample-id",
            "id",
            "--timestamp",
            "time",
            "--features",
            "x",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    plan = json.loads((plan_dir / "plan.json").read_text())
    direct = create_temporal_plan(
        source,
        mode="regularize",
        frequency="5min",
        roles={"sample_id_col": "id", "timestamp_col": "time", "feature_cols": ["x"]},
    )
    assert plan == direct
    output = tmp_path / "result"
    result = runner.invoke(
        cli,
        [
            "data",
            "temporal",
            "run",
            "--input",
            str(source),
            "--plan",
            str(plan_dir / "plan.json"),
            "--output",
            str(output),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0 and json.loads(result.stdout)["status"] == "success", result.output
    inspected = runner.invoke(cli, ["data", "temporal", "inspect", "--result", str(output)], catch_exceptions=False)
    assert inspected.exit_code == 0 and json.loads(inspected.stdout)["summary"]["grid_added_cells"] == 1


def test_temporal_parse_errors_and_budget_are_one_json(tmp_path):
    runner = CliRunner()
    for args in (["run", "--bad"], ["plan", "--mode", "unknown"], ["inspect"]):
        result = runner.invoke(cli, ["data", "temporal", *args])
        assert result.exit_code == 1 and json.loads(result.stdout)["status"] == "failed", result.output
    result = runner.invoke(cli, ["data", "temporal", "inspect", "--result", str(tmp_path), "--seconds", "31"])
    assert result.exit_code == 1 and "Budget" in json.loads(result.stdout)["error"]
