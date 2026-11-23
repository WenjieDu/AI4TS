"""Generic time preview and opt-in ingestion v2 CLI, without models/scalers."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import pytest

from click.testing import CliRunner
import pandas as pd

from ai4ts.cli.data_cmd import data


@pytest.mark.parametrize(
    "args",
    [["--time-basis", "wrong"], ["--time-basis", "observation_index"], ["--time-basis=wrong", "--task", "unknown"]],
)
def test_new_plan_parse_errors_are_json_but_legacy_errors_stay_legacy(args):
    new = CliRunner().invoke(data, ["plan", *args])
    assert new.exit_code == 1 and json.loads(new.stdout)["status"] == "failed"
    old = CliRunner().invoke(data, ["plan"])
    assert old.exit_code == 2 and old.stdout == ""


def source(tmp_path):
    path = tmp_path / "source.csv"
    pd.DataFrame(
        dict(
            time=[
                str(pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(minutes=i + (i >= 5) * 2)) for i in range(24)
            ],
            x=range(24),
        )
    ).to_csv(path, index=False)
    return path


def test_time_preview_is_single_json(tmp_path):
    path = source(tmp_path)
    result = CliRunner().invoke(
        data,
        [
            "time-basis",
            "--input",
            str(path),
            "--timestamp",
            "time",
            "--features",
            "x",
        ],
    )
    assert result.exit_code == 0, result.output
    value = json.loads(result.stdout)
    assert value["status"] == "ready" and "model_profile" not in value
    assert value["source"]["rows"] == 24
    result = CliRunner().invoke(data, ["time-basis", "--input", str(path), "--features", "x"])
    assert result.exit_code == 2 and json.loads(result.output)["status"] == "needs_clarification"
    result = CliRunner().invoke(data, ["time-basis", "--unknown"])
    assert result.exit_code == 1 and json.loads(result.output)["status"] == "failed"


def test_time_plan_states_and_old_default_compatibility(tmp_path):
    path = source(tmp_path)
    common = [
        "plan",
        "--input",
        str(path),
        "--task",
        "imputation",
        "--timestamp",
        "time",
        "--features",
        "x",
        "--split_policy",
        "chronological",
        "--window_size",
        "2",
    ]
    runner = CliRunner()
    args = [
        "--time-basis",
        "observation_index",
        "--time-confirmation",
        "Accepted record counts and ignored intervals",
    ]
    result = runner.invoke(data, [*common, "--output", str(tmp_path / "v2"), *args])
    assert result.exit_code == 0, result.output
    record = json.loads((tmp_path / "v2/plan.json").read_text())
    assert record["version"] == 2 and record["parameters"]["allow_irregular"] is True
    result = runner.invoke(
        data,
        [
            *common,
            "--output",
            str(tmp_path / "grid"),
            "--time-basis",
            "regular_grid",
            "--time-confirmation",
            "Accepted grid NaNs",
            "--frequency",
            "1min",
        ],
    )
    assert result.exit_code == 2 and json.loads(result.output)["status"] == "needs_preparation", result.output
    result = runner.invoke(data, [*common, "--output", str(tmp_path / "old"), "--allow_irregular"])
    assert result.exit_code == 0
    assert json.loads((tmp_path / "old/plan.json").read_text())["version"] == 1
    result = runner.invoke(data, [*common, "--output", str(tmp_path / "v2"), *args])
    assert result.exit_code == 1 and json.loads(result.output)["status"] == "failed"


def test_time_preview_never_imports_nn_or_statistical_backends(tmp_path):
    path = source(tmp_path)
    code = """import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'pypots','torch','transformers','statsmodels','sktime'}:raise RuntimeError('forbidden '+fullname)
sys.meta_path.insert(0,Block())
from ai4ts.cli import cli
cli.main(args=sys.argv[1:])
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            code,
            "data",
            "time-basis",
            "--input",
            str(path),
            "--timestamp",
            "time",
            "--features",
            "x",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "ready"
