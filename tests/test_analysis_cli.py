"""Analysis CLI JSON, clarification, role reuse and optional-backend boundaries."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys

from click.testing import CliRunner

from ai4ts.cli.analyze import analyze


def test_capabilities_and_old_cli_stay_lazy_without_optional_packages():
    code = """
import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'pypots', 'numpy', 'torch', 'statsmodels', 'sktime', 'scipy'}:
            raise ImportError('controlled optional backend import failure')
sys.meta_path.insert(0, Deny())
from click.testing import CliRunner
from ai4ts.cli import cli
from ai4ts.analysis import capabilities
original = capabilities.metadata.version
def missing(name):
    if name == 'statsmodels':
        raise capabilities.metadata.PackageNotFoundError(name)
    return original(name)
capabilities.metadata.version = missing
assert CliRunner().invoke(cli, ['--help']).exit_code == 0
result = CliRunner().invoke(cli, ['analyze', 'capabilities', '--json'])
assert result.exit_code == 0, result.output
import json
facts = json.loads(result.stdout)
assert facts['methods']['acf']['dependency_status'] == 'not_installed'
assert facts['methods']['seasonal-naive']['backend'] == 'numpy'
assert facts['methods']['seasonal-naive']['dependency_status'] == 'installed_not_probed'
assert not any(m in sys.modules for m in ('numpy','torch','statsmodels','sktime','scipy'))
print('lazy_optional_failure_check_passed')
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_clarification_and_source_matching_role_reuse(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("id,time,x\n001,2024-01-01T00:00:00Z,1\n001,2024-01-01T00:05:00Z,2\n")
    runner = CliRunner()
    result = runner.invoke(analyze, ["plan", "--input", str(source), "--output", str(tmp_path / "pending")])
    assert result.exit_code == 2 and json.loads(result.stdout)["status"] == "needs_clarification"
    result = runner.invoke(
        analyze,
        [
            "plan",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "confirmed"),
            "--sample-id",
            "id",
            "--timestamp",
            "time",
            "--features",
            "x",
        ],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        analyze,
        [
            "plan",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "reuse"),
            "--confirmed-plan",
            str(tmp_path / "confirmed/plan.json"),
            "--features",
            "x",
            "--methods",
            "stl",
        ],
    )
    assert result.exit_code == 2, result.output
    plan = json.loads((tmp_path / "reuse/plan.json").read_text())
    assert plan["roles"]["sample_id_col"] == "id" and plan["roles"]["timestamp_col"] == "time"
    assert plan["unresolved"] == ["Declare the STL period in observations"]
    assert not list(tmp_path.rglob("calls.jsonl"))


def test_parse_failure_is_one_json():
    result = CliRunner().invoke(analyze, ["run", "--unknown"])
    assert result.exit_code == 1 and json.loads(result.stdout)["status"] == "failed"


def test_run_uses_the_same_api(monkeypatch, tmp_path):
    from ai4ts import analysis

    source, plan = tmp_path / "source.csv", tmp_path / "plan.json"
    source.touch()
    plan.write_text(json.dumps(dict(request=dict(kind="diagnostics"))))
    calls = []

    def diagnose(*args):
        calls.append(args)
        return dict(status="partial", fact="same-public-api")

    monkeypatch.setattr(analysis, "diagnose", diagnose)
    result = CliRunner().invoke(
        analyze, ["run", "--input", str(source), "--plan", str(plan), "--output", str(tmp_path / "result")]
    )
    assert result.exit_code == 2 and json.loads(result.stdout)["fact"] == "same-public-api" and len(calls) == 1
