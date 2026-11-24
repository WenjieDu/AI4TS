"""CLI coverage for raw inspection and protocol validation."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from click.testing import CliRunner

from ai4ts.cli import cli


def test_inspect_uses_source_columns_and_denominators(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("x,note\n1,\n2,ok\n")
    original = source.read_bytes()
    result = CliRunner().invoke(cli, ["data", "inspect", str(source)])
    assert result.exit_code == 0, result.output
    assert "Has SAMPLE_ID : False" in result.stdout
    assert "Has TIMESTAMP : False" in result.stdout
    assert "Missing cells : 1/4 (25.00%) across all columns" in result.stdout
    assert any(
        line.strip().startswith("note") and "1/2 (50.00%)" in line for line in result.stdout.splitlines()
    )
    assert source.read_bytes() == original


def test_validate_reports_verdict_and_protocol_error(tmp_path):
    source = tmp_path / "data.csv"
    runner = CliRunner()
    args = ["data", "validate", str(source)]
    source.write_text("x\n1\n")
    accepted = runner.invoke(cli, args)
    assert accepted.exit_code == 0 and accepted.stdout.strip() == "Validation passed."
    source.write_text("note\nonly_text\n")
    rejected = runner.invoke(cli, args)
    assert rejected.exit_code == 1
    assert "No numeric feature columns found" in rejected.stderr
