"""Regression coverage for table, sample, and archive conversion."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import builtins

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from ai4ts.cli import cli


def test_convert_keeps_nan_unless_forward_fill_is_requested(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x,y\n1,10\n,20\n3,30\n")
    runner = CliRunner()
    args = ["data", "convert", str(source), "--context-len", "4"]
    default_output = tmp_path / "default.npy"
    default = runner.invoke(cli, [*args, "--output", str(default_output)])
    assert default.exit_code == 0, default.output
    samples = np.load(default_output)
    assert samples.shape == (1, 6, 2)
    np.testing.assert_array_equal(samples[0, 2:5], [[1, 10], [np.nan, 20], [3, 30]])

    explicit_output = tmp_path / "explicit.npy"
    explicit = runner.invoke(cli, [*args, "--output", str(explicit_output), "--mode", "samples"])
    assert explicit.exit_code == 0, explicit.output
    np.testing.assert_array_equal(np.load(explicit_output), samples)

    filled_output = tmp_path / "filled.npy"
    filled = runner.invoke(
        cli, [*args, "--output", str(filled_output), "--missing-value-strategy", "forward_fill"]
    )
    assert filled.exit_code == 0, filled.output
    samples[0, 3, 0] = 1
    np.testing.assert_array_equal(np.load(filled_output), samples)


def test_convert_missing_value_option_applies_only_to_csv_input(tmp_path):
    source = tmp_path / "source.npy"
    np.save(source, np.array([[1.0], [np.nan]]))
    result = CliRunner().invoke(cli, ["data", "convert", str(source), "--missing-value-strategy", "forward_fill"])
    assert result.exit_code == 2
    assert "Only supported for CSV to NPY" in result.stderr
    assert not source.with_suffix(".csv").exists()



def test_table_round_trip_retains_row_order_column_order_and_nan(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("z,a\n2,90\n1,\n,70\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["data", "convert", str(source), "--mode", "table"])
    assert result.exit_code == 0, result.output
    array_path = source.with_suffix(".npy")
    expected = np.array([[2, 90], [1, np.nan], [np.nan, 70]])
    np.testing.assert_array_equal(np.load(array_path), expected)

    for mode in ([], ["--mode", "table"]):
        output = tmp_path / "restored.csv"
        result = runner.invoke(cli, ["data", "convert", str(array_path), "-o", str(output), *mode])
        assert result.exit_code == 0, result.output
        restored = pd.read_csv(output)
        assert list(restored) == ["feat_0", "feat_1"]
        np.testing.assert_array_equal(restored.to_numpy(), expected)


@pytest.mark.parametrize("content", ["SAMPLE_ID,x\n1,3\n2,4\n", "a,b\nhello,1\nworld,2\n"])
def test_table_rejects_role_columns_and_non_numeric_content(tmp_path, content):
    source = tmp_path / "source.csv"
    source.write_text(content)
    result = CliRunner().invoke(cli, ["data", "convert", str(source), "--mode", "table"])
    assert result.exit_code == 1 and "Table mode requires" in result.stderr
    assert not source.with_suffix(".npy").exists()


def test_sample_export_requires_conversion_dependencies(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n2\n")
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "data_protocols.conversion":
            raise ImportError("simulated converter import failure")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    result = CliRunner().invoke(cli, ["data", "convert", str(source)])
    assert result.exit_code == 1 and "Conversion dependencies unavailable" in result.stderr
    assert not source.with_suffix(".npy").exists()


def test_samples_keep_validation_errors_at_the_cli_boundary(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("SAMPLE_ID,x\n1,1\n1,2\n1,3\n2,4\n2,5\n")
    result = CliRunner().invoke(cli, ["data", "convert", str(source), "--mode", "samples", "--context-len", "2"])
    assert result.exit_code == 1
    assert "Sample with 3 timesteps exceeds ts_context_len=2" in result.stderr
    assert not source.with_suffix(".npy").exists()


def test_samples_to_csv_preserves_stored_arrays_and_checks_mode(tmp_path):
    source = tmp_path / "samples.npy"
    np.save(source, np.array([[[1], [np.nan]], [[20], [30]]]))
    runner = CliRunner()
    incompatible = runner.invoke(cli, ["data", "convert", str(source), "--mode", "table"])
    assert incompatible.exit_code == 1 and "requires 2 dimensions" in incompatible.stderr
    assert not source.with_suffix(".csv").exists()

    result = runner.invoke(cli, ["data", "convert", str(source), "--mode", "samples"])
    assert result.exit_code == 0, result.output
    frame = pd.read_csv(source.with_suffix(".csv"))
    assert list(frame) == ["feat_0", "SAMPLE_ID"]
    np.testing.assert_array_equal(frame["feat_0"], [1, np.nan, 20, 30])
    assert frame["SAMPLE_ID"].tolist() == [0, 0, 1, 1]


def test_npz_extraction_requires_and_preserves_selected_key(tmp_path):
    source = tmp_path / "dataset.npz"
    selected = np.array([[3, np.nan], [1, 0]], dtype=np.float32)
    np.savez(source, X=np.zeros((2, 3, 2)), chosen=selected)
    runner = CliRunner()
    args = ["data", "convert", str(source)]
    output = source.with_suffix(".npy")
    implicit = runner.invoke(cli, args)
    assert implicit.exit_code == 2 and "requires --key" in implicit.stderr
    assert not output.exists()
    wrong = runner.invoke(cli, [*args, "--key", "missing"])
    assert wrong.exit_code == 1 and "not found" in wrong.stderr
    assert not output.exists()
    explicit = runner.invoke(cli, [*args, "--key", "chosen", "--mode", "table"])
    assert explicit.exit_code == 0, explicit.output
    extracted = np.load(output)
    assert extracted.dtype == selected.dtype
    np.testing.assert_array_equal(extracted, selected)
