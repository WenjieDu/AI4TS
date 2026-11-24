"""Dataset summaries are produced by the AI4TS CLI."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json

from click.testing import CliRunner

from ai4ts.cli import cli


def test_profile_prints_summary_and_json(tmp_path):
    source = tmp_path / "profile.csv"
    source.write_text(
        "SAMPLE_ID,x,y,CLAF_TARGET\n001,1,2,0\n001,,3,0\nNA,4,,1\nNA,6,,1\nNA,0,8,1\n"
    )
    runner = CliRunner()
    args = ["data", "profile", "--input", str(source)]
    human = runner.invoke(cli, args)
    assert human.exit_code == 0, human.output
    assert all(field in human.stdout for field in ("Samples:", "Features:", "Suggested strategy:"))
    assert "min=2, max=3, avg=2.5" in human.stdout
    assert "Task hint (inferred): classification" in human.stdout
    assert "x: 20.00%" in human.stdout
    assert "y: 40.00%" in human.stdout
    machine = runner.invoke(cli, [*args, "--json"])
    assert machine.exit_code == 0, machine.output
    value = json.loads(machine.stdout)
    assert (value["task_type"], value["task_type_source"]) == ("classification", "inferred")
    stats = value["dataset_stats"]
    assert (stats["n_samples"], stats["n_features"], stats["total_rows"]) == (2, 2, 5)
    assert stats["missing_rate"] == 0.3
    assert stats["per_feature_missing_rate"] == {"x": 0.2, "y": 0.4}
    assert (stats["min_sample_length"], stats["max_sample_length"], stats["avg_sample_length"]) == (2, 3, 2.5)
    assert stats["has_variable_length"] is True
    assert value["sample_lengths"] == [2, 3]
    assert value["schema_mapping"]["sample_id_col"] == "SAMPLE_ID"



def test_saved_profile_round_trips_feature_rates_and_accepts_older_profiles():
    from ai4ts.data_protocols.profile import DataProfile

    profile = DataProfile.from_dict({"dataset_stats": {"n_samples": 1}})
    assert profile.dataset_stats.per_feature_missing_rate == {}
    profile.dataset_stats.per_feature_missing_rate = {"x": 0.25}
    assert DataProfile.from_dict(profile.to_dict()) == profile



def test_profile_reports_sampling_and_an_explicit_task_hint(tmp_path):
    source = tmp_path / "series.csv"
    source.write_text("TIMESTAMP,x\n2024-01-01T00:00:00Z,1\n2024-01-01T00:05:00Z,2\n")
    runner = CliRunner()
    args = ["data", "profile", "--input", str(source), "--task", "forecasting"]
    human = runner.invoke(cli, args)
    assert human.exit_code == 0, human.output
    assert "Task hint (provided): forecasting" in human.stdout
    assert "Sampling: regular, median interval 300s" in human.stdout
    machine = runner.invoke(cli, [*args, "--json"])
    assert machine.exit_code == 0, machine.output
    value = json.loads(machine.stdout)
    assert (value["task_type"], value["task_type_source"]) == ("forecasting", "provided")
