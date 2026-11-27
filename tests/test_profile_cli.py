"""Dataset summaries are produced by the AI4TS CLI."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json

from click.testing import CliRunner

from ai4ts.cli import cli
from ai4ts.data_protocols.profile import DataProfile, analyze_dataset


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
    profile = DataProfile.from_dict({"dataset_stats": {"n_samples": 1}})
    assert profile.dataset_stats.per_feature_missing_rate == {}
    assert profile.dataset_stats.per_feature_value_domain == {}
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


def test_profile_json_reports_finite_value_domains_and_gaps_by_entity(tmp_path):
    source = tmp_path / "domains.csv"
    csv = (
        "SAMPLE_ID,x,negative,mixed,empty,nonfinite,CLAF_TARGET\n"
        "NA,5,-2.5,1,,inf,0\n"
        "001,inf,-2.5,-1,,,-1\n"
        "NA,-inf,-2.5,1,,,0\n"
        "001,1,-2.5,-1,,-inf,-1\n"
        "NA,,-2.5,1,,,0\n"
        "001,,-2.5,-1,,,-1\n"
        "NA,0,-2.5,1,,,0\n"
        "001,0,-2.5,-1,,,-1\n"
        "NA,,-2.5,1,,,0\n"
        "NA,,-2.5,1,,,0\n"
        "NA,,-2.5,1,,,0\n"
        "empty,,-2.5,0,,,0\n"
        "empty,,-2.5,0,,,0\n"
    )
    source.write_text(csv)
    result = CliRunner().invoke(cli, ["data", "profile", "--input", str(source), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    domains = payload["dataset_stats"]["per_feature_value_domain"]
    assert set(domains) == {"x", "negative", "mixed", "empty", "nonfinite"}
    assert domains["x"] == {
        "nonnegative": True,
        "integer_valued": True,
        "constant": False,
        "one_signed": True,
        "leading_missing": [
            {"entity": "001", "count": 1},
            {"entity": "NA", "count": 0},
            {"entity": "empty", "count": 2},
        ],
        "longest_gap": {"length": 2, "entity": "NA"},
    }
    assert domains["negative"] == {
        "nonnegative": False,
        "integer_valued": False,
        "constant": True,
        "one_signed": True,
        "leading_missing": [
            {"entity": "001", "count": 0},
            {"entity": "NA", "count": 0},
            {"entity": "empty", "count": 0},
        ],
        "longest_gap": {"length": 0, "entity": None},
    }
    assert domains["mixed"]["one_signed"] is False
    unavailable = {**dict.fromkeys(domains["x"]), "reason": "no_finite_observations"}
    assert domains["empty"] == domains["nonfinite"] == unavailable
    assert DataProfile.from_dict(payload).dataset_stats.per_feature_value_domain == domains
    assert source.read_text() == csv


def test_profile_orders_gaps_by_confirmed_time_within_each_entity(tmp_path):
    source = tmp_path / "unordered.csv"
    source.write_text(
        "entity,time,value,category\n"
        "B,2024-01-03,0,high\nA,2024-01-03,0,low\n"
        "B,2024-01-01,-2,high\nA,2024-01-01,-3,low\n"
        "B,2024-01-02,,high\nA,2024-01-02,,low\n"
    )
    profile = analyze_dataset(
        source,
        schema_mapping={"sample_id_col": "entity", "timestamp_col": "time", "feature_cols": ["value", "category"]},
        categorical_columns=("category",),
    )
    domains = profile.dataset_stats.per_feature_value_domain
    assert domains == {
        "value": {
            "nonnegative": False,
            "integer_valued": True,
            "constant": False,
            "one_signed": True,
            "leading_missing": [{"entity": "A", "count": 0}, {"entity": "B", "count": 0}],
            "longest_gap": {"length": 1, "entity": "A"},
        }
    }
    assert profile.timestamp_info.median_step_seconds == 86400
    assert DataProfile.from_dict(profile.to_dict()) == profile


def test_profile_reports_implicit_entity_and_preserves_integer_precision(tmp_path):
    source = tmp_path / "single.csv"
    source.write_text(
        "TIMESTAMP,value,wide\n"
        "2024-01-01,,9007199254740992\n"
        "invalid,0,9007199254740993\n"
        "2024-01-03,,9007199254740992\n"
        "2024-01-02,0,9007199254740992\n"
    )
    profile = analyze_dataset(source)
    domains = profile.dataset_stats.per_feature_value_domain
    assert domains["value"] == {
        "nonnegative": True,
        "integer_valued": True,
        "constant": True,
        "one_signed": True,
        "leading_missing": [{"entity": None, "count": 1}],
        "longest_gap": {"length": 1, "entity": None},
    }
    assert domains["wide"]["integer_valued"] is True
    assert domains["wide"]["constant"] is False
    assert profile.timestamp_info.warning == "Could not parse 'TIMESTAMP' as datetime."
