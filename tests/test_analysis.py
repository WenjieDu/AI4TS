"""No-model tests of two-dimensional analysis semantics."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import pandas as pd
import pytest

from ai4ts.data_protocols import analysis as a
from ai4ts.data_protocols.ingestion import create_ingestion_plan


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "id": ["001"] * 240,
            "time": pd.date_range("2024-01-01", periods=240, freq="5min"),
            "x": np.sin(np.arange(240)),
            "note": ["keep"] * 240,
        }
    ).to_csv(path, index=False)
    return path


ROLES = dict(sample_id_col="id", timestamp_col="time", feature_cols=["x"], label_col=None)


def test_plan_is_source_bound_relocatable_and_does_not_prepare(csv, tmp_path, monkeypatch):
    import ai4ts.data_protocols.pipeline as pipeline

    monkeypatch.setattr(pipeline, "prepare_for_pypots", lambda *a, **k: pytest.fail("tensor preparation"))
    before = csv.read_bytes()
    plan = a.create_analysis_plan(csv, roles=ROLES, methods=["acf"])
    assert plan["status"] == "ready", plan
    assert plan["groups"][0]["entity"] == "001"
    assert plan["groups"][0]["time"]["regular_step_seconds"] == 300
    moved = tmp_path / "moved.csv"
    moved.write_bytes(before)
    a.validate_analysis_plan(plan, moved)
    moved.write_text(moved.read_text().replace("keep", "changed", 1))
    with pytest.raises(ValueError, match="changed"):
        a.validate_analysis_plan(plan, moved)
    forged = copy.deepcopy(plan)
    forged["selected_rows"] = [0]
    with pytest.raises(ValueError):
        a.validate_analysis_plan(forged, csv)
    assert csv.read_bytes() == before


def test_clarification_and_method_conditions(csv):
    pending = a.create_analysis_plan(csv)
    assert pending["status"] == "needs_clarification" and pending["source"]["candidates"]
    pending = a.create_analysis_plan(csv, roles=ROLES, methods=["stl", "lombscargle"])
    assert pending["status"] == "needs_clarification" and len(pending["unresolved"]) == 2
    plan = a.create_analysis_plan(csv, roles=ROLES, methods=["acf", "pacf", "adf", "kpss", "stl"], period=12)
    assert {r["status"] for r in plan["applicability"]} == {"ready"}
    frame = pd.read_csv(csv, dtype={"id": str})
    frame.loc[2:5, "x"] = np.nan
    frame.to_csv(csv, index=False)
    plan = a.create_analysis_plan(csv, roles=ROLES, methods=["acf", "pacf"])
    assert [r["status"] for r in plan["applicability"]] == ["ready", "unsupported"]


def test_time_defects_are_reported_not_repaired(csv):
    frame = pd.read_csv(csv, dtype={"id": str})
    frame.loc[2, "time"] = frame.loc[0, "time"]
    frame.to_csv(csv, index=False)
    before = csv.read_bytes()
    plan = a.create_analysis_plan(csv, roles=ROLES, methods=["acf"])
    assert plan["status"] == "ready"
    temporal = plan["groups"][0]["time"]
    assert temporal["duplicate_rows"] == [0, 2] and temporal["reversed_rows"] == [2]
    assert plan["applicability"][0]["status"] == "unsupported"
    assert csv.read_bytes() == before


def test_irregular_does_not_imply_missing_schedule(csv):
    frame = pd.read_csv(csv, dtype={"id": str}).drop(index=4)
    frame.to_csv(csv, index=False)
    unknown = a.create_analysis_plan(csv, roles=ROLES)
    assert unknown["groups"][0]["time"]["missing_time_points"] is None
    declared = a.create_analysis_plan(csv, roles=ROLES, expected_frequency="5min")
    assert declared["groups"][0]["time"]["missing_time_points"] == 1


def test_mixed_frequencies_ids_and_missing_values(tmp_path):
    path = tmp_path / "mixed.csv"
    frames = [
        pd.DataFrame(
            {"id": sid, "time": pd.date_range("2024-01-01", periods=40, freq=freq), "x": np.arange(40, dtype=float)}
        )
        for sid, freq in (("001", "5min"), ("NA", "10min"))
    ]
    pd.concat(frames).to_csv(path, index=False)
    source, roles, _ = a.read_analysis_source(path, ROLES)
    assert set(source.id) == {"001", "NA"}
    plan = a.create_analysis_plan(path, roles=ROLES)
    assert [g["time"]["regular_step_seconds"] for g in plan["groups"]] == [300, 600]


def test_baseline_origin_counts_and_unsupported_purpose(csv):
    plan = a.create_analysis_plan(
        csv, "forecast-baselines", ROLES, targets=["x"], split_policy="chronological", seasonal_period=12
    )
    assert plan["status"] == "ready", plan
    group = plan["groups"][0]
    assert {k: len(v) for k, v in group["partitions"].items()} == {"train": 168, "val": 24, "test": 48}
    assert group["origin_row"] == 167 and group["test_fh"] == list(range(25, 73))
    assert plan["budget"]["baseline_prepare"] == 2 and plan["budget"]["baseline_predict"] == 3
    pending = a.create_analysis_plan(csv, "forecast-baselines", ROLES, targets=["x"], split_policy="entity")
    assert pending["status"] != "ready"


def test_existing_split_scope_is_preserved(csv):
    old = create_ingestion_plan(csv, "imputation", roles=ROLES, split_policy="chronological", window_size=12)
    assert old["status"] == "ready", old
    plan = a.create_analysis_plan(csv, roles=ROLES, scope="train", scope_plan=old)
    assert plan["status"] == "ready", plan
    assert plan["selected_rows"] == old["split"]["rows"]["train"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"period": 1},
        {"period_range": [60, 30]},
        {"rows": [-1]},
        {"rows": [3, 2]},
        {"methods": ["acf", "acf"]},
        {"expected_frequency": "-1min"},
    ],
)
def test_invalid_parameters_are_blocked(csv, kwargs):
    assert a.create_analysis_plan(csv, roles=ROLES, **kwargs)["status"] == "blocked"


def test_bounded_reader_and_invalid_numeric(csv, monkeypatch):
    import ai4ts.data_protocols.ingestion as intake

    monkeypatch.setattr(intake, "MAX_CSV_BYTES", 1)
    assert a.create_analysis_plan(csv, roles=ROLES)["status"] == "blocked"
    monkeypatch.setattr(intake, "MAX_CSV_BYTES", 32 * 1024 * 1024)
    csv.write_text("id,time,x\n001,2024-01-01,inf\n")
    assert a.create_analysis_plan(csv, roles=ROLES)["status"] == "blocked"
    csv.write_text("id,x,x\n001,1,2\n")
    assert a.create_analysis_plan(csv, roles=ROLES)["status"] == "blocked"


def test_unknown_units_and_feature_role_conflicts(csv):
    plan = a.create_analysis_plan(csv, roles=ROLES)
    assert plan["feature_semantics"] is None
    confirmed = a.create_analysis_plan(csv, roles=ROLES, feature_units={"x": "degC"})
    assert confirmed["feature_semantics"]["units"]["x"] == "degC"
    assert a.create_analysis_plan(csv, roles={**ROLES, "feature_cols": ["id"]})["status"] == "blocked"
