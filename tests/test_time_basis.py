"""Source-count-independent time choices and sparse grid support; no fitted state."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai4ts.data_protocols.ingestion import (
    create_ingestion_plan,
    validate_ingestion_plan,
    content_sha256,
)

ROLES = dict(sample_id_col="id", timestamp_col="time", feature_cols=["x"], label_col=None)


def test_saved_choice_does_not_silently_change_window_or_target_context(tmp_path):
    path = source(tmp_path)
    api()
    original = plan(path, choice())
    changed = create_ingestion_plan(
        path,
        "imputation",
        roles=ROLES,
        split_policy="chronological",
        window_size=3,
        time_selection=original["time_selection"],
    )
    assert changed["status"] == "blocked" and "context" in " ".join(changed["errors"])


def api():
    assert (Path(__file__).parents[1] / "ai4ts/data_protocols/time_basis.py").is_file(), "Shared time basis is missing"
    return importlib.import_module("ai4ts.data_protocols.time_basis")


def source(tmp_path, *, regular=False, naive=False, rows=24):
    path = tmp_path / "source.csv"
    times = [
        pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(minutes=i if regular or i < 5 else i + 2)
        for i in range(rows)
    ]
    if naive:
        times = [stamp.tz_localize(None) for stamp in times]
    pd.DataFrame(dict(id=["001"] * rows, time=[str(t) for t in times], x=range(rows))).to_csv(path, index=False)
    return path


def choice(kind="observation_index", **extra):
    return dict(
        kind=kind,
        reason="User accepts the time representation and its consequences",
        impact_acknowledged=True,
        context=dict(task="imputation"),
        **extra,
    )


def plan(path, selection, **kwargs):
    return create_ingestion_plan(
        path,
        "imputation",
        roles=ROLES,
        split_policy="chronological",
        window_size=2,
        time_selection=selection,
        **kwargs,
    )


def test_preview_preserves_clock_rows_ids_and_unknown_sampling(tmp_path):
    path = source(tmp_path)
    before = path.read_bytes()
    value = api().preview_time_basis(path, ROLES)
    assert value["status"] == "ready", value
    assert value["entities"][0]["entity"] == "001"
    entity = value["entities"][0]
    assert entity["rows"] == list(range(24)) and entity["interval_seconds"] == dict(min=60.0, median=60.0, max=180.0)
    assert entity["missing_time_points"] is None and entity["regular_step_seconds"] is None
    assert value["finite_support"]["x"] == dict(finite=24, missing=0)
    assert value["grid"] is None and path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "case,status",
    [
        ("naive", "needs_clarification"),
        ("epoch", "blocked"),
        ("mixed", "blocked"),
        ("dst", "blocked"),
        ("roles", "needs_clarification"),
    ],
)
def test_no_clock_or_role_guessing(tmp_path, case, status):
    path = source(tmp_path, naive=case == "naive")
    if case == "epoch":
        path.write_text("id,time,x\n001,1704067200,1\n")
    if case == "mixed":
        path.write_text("id,time,x\n001,2024-01-01,1\n001,2024-01-02T00:00:00Z,2\n")
    if case == "dst":
        path.write_text("id,time,x\n001,2024-11-03T01:30:00,1\n")
    value = api().preview_time_basis(
        path,
        None if case == "roles" else ROLES,
        timezone="America/New_York" if case == "dst" else None,
    )
    assert value["status"] == status, value


def test_offsets_literal_ids_and_unsorted_facts(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text(
        "id,time,x\nNA,2024-01-01T08:01:00+08:00,0\nNA,2024-01-01T00:00:00Z,1\n001,2024-01-01T00:00:00Z,2\n"
    )
    value = api().preview_time_basis(path, ROLES)
    assert [g["entity"] for g in value["entities"]] == ["NA", "001"]
    assert value["entities"][0]["reversed_rows"] == [1]
    assert value["entities"][0]["times_ns"][0] - value["entities"][0]["times_ns"][1] == 60_000_000_000


def test_sparse_grid_counts_and_limits_without_producing(tmp_path):
    path = source(tmp_path)
    value = api().preview_time_basis(path, ROLES, frequency="1min", mode="regularize")
    assert value["status"] == "ready", value
    grid = value["grid"]
    assert grid["output_rows"] == 26 and grid["features"]["x"] == dict(finite=24, missing=2, longest_unsupported_run=2)
    fine = api().preview_time_basis(path, ROLES, frequency="1s", mode="regularize")
    assert fine["grid"]["output_rows"] == 1501 and fine["grid"]["features"]["x"]["finite"] == 24
    path.write_text("id,time,x\n001,2024-01-01T00:00:00Z,1\n001,2025-01-01T00:00:00Z,2\n")
    blocked = api().preview_time_basis(path, ROLES, frequency="1s", mode="regularize")
    assert blocked["status"] == "blocked" and "limits" in " ".join(blocked["errors"])
    assert not any(p.is_dir() for p in tmp_path.iterdir())


@pytest.mark.parametrize("reducer,finite", [("mean", 2), ("sum", 2), ("last", 1)])
def test_per_feature_support_follows_actual_reducer_including_missing_last(tmp_path, reducer, finite):
    path = tmp_path / "input.csv"
    path.write_text(
        "id,time,x,y\n001,2024-01-01T00:00:00Z,1,\n001,2024-01-01T00:00:30Z,,\n001,2024-01-01T00:01:30Z,3,\n"
    )
    roles = dict(ROLES, feature_cols=["x", "y"])
    value = api().preview_time_basis(
        path,
        roles,
        frequency="1min",
        mode="aggregate",
        reducers=dict(x=reducer, y=reducer),
    )
    assert value["status"] == "ready", value
    assert value["grid"]["features"]["x"]["finite"] == finite
    assert value["grid"]["features"]["y"] == dict(finite=0, missing=2, longest_unsupported_run=2)
    off = api().preview_time_basis(path, roles, frequency="1min", mode="regularize")
    assert off["status"] == "blocked" and "off" in " ".join(off["errors"]).lower()


def test_v2_observation_index_and_v1_keep_separate_semantics(tmp_path):
    path = source(tmp_path)
    api()
    v2 = plan(path, choice())
    assert v2["status"] == "ready" and v2["version"] == 2, v2
    assert v2["parameters"]["allow_irregular"] is True and v2["time_basis"] == "observation_index"
    assert (
        v2["time_selection"]["origin_source_sha256"] == v2["time_selection"]["input_sha256"] == v2["source"]["sha256"]
    )
    assert v2["time_selection"]["horizon_unit"] == "record_steps"
    validate_ingestion_plan(v2, path)
    assert plan(path, v2["time_selection"]) == v2
    renamed = tmp_path / "renamed.csv"
    renamed.write_bytes(path.read_bytes())
    assert plan(renamed, v2["time_selection"]) == v2
    legacy = create_ingestion_plan(
        path,
        "imputation",
        roles=ROLES,
        split_policy="chronological",
        window_size=2,
        allow_irregular=True,
    )
    assert legacy["version"] == 1 and "time_selection" not in legacy and "time_selection" not in legacy["request"]
    validate_ingestion_plan(legacy, path)
    for version in (0, 3, True):
        bad = copy.deepcopy(v2)
        bad["version"] = version
        with pytest.raises(ValueError):
            validate_ingestion_plan(bad, path)
    path.write_text(path.read_text().replace(",0\n", ",9\n", 1))
    assert plan(path, v2["time_selection"])["status"] == "blocked"
    with pytest.raises(ValueError):
        validate_ingestion_plan(v2, path)


def test_regular_grid_requires_existing_preparation_and_naive_requires_normalize(
    tmp_path,
):
    path = source(tmp_path)
    api()
    selected = choice("regular_grid", grid=dict(mode="regularize", frequency="1min", origin="epoch"))
    waiting = plan(path, selected)
    assert waiting["status"] == "needs_preparation" and waiting["next_step"]["operation"] == "temporal", waiting
    assert len(list(tmp_path.iterdir())) == 1
    assert plan(path, selected, allow_irregular=True)["status"] == "blocked"
    path = source(tmp_path, regular=True)
    ready = plan(path, selected)
    assert ready["status"] == "ready" and ready["parameters"]["allow_irregular"] is False, ready
    path = source(tmp_path, naive=True)
    pending = plan(path, choice(timezone="UTC"))
    assert pending["status"] == "needs_preparation" and pending["next_step"]["mode"] == "normalize", pending
    assert plan(path, dict(choice(), impact_acknowledged=False))["status"] == "needs_clarification"


def test_naive_grid_normalizes_and_regularizes_in_one_existing_transform(tmp_path):
    from ai4ts.data_protocols.temporal_preprocessing import (
        create_temporal_plan,
        run_temporal_preprocessing,
    )

    path = source(tmp_path, naive=True)
    api()
    waiting = plan(
        path,
        choice(
            "regular_grid",
            timezone="UTC",
            grid=dict(mode="regularize", frequency="1min", origin="epoch"),
        ),
    )
    assert waiting["status"] == "needs_preparation" and waiting["next_step"]["mode"] == "regularize", waiting
    temporal = create_temporal_plan(path, mode="regularize", timezone="UTC", roles=ROLES, frequency="1min")
    root = tmp_path / "grid"
    assert run_temporal_preprocessing(path, temporal, root)["status"] == "success"
    final = plan(root / "data.csv", waiting["time_selection"], temporal_result=root)
    assert final["status"] == "ready", final


def test_changed_roles_context_or_grid_cannot_reuse_selection(tmp_path):
    path = source(tmp_path)
    api()
    selection = plan(path, choice())["time_selection"]
    for updates in [
        dict(roles_sha256="0" * 64),
        dict(context=dict(task="forecasting")),
    ]:
        assert plan(path, dict(selection, **updates))["status"] == "blocked"
    selection = plan(
        path,
        choice(
            "regular_grid",
            grid=dict(mode="regularize", frequency="1min", origin="epoch"),
        ),
    )["time_selection"]
    mutated = copy.deepcopy(selection)
    mutated["grid"]["frequency"] = "2min"
    assert plan(path, mutated)["status"] == "blocked"


def test_verified_derived_selection_keeps_origin_and_input_separate(tmp_path):
    from ai4ts.data_protocols.temporal_preprocessing import (
        create_temporal_plan,
        run_temporal_preprocessing,
    )
    from ai4ts.data_protocols.pipeline import prepare_for_pypots

    path = source(tmp_path)
    api()
    waiting = plan(
        path,
        choice(
            "regular_grid",
            grid=dict(mode="regularize", frequency="1min", origin="epoch"),
        ),
    )
    temporal = create_temporal_plan(path, mode="regularize", roles=ROLES, frequency="1min")
    root = tmp_path / "prepared"
    result = run_temporal_preprocessing(path, temporal, root)
    assert result["status"] == "success", result
    final = plan(root / "data.csv", waiting["time_selection"], temporal_result=root)
    assert final["status"] == "ready", final
    selection = final["time_selection"]
    assert selection["origin_source_sha256"] == waiting["source"]["sha256"]
    assert selection["input_sha256"] == final["source"]["sha256"] != selection["origin_source_sha256"]
    validate_ingestion_plan(final, root / "data.csv")
    rows = final["split"]["rows"]["train"]
    raw = prepare_for_pypots(
        root / "data.csv",
        schema_mapping=ROLES,
        row_indices=rows,
        split_info=final["split"],
        window_size=2,
    )
    assert raw["scaler_state"] is None if "scaler_state" in raw else True
    assert set(sum([entry.row_indices for entry in raw["registry"].entries], [])) == set(rows)
    assert np.nanmax(raw["X"]) > 1
    bad = copy.deepcopy(final)
    bad["time_selection"]["input_sha256"] = "0" * 64
    bad["sha256"] = content_sha256({k: v for k, v in bad.items() if k != "sha256"})
    with pytest.raises(ValueError):
        validate_ingestion_plan(bad, root / "data.csv")


def test_inherited_split_survives_grid_and_bucket_conflict_is_rejected(tmp_path):
    from ai4ts.data_protocols.temporal_preprocessing import (
        create_temporal_plan,
        run_temporal_preprocessing,
    )

    path = source(tmp_path, rows=36)
    api()
    donor = create_ingestion_plan(
        path,
        "imputation",
        roles=ROLES,
        split_policy="chronological",
        window_size=2,
        allow_irregular=True,
    )
    waiting = plan(
        path,
        choice(
            "regular_grid",
            grid=dict(mode="regularize", frequency="1min", origin="epoch"),
        ),
    )
    prepared = create_temporal_plan(path, mode="regularize", roles=ROLES, frequency="1min", split_plan=donor)
    root = tmp_path / "grid"
    assert run_temporal_preprocessing(path, prepared, root)["status"] == "success"
    ready = plan(root / "data.csv", waiting["time_selection"], temporal_result=root)
    assert ready["status"] == "ready", ready
    inherited = json.loads((root / "split.json").read_text())["split"]
    assert ready["split"] == inherited
    assert len(ready["split"]["rows"]["train"]) == len(donor["split"]["rows"]["train"]) + 2
    conflict = create_temporal_plan(
        path,
        mode="aggregate",
        roles=ROLES,
        frequency="1h",
        reducers={"x": "mean"},
        split_plan=donor,
    )
    assert conflict["status"] == "blocked" and "boundary" in " ".join(conflict["errors"])
