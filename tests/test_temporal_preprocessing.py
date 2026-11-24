"""Direct physical-time and data-lineage contracts; no model execution."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import numpy as np
import pandas as pd
import pytest

from ai4ts.data_protocols import (
    create_temporal_plan,
    validate_temporal_plan,
    run_temporal_preprocessing,
    inspect_temporal_result,
)
from ai4ts.data_protocols.registry import file_sha256
from ai4ts.data_protocols.analysis import create_analysis_plan

ROLES = dict(sample_id_col="id", timestamp_col="time", feature_cols=["a", "b"], label_col=None)


def source(tmp_path, rows):
    path = tmp_path / "source.csv"
    path.write_text("id,time,a,b,note\n" + rows)
    return path


def execute(tmp_path, path, **options):
    plan = create_temporal_plan(path, roles=ROLES, **options)
    assert plan["status"] == "ready", plan
    root = tmp_path / "result"
    outcome = run_temporal_preprocessing(path, plan, root)
    assert outcome["status"] == "success", outcome
    verified = inspect_temporal_result(root)
    assert verified["status"] == "success", verified
    return plan, root, pd.read_csv(root / "data.csv", dtype={"id": str}, keep_default_na=False), verified


def test_plan_is_source_bound_and_does_not_produce_data(tmp_path):
    path = source(tmp_path, "001,2024-01-01T00:00:00Z,1,2,keep\n")
    pending = create_temporal_plan(path)
    assert pending["status"] == "needs_clarification"
    plan = create_temporal_plan(path, mode="normalize", roles=ROLES)
    assert plan["status"] == "ready", plan
    assert plan["groups"][0]["entity"] == "001"
    assert not (tmp_path / "data.csv").exists()
    validate_temporal_plan(plan, path)
    corrupt = copy.deepcopy(plan)
    corrupt["request"]["duplicate_policy"] = "keep-first"
    with pytest.raises(ValueError):
        validate_temporal_plan(corrupt, path)
    path.write_text(path.read_text().replace(",1,2,", ",3,2,"))
    with pytest.raises(ValueError):
        validate_temporal_plan(plan, path)


def test_confirmed_roles_wrong_source_and_invalid_roles(tmp_path):
    path = source(tmp_path, "001,2024-01-01T00:00:00Z,1,2,keep\n")
    prior = create_analysis_plan(path, roles=ROLES)
    plan = create_temporal_plan(path, mode="normalize", confirmed_plan=prior)
    assert plan["status"] == "ready" and plan["roles"] == ROLES
    roles = dict(ROLES, feature_cols=["id", "a"])
    assert create_temporal_plan(path, mode="normalize", roles=roles)["status"] == "blocked"
    assert create_temporal_plan(path, mode="regularize", roles=ROLES)["status"] == "needs_clarification"
    path.write_text(path.read_text().replace(",1,2,", ",9,2,"))
    rejected = create_temporal_plan(path, mode="normalize", confirmed_plan=prior)
    assert rejected["status"] == "blocked" and "source" in rejected["errors"][0]


@pytest.mark.parametrize(
    "stamp,timezone,status",
    [
        ("2024-01-01 00:00:00", None, "needs_clarification"),
        ("2024-01-01 00:00:00", "Asia/Singapore", "ready"),
        ("2024-11-03 01:30:00", "America/New_York", "blocked"),
        ("2024-03-10 02:30:00", "America/New_York", "blocked"),
        ("1704067200", "UTC", "blocked"),
        ("invalid", "UTC", "blocked"),
    ],
)
def test_clock_interpretation(tmp_path, stamp, timezone, status):
    path = source(tmp_path, f"001,{stamp},1,2,keep\n")
    plan = create_temporal_plan(path, mode="normalize", roles=ROLES, timezone=timezone)
    assert plan["status"] == status, plan
    if status == "ready":
        assert plan["times_ns"] == [pd.Timestamp("2023-12-31T16:00:00Z").value]


def test_precision_offsets_duplicates_and_whole_record_selection(tmp_path):
    path = source(
        tmp_path,
        "001,2024-01-01T00:05:00.123456789Z,4,5,later\n"
        "001,2024-01-01T08:00:00+08:00,1,,first\n"
        "001,2024-01-01T00:00:00Z,,2,last\n",
    )
    blocked = create_temporal_plan(path, mode="normalize", roles=ROLES)
    assert blocked["status"] == "blocked" and "Duplicate" in blocked["errors"][0]
    original = path.read_bytes()
    plan, root, data, _ = execute(tmp_path, path, mode="normalize", duplicate_policy="keep-last")
    assert data.a.tolist() == ["", "4"] and data.b.tolist() == [2, 5]
    assert data.note.tolist() == ["last", "later"]
    assert data.time.tolist()[-1].endswith(".123456789Z")
    assert plan["groups"][0]["discarded"] == [dict(row=1, retained=2, reason="unselected-duplicate")]
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError):
        run_temporal_preprocessing(path, plan, root)


def test_grid_preserves_original_and_new_unknowns_with_shared_phase(tmp_path):
    path = source(
        tmp_path,
        "001,2024-01-01T00:00:00Z,1,,first\n"
        "001,2024-01-01T00:10:00Z,3,4,last\n"
        "NA,2024-01-01T00:05:00Z,5,6,constant\n"
        "NA,2024-01-01T00:15:00Z,7,8,constant\n",
    )
    _, root, data, verified = execute(tmp_path, path, mode="regularize", frequency="5min")
    assert data.id.tolist() == ["001"] * 3 + ["NA"] * 3
    assert data.a.tolist() == ["1", "", "3", "5", "", "7"]
    assert data.note.tolist() == ["first", "", "last", "constant", "constant", "constant"]
    assert verified["summary"]["original_missing_cells"] == 1
    assert verified["summary"]["grid_added_cells"] == 4
    lineage = [json.loads(line) for line in (root / "lineage.jsonl").read_text().splitlines()]
    assert lineage[1]["metadata_unknown"] == ["note"]
    assert lineage[1]["source_rows"] == []


@pytest.mark.parametrize(
    "option,value",
    [
        ("frequency", "1M"),
        ("frequency", "0s"),
        ("frequency", "1.5s"),
        ("origin", "2024-01-01"),
        ("duplicate_policy", "average"),
    ],
)
def test_unsupported_semantics_are_blocked(tmp_path, option, value):
    path = source(tmp_path, "001,2024-01-01T00:00:00Z,1,2,keep\n")
    options = dict(mode="regularize", frequency="5min", roles=ROLES)
    options[option] = value
    assert create_temporal_plan(path, **options)["status"] == "blocked"


def test_off_grid_mixed_clock_and_expansion_block_before_output(tmp_path):
    path = source(tmp_path, "001,2024-01-01T00:00:00Z,1,2,keep\n001,2024-01-01T00:05:03Z,2,3,keep\n")
    assert "off" in create_temporal_plan(path, mode="regularize", roles=ROLES, frequency="5min")["errors"][0]
    path.write_text(path.read_text().replace("00:05:03Z", "00:05:03"))
    assert "Mixed" in create_temporal_plan(path, mode="normalize", roles=ROLES)["errors"][0]
    path.write_text(path.read_text().replace("2024-01-01T00:05:03", "2034-01-01T00:05:03Z"))
    blocked = create_temporal_plan(path, mode="regularize", roles=ROLES, frequency="1s")
    assert blocked["status"] == "blocked" and "limit" in blocked["errors"][0]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["source.csv"]


def test_aggregate_direct_values_unknowns_units_and_boundary(tmp_path):
    path = source(
        tmp_path,
        "001,2024-01-01T00:00:00Z,1,7,keep\n"
        "001,2024-01-01T00:30:00Z,3,,keep\n"
        "001,2024-01-01T01:00:00Z,-2,0,keep\n"
        "001,2024-01-01T01:30:00Z,2,0,keep\n"
        "001,2024-01-01T03:00:00Z,,,keep\n",
    )
    plan, root, data, checked = execute(
        tmp_path, path, mode="aggregate", frequency="1h", reducers={"a": "sum", "b": "last"}, feature_units={"a": "kWh"}
    )
    assert data.a.tolist() == ["4", "0", "", ""]
    assert data.b.tolist() == ["", "0", "", ""]
    assert data.time.tolist()[0] == "2024-01-01T01:00:00Z"
    assert checked["summary"]["empty_buckets"] == 1
    assert checked["summary"]["last_missing_cells"] == 1
    assert checked["feature_semantics"]["units"] == {"a": "kWh", "b": "unknown"}
    assert checked["feature_semantics"]["source_sha256"] == file_sha256(root / "data.csv")
    assert checked["feature_semantics"]["declaration"]["source_sha256"] == plan["source"]["sha256"]
    lineage = [json.loads(line) for line in (root / "lineage.jsonl").read_text().splitlines()]
    assert lineage[0]["source_rows"] == [0, 1] and lineage[1]["source_rows"] == [2, 3]
    assert lineage[0]["cells"]["b"]["selected_row"] == 1


def test_aggregate_metadata_projection_is_explicit(tmp_path):
    path = source(tmp_path, "001,2024-01-01T00:00:00Z,1,2,a\n001,2024-01-01T00:30:00Z,3,4,b\n")
    options = dict(mode="aggregate", roles=ROLES, frequency="1h", reducers={"a": "mean", "b": "mean"})
    assert "metadata" in create_temporal_plan(path, **options)["errors"][0]
    plan = create_temporal_plan(path, drop_columns=["note"], **options)
    assert plan["status"] == "ready"
    result = run_temporal_preprocessing(path, plan, tmp_path / "result")
    assert result["status"] == "success", result
    data = pd.read_csv(tmp_path / "result/data.csv")
    assert data.a.tolist() == [2] and data.b.tolist() == [3] and "note" not in data
    assert inspect_temporal_result(tmp_path / "result")["status"] == "success"


def test_aggregate_retains_small_residuals_between_large_values(tmp_path):
    path = source(
        tmp_path,
        "001,2024-01-01T00:00:00Z,1e16,1e16,keep\n"
        "001,2024-01-01T00:10:00Z,1,1,keep\n"
        "001,2024-01-01T00:20:00Z,-1e16,-1e16,keep\n",
    )
    _, _, data, _ = execute(tmp_path, path, mode="aggregate", frequency="1h", reducers={"a": "sum", "b": "mean"})
    assert data.a.tolist() == [1] and data.b.tolist() == [pytest.approx(1 / 3)]


@pytest.mark.parametrize("mutation", ["denominator", "value", "membership"])
def test_independent_reader_rejects_rehashed_corruption(tmp_path, mutation):
    path = source(tmp_path, "001,2024-01-01T00:00:00Z,1,2,keep\n001,2024-01-01T00:30:00Z,3,4,keep\n")
    _, root, _, _ = execute(tmp_path, path, mode="aggregate", frequency="1h", reducers={"a": "mean", "b": "sum"})
    manifest = json.loads((root / "result.json").read_text())
    if mutation == "value":
        data = pd.read_csv(root / "data.csv", dtype=str)
        data.loc[0, "a"] = "99"
        data.to_csv(root / "data.csv", index=False)
        manifest["derived_sha256"] = file_sha256(root / "data.csv")
        manifest["files"]["data.csv"] = manifest["derived_sha256"]
    else:
        lines = [json.loads(line) for line in (root / "lineage.jsonl").read_text().splitlines()]
        if mutation == "denominator":
            lines[0]["cells"]["a"]["finite_count"] = 99
        else:
            lines[0]["source_rows"] = [1]
        (root / "lineage.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")
        manifest["files"]["lineage.jsonl"] = file_sha256(root / "lineage.jsonl")
    (root / "result.json").write_text(json.dumps(manifest))
    assert inspect_temporal_result(root)["status"] == "failed"


def test_inherited_rows_survive_grid_expansion_and_training_prepare(tmp_path):
    from ai4ts.data_protocols.ingestion import create_ingestion_plan, validate_ingestion_plan
    from ai4ts.data_protocols.pipeline import prepare_for_pypots, split_source_rows

    origin = pd.Timestamp("2024-01-01T00:00:00Z")
    path = source(
        tmp_path,
        "".join(
            f"001,{(origin + pd.Timedelta(minutes=5*i + (60 if i >= 20 else 0))).isoformat()},{i},{i*10},keep\n"
            for i in range(50)
        ),
    )
    old = create_ingestion_plan(
        path, "imputation", roles=ROLES, split_policy="entity_chronological", window_size=3, allow_irregular=True
    )
    assert old["status"] == "ready", old
    plan, root, _, _ = execute(
        tmp_path, path, mode="regularize", frequency="5min", split_plan=old, feature_units={"a": "C"}
    )
    assert list(map(len, plan["split"]["rows"].values())) == [47, 5, 10]
    data = root / "data.csv"
    fresh = split_source_rows(data, ROLES, policy="entity_chronological")
    assert fresh["rows"] != plan["split"]["rows"]
    prepared_plan = create_ingestion_plan(data, "imputation", temporal_result=root, window_size=3)
    assert prepared_plan["status"] == "ready", prepared_plan
    assert prepared_plan["split"] == plan["split"]
    validate_ingestion_plan(prepared_plan, data)
    training = prepare_for_pypots(data, window_size=3, ingestion_plan=prepared_plan, preprocessing="standard-v1")
    assert training["scaler_state"]["mean"] == [17.0, 170.0]
    assert training["feature_semantics"]["units"] == {"a": "C", "b": "unknown"}
    actual_rows = [r for e in training["registry"].entries for r in e.row_indices]
    assert actual_rows == plan["split"]["rows"]["train"]
    analysis = create_analysis_plan(data, temporal_result=root, scope="train", methods=[])
    assert analysis["status"] == "ready", analysis
    assert analysis["selected_rows"] == plan["split"]["rows"]["train"]
    assert analysis["feature_semantics"]["units"]["a"] == "C"
    subset = create_analysis_plan(data, temporal_result=root, roles=dict(ROLES, feature_cols=["b"]), scope="test")
    assert subset["status"] == "ready" and subset["selected_rows"] == plan["split"]["rows"]["test"]
    conflict = create_ingestion_plan(data, "imputation", temporal_result=root, split_policy="entity", window_size=3)
    assert conflict["status"] == "blocked" and "override" in conflict["errors"][0]


def test_aggregate_preserves_train_bucket_right_label_and_rejects_crossing(tmp_path):
    from ai4ts.data_protocols.ingestion import create_ingestion_plan

    origin = pd.Timestamp("2024-01-01T00:00:00Z")
    path = source(
        tmp_path, "".join(f"001,{(origin + pd.Timedelta(minutes=5*i)).isoformat()},{i},{i},keep\n" for i in range(40))
    )
    old = create_ingestion_plan(path, "imputation", roles=ROLES, split_policy="entity_chronological", window_size=2)
    assert old["status"] == "ready"
    plan = create_temporal_plan(
        path, mode="aggregate", roles=ROLES, frequency="10min", split_plan=old, reducers={"a": "mean", "b": "sum"}
    )
    assert plan["status"] == "ready", plan
    root = tmp_path / "result"
    assert run_temporal_preprocessing(path, plan, root)["status"] == "success"
    assert inspect_temporal_result(root)["status"] == "success"
    rows = [json.loads(line) for line in (root / "lineage.jsonl").read_text().splitlines()]
    assert rows[13]["interval_end"] == (origin + pd.Timedelta(minutes=140)).isoformat().replace("+00:00", "Z")
    assert rows[13]["split"] == "train" and rows[13]["source_rows"] == [26, 27]
    assert rows[14]["split"] == "val" and rows[14]["source_rows"] == [28, 29]
    crossing = create_temporal_plan(
        path, mode="aggregate", roles=ROLES, frequency="15min", split_plan=old, reducers={"a": "mean", "b": "sum"}
    )
    assert crossing["status"] == "blocked" and "boundary" in crossing["errors"][0]


def test_write_failure_source_links_and_readonly_inspection(tmp_path, monkeypatch):
    from ai4ts.data_protocols import temporal_artifacts as artifacts

    path = source(tmp_path, "001,2024-01-01T00:00:00Z,1,2,keep\n")
    plan, root, _, _ = execute(tmp_path, path, mode="normalize")
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    monkeypatch.setattr(artifacts, "_produce", lambda *args: pytest.fail("inspect called producer"))
    assert artifacts._verify(root)["status"] == "success"
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    monkeypatch.undo()
    failed = tmp_path / "failed"
    failed.mkdir()
    original = artifacts.write_json

    def fail_execution(path, value):
        if path.name == "execution.json":
            raise OSError("controlled disk write failure")
        return original(path, value)

    monkeypatch.setattr(artifacts, "write_json", fail_execution)
    outcome = artifacts._work(dict(operation="run", source=str(path), plan=plan, output=str(failed)))
    assert outcome["status"] == "failed" and (failed / "data.csv").is_file()
    assert not (failed / "result.json").exists()
    monkeypatch.undo()
    link = tmp_path / "linked.csv"
    link.symlink_to(path)
    assert create_temporal_plan(link, mode="normalize", roles=ROLES)["status"] == "blocked"
    (root / "lineage.jsonl").rename(root / "original-lineage.jsonl")
    (root / "lineage.jsonl").symlink_to(root / "original-lineage.jsonl")
    assert inspect_temporal_result(root)["status"] == "failed"


def test_timeout_has_owned_failure_evidence_and_no_retry(tmp_path, monkeypatch):
    from ai4ts.data_protocols import temporal_artifacts as artifacts
    import subprocess

    path = source(tmp_path, "001,2024-01-01T00:00:00Z,1,2,keep\n")
    plan = create_temporal_plan(path, mode="normalize", roles=ROLES)
    calls = []

    def timeout(*args, **kwargs):
        calls.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(artifacts.subprocess, "run", timeout)
    outcome = run_temporal_preprocessing(path, plan, tmp_path / "timeout", timeout=0.01)
    assert outcome["status"] == "failed" and outcome["timed_out"]
    assert calls == [0.01]
    assert (tmp_path / "timeout/failure.json").is_file()
    assert not (tmp_path / "timeout/result.json").exists()
    assert inspect_temporal_result(tmp_path / "timeout", timeout=0.01)["status"] == "failed"


def test_entity_isolation_and_confirmation_survive_physical_grid(tmp_path):
    from ai4ts.data_protocols.ingestion import create_ingestion_plan

    path = source(
        tmp_path,
        "".join(
            f"{entity},2024-01-01 00:{minute:02}:00,{minute},2,keep\n"
            for entity in ("001", "NA", "C", "D", "E")
            for minute in (0, 10, 20)
        ),
    )
    prior = create_temporal_plan(path, mode="normalize", roles=ROLES, timezone="UTC")
    reused = create_temporal_plan(path, mode="regularize", frequency="5min", confirmed_plan=prior)
    assert reused["status"] == "ready" and reused["request"]["timezone"] == "UTC"
    donor = create_ingestion_plan(path, "imputation", roles=ROLES, split_policy="entity", window_size=2)
    assert donor["status"] == "ready", donor
    plan = create_temporal_plan(
        path, mode="regularize", frequency="5min", roles=ROLES, timezone="UTC", split_plan=donor
    )
    assert plan["status"] == "ready", plan
    root = tmp_path / "result"
    assert run_temporal_preprocessing(path, plan, root)["status"] == "success"
    checked = inspect_temporal_result(root)
    assert checked["status"] == "success", checked
    data = pd.read_csv(root / "data.csv", dtype=str, keep_default_na=False)
    original = pd.read_csv(path, dtype=str, keep_default_na=False)
    for role, derived_rows in plan["split"]["rows"].items():
        assert set(data.loc[derived_rows, "id"]) == set(original.loc[donor["split"]["rows"][role], "id"])
