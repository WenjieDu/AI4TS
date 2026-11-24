"""Long measurement rows retain their meaning through explicit reshaping."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import copy
import csv
import json

import pytest

import ai4ts.data_protocols as protocols

ROLES = dict(sample_id_col="id", timestamp_col="time", variable_col="metric", value_col="value")


def plan(path, **options):
    factory = getattr(protocols, "create_reshape_plan", None)
    assert callable(factory), "The public source-bound long-to-wide planner is missing"
    return factory(path, **(dict(roles=ROLES, variables=["b", "a"]) | options))


def source(tmp_path, rows, header="id,time,metric,value,note\n"):
    path = tmp_path / "long.csv"
    path.write_text(header + rows)
    return path


def test_ready_plan_preserves_selected_key_union_and_order_without_export(tmp_path):
    path = source(
        tmp_path,
        "001,2024-01-01T00:05:00Z,a,3,k\nNA,2024-01-01T00:00:00Z,b,0,k\n"
        "001,2024-01-01T08:00:00+08:00,a,,k\n001,2024-01-01T00:00:00Z,b,2,k\n"
        "001,2024-01-01T00:10:00Z,excluded,99,k\n",
    )
    value = plan(path)
    assert value["status"] == "ready", value
    assert value["wide_roles"] == dict(
        sample_id_col="id", timestamp_col="time", feature_cols=["b", "a"], label_col=None
    )
    assert value["preview"]["output_rows"] == 3
    assert value["preview"]["absent_measurements"] == 2
    assert value["preview"]["recorded_missing"] == 1
    assert value["preview"]["excluded_variable_records"] == 1
    assert value["preview"]["split_status"] == "unsplit"
    assert list(tmp_path.iterdir()) == [path]
    protocols.validate_reshape_plan(value, path)
    path.write_text(path.read_text().replace(",3,k", ",4,k"))
    with pytest.raises(ValueError):
        protocols.validate_reshape_plan(value, path)


def test_missing_declarations_reuse_and_wrong_source(tmp_path):
    path = source(tmp_path, "001,2024-01-01T00:00:00Z,a,1,k\n001,2024-01-01T00:00:00Z,b,2,k\n")
    factory = getattr(protocols, "create_reshape_plan", None)
    assert callable(factory), "The public source-bound long-to-wide planner is missing"
    assert factory(path)["status"] == "needs_clarification"
    prior = plan(path)
    reused = factory(path, confirmed_plan=prior)
    assert reused["status"] == "ready" and reused["wide_roles"] == prior["wide_roles"]
    path.write_text(path.read_text().replace(",1,k", ",8,k"))
    assert factory(path, confirmed_plan=prior)["status"] == "blocked"


def test_timezone_clarification_keeps_confirmed_roles_and_source(tmp_path):
    path = source(tmp_path, "001,2024-01-01 00:00:00,a,1,k\n001,2024-01-01 00:00:00,b,2,k\n")
    pending = plan(path)
    assert pending["status"] == "needs_clarification"
    assert pending["source"]["sha256"] and pending["request"]["variables"] == ["b", "a"]
    ready = protocols.create_reshape_plan(path, confirmed_plan=pending, timezone="UTC")
    assert ready["status"] == "ready", ready
    protocols.validate_reshape_plan(ready, path)
    path.write_text(path.read_text().replace(",1,k", ",8,k"))
    assert protocols.create_reshape_plan(path, confirmed_plan=pending, timezone="UTC")["status"] == "blocked"


def test_plan_file_budget_rejects_unreadable_ready_declarations(tmp_path, monkeypatch):
    from ai4ts.data_protocols import reshape

    monkeypatch.setattr(reshape, "MAX_PLAN_BYTES", 2000, raising=False)
    path = source(tmp_path, "001,2024-01-01T00:00:00Z,a,1,k\n001,2024-01-01T00:00:00Z,b,2,k\n")
    value = plan(path)
    assert value["status"] == "blocked" and "declaration byte limit" in value["errors"][0]
    assert len((json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()) <= 2000


@pytest.mark.parametrize(
    "stamp,zone,status",
    [
        ("2024-01-01 00:00:00", None, "needs_clarification"),
        ("2024-01-01 00:00:00", "Asia/Singapore", "ready"),
        ("2024-11-03 01:30:00", "America/New_York", "blocked"),
        ("2024-03-10 02:30:00", "America/New_York", "blocked"),
        ("1704067200", "UTC", "blocked"),
        ("bad", "UTC", "blocked"),
    ],
)
def test_physical_clock_requires_an_interpretation(tmp_path, stamp, zone, status):
    path = source(tmp_path, f"001,{stamp},a,1,k\n001,{stamp},b,2,k\n")
    value = plan(path, timezone=zone)
    assert value["status"] == status, value
    if zone == "Asia/Singapore":
        assert value["rows"][0]["time_ns"] == 1704038400000000000


def test_duplicate_default_and_explicit_last_keep_missing_record(tmp_path):
    path = source(tmp_path, "001,2024-01-01T00:00:00Z,a,1,first\n" "001,2024-01-01T08:00:00+08:00,a,,last\n")
    assert plan(path, variables=["a"])["status"] == "blocked"
    last = plan(path, variables=["a"], duplicate_policy="keep-last")
    assert last["status"] == "ready", last
    assert last["preview"]["recorded_missing"] == 1 and last["preview"]["unselected_duplicates"] == 1
    first = plan(path, variables=["a"], duplicate_policy="keep-first")
    assert first["status"] == "ready" and first["preview"]["recorded_missing"] == 0


@pytest.mark.parametrize(
    "change",
    [
        dict(roles=dict(ROLES, value_col="id")),
        dict(variables=["time"]),
        dict(variables=["a", "a"]),
        dict(variables=["not-present"]),
        dict(duplicate_policy="mean"),
        dict(drop_columns=["metric"]),
    ],
)
def test_invalid_roles_names_and_projection_block(tmp_path, change):
    path = source(tmp_path, "001,2024-01-01T00:00:00Z,a,1,k\n001,2024-01-01T00:00:00Z,b,2,k\n")
    assert plan(path, **change)["status"] == "blocked"


@pytest.mark.parametrize("reading", ["no-value", "inf", "-inf", "NaN"])
def test_nonempty_invalid_measurements_are_not_silently_missing(tmp_path, reading):
    path = source(tmp_path, f"001,2024-01-01T00:00:00Z,a,{reading},k\n")
    assert plan(path, variables=["a"])["status"] == "blocked"


def test_metadata_needs_explicit_drop_and_labels_never_collapse(tmp_path):
    path = source(tmp_path, "001,2024-01-01T00:00:00Z,a,1,left\n001,2024-01-01T00:00:00Z,b,2,right\n")
    assert plan(path)["status"] == "blocked"
    assert plan(path, drop_columns=["note"])["status"] == "ready"
    assert plan(path, roles=dict(ROLES, label_col="note"), drop_columns=["note"])["status"] == "blocked"


def test_unit_evidence_conflicts_and_unknown_rows_need_confirmation(tmp_path):
    path = source(
        tmp_path,
        "001,2024-01-01T00:00:00Z,a,1,C\n001,2024-01-01T00:05:00Z,a,2,C\n",
        header="id,time,metric,value,unit\n",
    )
    options = dict(roles=dict(ROLES, unit_col="unit"), variables=["a"])
    value = plan(path, **options)
    assert value["status"] == "ready" and value["feature_semantics"]["units"] == {"a": "C"}
    path.write_text(path.read_text().replace(",2,C", ",2,"))
    assert plan(path, **options)["status"] == "needs_clarification"
    assert plan(path, feature_units={"a": "C"}, **options)["status"] == "ready"
    path.write_text(path.read_text().replace(",2,\n", ",2,K\n"))
    assert plan(path, feature_units={"a": "C"}, **options)["status"] == "blocked"


def test_width_expansion_is_rejected_without_export(tmp_path):
    path = tmp_path / "long.csv"
    with path.open("w") as stream:
        stream.write("id,time,metric,value\n")
        for i in range(1001):
            stream.write(f"001,2024-01-01T00:{i//60:02d}:{i%60:02d}Z,v{i},1\n")
    value = plan(path, variables=[f"v{i}" for i in range(1001)])
    assert value["status"] == "blocked" and "limit" in str(value["errors"]).lower()
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("large_field", ["header", "metadata"])
def test_complete_csv_estimate_includes_large_utf8_headers_and_metadata(tmp_path, monkeypatch, large_field):
    from ai4ts.data_protocols import reshape

    variable, note = ("温" * 2000, "k") if large_field == "header" else ("a", "注" * 2000)
    path = source(tmp_path, f"001,2024-01-01T00:00:00Z,{variable},1,{note}\n")
    monkeypatch.setattr(reshape, "MAX_CSV_BYTES", 4096)
    value = plan(path, variables=[variable])
    assert value["status"] == "blocked" and "byte" in str(value["errors"]).lower()
    assert list(tmp_path.iterdir()) == [path]


def execute(tmp_path, path, **options):
    value = plan(path, **options)
    assert value["status"] == "ready", value
    run = getattr(protocols, "run_reshape", None)
    assert callable(run), "The bounded long-to-wide executor is missing"
    root = tmp_path / "result"
    outcome = run(path, value, root)
    assert outcome["status"] == "success", outcome
    verified = protocols.inspect_reshape_result(root)
    assert verified["status"] == "success", verified
    with (root / "data.csv").open() as stream:
        records = list(csv.DictReader(stream))
    return value, root, records, verified


def test_export_retains_physical_keys_cells_missing_reasons_and_source(tmp_path):
    path = source(
        tmp_path,
        "001,2024-01-01T00:00:00.123456789Z,a,,keep\n"
        "001,2024-01-01T00:00:03.123456789Z,b,0,keep\n"
        "NA,2024-01-01T00:00:00Z,a,7,keep\n",
    )
    original = path.read_bytes()
    value, root, records, verified = execute(tmp_path, path)
    assert [(r["id"], r["a"], r["b"]) for r in records] == [("001", "", ""), ("001", "", "0"), ("NA", "7", "")]
    assert records[0]["time"] == "2024-01-01T00:00:00.123456789Z"
    lines = [json.loads(s) for s in (root / "lineage.jsonl").read_text().splitlines()]
    assert lines[0]["cells"]["a"] == dict(
        source_rows=[0], selected_row=0, finite_count=0, unknown_reason="recorded-missing"
    )
    assert lines[0]["cells"]["b"] == dict(
        source_rows=[], selected_row=None, finite_count=0, unknown_reason="absent-measurement"
    )
    assert verified["split"] is None and verified["split_status"] == "unsplit"
    assert original == path.read_bytes() == (root / "source.csv").read_bytes()
    assert verified["summary"]["output_missing"] == 4
    assert (root / "supervisor.json").is_file() and not (root / "pending-result.json").exists()
    assert (root / "result.json").stat().st_mtime_ns >= (root / "supervisor.json").stat().st_mtime_ns
    with pytest.raises(FileExistsError):
        protocols.run_reshape(path, value, root)


def test_single_stream_without_entity_preserves_quoted_variable_and_precision(tmp_path):
    path = source(tmp_path, '2024-01-01T00:00:00.123456789Z,"temperature,C",1\n', header="time,metric,value\n")
    roles = dict(timestamp_col="time", variable_col="metric", value_col="value")
    _, _, records, verified = execute(tmp_path, path, roles=roles, variables=["temperature,C"])
    assert list(records[0]) == ["time", "temperature,C"] and records[0]["temperature,C"] == "1"
    assert verified["roles"]["sample_id_col"] is None


def test_mixed_naive_and_aware_clock_is_not_inferred(tmp_path):
    path = source(tmp_path, "001,2024-01-01T00:00:00Z,a,1,k\n001,2024-01-01 00:00:00,b,2,k\n")
    value = plan(path, timezone="UTC")
    assert value["status"] == "blocked" and "Mixed" in str(value["errors"])


def test_export_selected_duplicate_units_and_metadata_are_whole_records(tmp_path):
    path = source(
        tmp_path,
        "001,2024-01-01T00:00:00Z,a,9,C,first\n" "001,2024-01-01T00:00:00Z,a,,C,last\n",
        header="id,time,metric,value,unit,note\n",
    )
    _, root, records, verified = execute(
        tmp_path, path, variables=["a"], roles=dict(ROLES, unit_col="unit"), duplicate_policy="keep-last"
    )
    assert records[0]["a"] == "" and records[0]["note"] == "last"
    assert verified["feature_semantics"]["units"] == {"a": "C"}
    disposition = json.loads((root / "row-disposition.json").read_text())
    assert disposition["discarded"] == [dict(row=0, retained=1, reason="unselected-duplicate")]


@pytest.mark.parametrize("fault", ["value", "support", "units", "mapping"])
def test_rehashed_tampering_still_fails_independent_inspection(tmp_path, fault):
    path = source(tmp_path, "001,2024-01-01T00:00:00Z,a,1,k\n001,2024-01-01T00:00:00Z,b,2,k\n")
    _, root, _, _ = execute(tmp_path, path, feature_units={"a": "C"})
    saved = json.loads((root / "result.json").read_text())
    from ai4ts.data_protocols.registry import file_sha256

    if fault == "value":
        target = root / "data.csv"
        target.write_text(target.read_text().replace(",2,1,k", ",2,8,k"))
        saved["derived_sha256"] = file_sha256(target)
    elif fault == "units":
        target = root / "feature-semantics.json"
        item = json.loads(target.read_text())
        item["units"]["a"] = "K"
        target.write_text(json.dumps(item))
    else:
        target = root / "lineage.jsonl"
        item = json.loads(target.read_text().splitlines()[0])
        item["cells"]["a"]["finite_count" if fault == "support" else "selected_row"] = 99
        target.write_text(json.dumps(item) + "\n")
    saved["files"][target.name] = file_sha256(target)
    (root / "result.json").write_text(json.dumps(saved))
    assert protocols.inspect_reshape_result(root)["status"] == "failed"


def test_inspect_never_calls_producer_or_changes_artifacts(tmp_path, monkeypatch):
    path = source(tmp_path, "001,2024-01-01T00:00:00Z,a,1,k\n")
    _, root, _, _ = execute(tmp_path, path, variables=["a"])
    from ai4ts.data_protocols import reshape_artifacts

    before = {p.name: p.read_bytes() for p in root.iterdir()}
    monkeypatch.setattr(reshape_artifacts, "_produce", lambda *args: pytest.fail("inspect invoked producer"))
    assert reshape_artifacts._verify(root)["status"] == "success"
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


def test_partial_write_failure_symlinks_and_timeout_leave_no_success(tmp_path, monkeypatch):
    path = source(tmp_path, "001,2024-01-01T00:00:00Z,a,1,k\n")
    value, root, _, _ = execute(tmp_path, path, variables=["a"])
    from ai4ts.data_protocols import reshape_artifacts as artifacts

    original = artifacts.write_json

    def fail_write(path, data):
        if path.name == "execution.json":
            raise OSError("controlled write failure")
        return original(path, data)

    monkeypatch.setattr(artifacts, "write_json", fail_write)
    partial = tmp_path / "partial"
    partial.mkdir()
    result = artifacts._work(dict(operation="run", source=str(path), plan=value, output=str(partial)))
    assert result["status"] == "failed" and (partial / "data.csv").is_file()
    assert not (partial / "result.json").exists()
    monkeypatch.undo()
    link = tmp_path / "link.csv"
    link.symlink_to(path)
    assert plan(link, variables=["a"])["status"] == "blocked"
    result = protocols.run_reshape(path, value, tmp_path / "timeout", timeout=0.001)
    assert result["status"] == "failed" and result["timed_out"]
    assert (tmp_path / "timeout/failure.json").is_file() and not (tmp_path / "timeout/result.json").exists()
    (root / "lineage.jsonl").rename(root / "original.jsonl")
    (root / "lineage.jsonl").symlink_to(root / "original.jsonl")
    assert protocols.inspect_reshape_result(root)["status"] == "failed"
