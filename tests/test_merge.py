"""Exact multi-source data contracts; no estimator/model calls."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest


def api():
    assert (Path(__file__).parents[1] / "ai4ts/data_protocols/merge.py").is_file(), "Merge capability missing"
    return importlib.import_module("ai4ts.data_protocols.merge")


def sources(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text(
        "id,time,x,label,note\n001,2024-01-01T00:00:00Z,0,A,first\n001,2024-01-01T00:02:00Z,,A,last\nNA,2024-01-01T00:00:00Z,2,B,z\n"
    )
    b.write_text(
        "device,clock,y\n001,2024-01-01T08:00:00+08:00,3\n001,2024-01-01T00:01:00.000001Z,\nNA,2024-01-01T00:00:00Z,4\n"
    )
    return [
        dict(
            source_id="a",
            path=str(a),
            roles=dict(sample_id_col="id", timestamp_col="time", feature_cols=["x"], label_col="label"),
            retain_columns=["label"],
            drop_columns=["note"],
        ),
        dict(
            source_id="b",
            path=str(b),
            roles=dict(sample_id_col="device", timestamp_col="clock", feature_cols=["y"], label_col=None),
            retain_columns=[],
            drop_columns=[],
        ),
    ]


def plan(inputs, **kwargs):
    return api().create_merge_plan(inputs, join="left", entity_meaning="Same confirmed device identifiers", **kwargs)


def test_exact_keys_literal_ids_order_and_preview_only(tmp_path):
    value = plan(sources(tmp_path))
    assert value["status"] == "ready", value
    assert value["roles"]["feature_cols"] == ["x", "y"]
    assert [(row["entity"], row["source_rows"]) for row in value["rows"]] == [
        ("001", {"a": 0, "b": 0}),
        ("001", {"a": 1, "b": None}),
        ("NA", {"a": 2, "b": 2}),
    ]
    assert value["preview"]["sources"]["b"]["excluded_rows"] == [1]
    assert value["preview"]["features"]["x"] == dict(finite=2, original_missing=1, unmatched=0)
    assert not (tmp_path / "data.csv").exists()


@pytest.mark.parametrize(
    "case", ["duplicate", "feature", "mixed", "label", "units", "metadata", "unsafe", "nested", "preserve"]
)
def test_contract_conflicts_before_production(tmp_path, case):
    inputs = sources(tmp_path)
    extra = {}
    if case == "duplicate":
        p = Path(inputs[1]["path"])
        p.write_text(p.read_text() + p.read_text().splitlines()[1] + "\n")
    if case == "feature":
        inputs[1]["roles"]["feature_cols"] = ["x"]
    if case == "mixed":
        inputs[1]["roles"]["sample_id_col"] = None
        inputs[1]["drop_columns"] = ["device"]
    if case == "label":
        inputs[1]["roles"]["label_col"] = "device"
        inputs[1]["roles"]["sample_id_col"] = None
    if case == "units":
        inputs[0]["feature_units"] = {"x": ""}
    if case == "metadata":
        inputs[0]["drop_columns"] = []
    if case == "unsafe":
        inputs[0]["source_id"] = "../a"
    if case == "nested":
        inputs[0]["merge_result"] = str(tmp_path)
    if case == "preserve":
        extra["preserve_split"] = True
    assert plan(inputs, **extra)["status"] == "blocked"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.csv", "b.csv"]


def test_unknown_semantics_and_clock_are_not_guessed(tmp_path):
    inputs = sources(tmp_path)
    assert api().create_merge_plan(inputs, join="left")["status"] == "needs_clarification"
    assert api().create_merge_plan(inputs, entity_meaning="same")["status"] == "needs_clarification"
    p = Path(inputs[1]["path"])
    p.write_text(
        p.read_text()
        .replace("2024-01-01T08:00:00+08:00", "2024-01-01T00:00:00")
        .replace("2024-01-01T00:01:00.000001Z", "2024-01-01T00:01:00")
        .replace("2024-01-01T00:00:00Z", "2024-01-01T00:00:00")
    )
    assert plan(inputs)["status"] == "needs_clarification"


@pytest.mark.parametrize("count", [3, 4])
def test_bounded_source_fan_in_and_actual_output_name_conflict(tmp_path, count):
    inputs = sources(tmp_path)
    for i in range(2, count):
        path = tmp_path / f"source-{i}.csv"
        path.write_text(f"entity,time,f{i}\n001,2024-01-01T00:00:00Z,{i}\n")
        inputs.append(dict(source_id=f"s{i}", path=str(path), roles=dict(sample_id_col="entity", timestamp_col="time", feature_cols=[f"f{i}"]), retain_columns=[], drop_columns=[]))
    assert plan(inputs)["status"] == "ready"
    extra = copy.deepcopy(inputs[0]); extra["source_id"] = "extra"
    assert plan(inputs + [extra])["status"] == "blocked"
    p = Path(inputs[1]["path"])
    p.write_text(p.read_text().replace("clock,y\n", "clock,x\n"))
    inputs[1]["roles"]["feature_cols"] = ["x"]
    value = plan(inputs)
    assert value["status"] == "blocked" and "feature names conflict" in " ".join(value["errors"])


def test_source_change_and_sparse_limit_reject(tmp_path, monkeypatch):
    inputs = sources(tmp_path)
    selected = plan(inputs)
    Path(inputs[0]["path"]).write_text(Path(inputs[0]["path"]).read_text().replace(",0,A", ",9,A"))
    with pytest.raises(ValueError, match="changed"):
        api().validate_merge_plan(selected)
    monkeypatch.setitem(api().LIMITS, "feature_cells", 1)
    assert plan(inputs)["status"] == "blocked"


def test_outer_lineage_independent_inspection_and_rehashed_tamper(tmp_path, monkeypatch):
    inputs = sources(tmp_path)
    value = api().create_merge_plan(inputs, join="outer", entity_meaning="same devices")
    output = tmp_path / "merged"
    assert api().run_merge(value, output)["status"] == "success"
    data = pd.read_csv(output / "data.csv", dtype=str, keep_default_na=False)
    assert list(data["id"]) == ["001", "001", "001", "NA"]
    assert data.iloc[0]["x"] == "0" and data.iloc[1]["label"] == ""
    lineage = [json.loads(s) for s in (output / "lineage.jsonl").read_text().splitlines()]
    assert lineage[1]["cells"]["x"]["status"] == "unmatched-source"
    assert lineage[1]["cells"]["y"]["status"] == "original-missing"
    before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    for source in inputs:
        Path(source["path"]).unlink()
    assert api().inspect_merge_result(output)["status"] == "success"
    assert before == {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    import ai4ts.data_protocols.merge_artifacts as artifacts

    monkeypatch.setattr(artifacts, "_produce", lambda *args: pytest.fail("Inspector invoked producer"))
    assert artifacts._verify(output)["status"] == "success"
    (output / "data.csv").write_text((output / "data.csv").read_text().replace(",0,3,", ",8,3,"))
    from ai4ts.data_protocols.registry import file_sha256

    result = json.loads((output / "result.json").read_text())
    result["derived_sha256"] = result["files"]["data.csv"] = file_sha256(output / "data.csv")
    (output / "result.json").write_text(json.dumps(result))
    assert api().inspect_merge_result(output)["status"] == "failed"


def test_merge_handoff_is_unique_unsplit_and_time_choice_is_generic(tmp_path):
    from ai4ts.data_protocols.analysis import create_analysis_plan
    from ai4ts.data_protocols.ingestion import create_ingestion_plan
    from ai4ts.data_protocols.time_basis import preview_time_basis
    from ai4ts.data_protocols.temporal_preprocessing import (
        create_temporal_plan,
        run_temporal_preprocessing,
        inspect_temporal_result,
    )

    inputs = sources(tmp_path)
    for item, feature in zip(inputs, ["x", "y"]):
        role = item["roles"]
        item.update(retain_columns=[], drop_columns=[])
        role["label_col"] = None
        pd.DataFrame(
            {
                role["sample_id_col"]: ["001"] * 24,
                role["timestamp_col"]: [f"2024-01-01T00:{i:02}:00Z" for i in range(24)],
                feature: range(24),
            }
        ).to_csv(item["path"], index=False)
    merged = tmp_path / "merge"
    assert api().run_merge(plan(inputs), merged)["status"] == "success"
    csv = merged / "data.csv"
    assert create_analysis_plan(csv, merge_result=merged)["status"] == "ready"
    assert preview_time_basis(csv, merge_result=merged)["status"] == "ready"
    assert create_ingestion_plan(csv, "imputation", merge_result=merged)["status"] == "needs_clarification"
    chosen = create_ingestion_plan(
        csv,
        "imputation",
        split_policy="chronological",
        window_size=2,
        merge_result=merged,
        time_selection=dict(
            kind="observation_index",
            reason="Accepted record counts",
            impact_acknowledged=True,
            context=dict(task="imputation"),
        ),
    )
    assert chosen["status"] == "ready", chosen
    assert chosen["split"]["policy"] == "chronological" and "merge_result" in chosen
    temporal = create_temporal_plan(csv, mode="normalize", merge_result=merged)
    assert temporal["status"] == "ready", temporal
    output = tmp_path / "normalized"
    assert run_temporal_preprocessing(csv, temporal, output)["status"] == "success"
    assert inspect_temporal_result(output)["status"] == "success"
    ready = create_ingestion_plan(
        output / "data.csv", "imputation", split_policy="chronological", window_size=2, temporal_result=output
    )
    assert ready["status"] == "ready" and "merge_result" not in ready and "temporal_result" in ready
    assert create_analysis_plan(csv, merge_result=merged, temporal_result=output)["status"] == "blocked"
    # Fan-in is fixed; it cannot recursively merge an existing merge or its temporal child.
    inputs[0].update(path=str(output / "data.csv"), temporal_result=str(output), roles=ready["roles"])
    assert plan(inputs)["status"] == "blocked"


def test_reshape_temporal_parent_lineage_units_and_missing_parent(tmp_path):
    from ai4ts.data_protocols.reshape import create_reshape_plan, run_reshape
    from ai4ts.data_protocols.temporal_preprocessing import create_temporal_plan, run_temporal_preprocessing

    inputs = sources(tmp_path)
    long = tmp_path / "long.csv"
    long.write_text("id,time,variable,value\n001,2024-01-01T00:00:00Z,x,0\n001,2024-01-01T00:02:00Z,x,2\n")
    wide = tmp_path / "wide"
    rp = create_reshape_plan(
        long,
        roles=dict(sample_id_col="id", timestamp_col="time", variable_col="variable", value_col="value"),
        variables=["x"],
        feature_units={"x": "C"},
    )
    assert run_reshape(long, rp, wide)["status"] == "success"
    grid = tmp_path / "grid"
    tp = create_temporal_plan(wide / "data.csv", mode="regularize", frequency="1min", reshape_result=wide)
    assert run_temporal_preprocessing(wide / "data.csv", tp, grid)["status"] == "success"
    inputs[0] = dict(
        source_id="a", path=str(grid / "data.csv"), temporal_result=str(grid), retain_columns=[], drop_columns=[]
    )
    value = api().create_merge_plan(inputs, join="outer", entity_meaning="same devices")
    assert value["status"] == "ready", value
    output = tmp_path / "merged"
    assert api().run_merge(value, output)["status"] == "success"
    lines = [json.loads(s) for s in (output / "lineage.jsonl").read_text().splitlines()]
    assert any(row["cells"]["x"]["status"] == "parent-grid-added" for row in lines)
    assert value["preview"]["features"]["x"]["original_missing"] == 0
    assert value["preview"]["features"]["x"]["parent_missing"] == {"grid-added": 1}
    assert api().merge_context(output, output / "data.csv")["feature_semantics"]["units"] == {"x": "C", "y": "unknown"}
    (wide / "result.json").unlink()
    assert api().inspect_merge_result(output)["status"] == "failed"


def test_parent_aliases_cannot_bypass_bounded_fan_in(tmp_path):
    root = tmp_path / "wide"
    root.mkdir()
    (root / "plan.json").write_text(json.dumps(dict(schema="timeseries-reshape-plan-v1")))
    with pytest.raises(ValueError, match="Duplicate/cyclic"):
        api()._parents([dict(reshape_result=str(root)), dict(reshape_result=str(root / ".." / "wide"))])


def test_temporal_parent_retains_absent_measurement_cause(tmp_path):
    from ai4ts.data_protocols.reshape import create_reshape_plan, run_reshape
    from ai4ts.data_protocols.temporal_preprocessing import create_temporal_plan, run_temporal_preprocessing

    inputs = sources(tmp_path)
    long = tmp_path / "long.csv"
    long.write_text("id,time,variable,value\n001,2024-01-01T00:00:00Z,x,0\n001,2024-01-01T00:01:00Z,z,2\n")
    wide = tmp_path / "wide"
    rp = create_reshape_plan(
        long,
        roles=dict(sample_id_col="id", timestamp_col="time", variable_col="variable", value_col="value"),
        variables=["x", "z"],
    )
    assert run_reshape(long, rp, wide)["status"] == "success"
    temporal = tmp_path / "normalized"
    tp = create_temporal_plan(wide / "data.csv", mode="normalize", reshape_result=wide)
    assert run_temporal_preprocessing(wide / "data.csv", tp, temporal)["status"] == "success"
    inputs[0] = dict(
        source_id="a",
        path=str(temporal / "data.csv"),
        temporal_result=str(temporal),
        retain_columns=[],
        drop_columns=[],
    )
    selected = plan(inputs)
    assert selected["preview"]["features"]["x"]["original_missing"] == 0
    assert selected["preview"]["features"]["x"]["parent_missing"] == {"absent-measurement": 1}
    result = tmp_path / "merged"
    assert api().run_merge(selected, result)["status"] == "success"
    rows = [json.loads(s) for s in (result / "lineage.jsonl").read_text().splitlines()]
    assert rows[1]["cells"]["x"]["status"] == "parent-absent-measurement"
    assert (
        rows[1]["cells"]["x"]["parent"]["evidence"]["ancestor_cells"][0]["evidence"]["unknown_reason"]
        == "absent-measurement"
    )
    assert api().inspect_merge_result(result)["status"] == "success"


def test_real_timeout_and_partial_write_failure_preserve_inputs(tmp_path, monkeypatch):
    inputs = sources(tmp_path)
    selected = plan(inputs)
    output = tmp_path / "timeout"
    result = api().run_merge(selected, output, timeout=1e-9)
    assert result["status"] == "failed" and result["timed_out"]
    assert not (output / "result.json").exists()
    with pytest.raises(FileExistsError):
        api().run_merge(selected, output)
    import ai4ts.data_protocols.merge_artifacts as artifacts

    original = artifacts.write_json

    def fail(path, value):
        if Path(path).name == "match-summary.json":
            raise OSError("Controlled write fault")
        return original(path, value)

    monkeypatch.setattr(artifacts, "write_json", fail)
    partial = tmp_path / "partial"
    partial.mkdir()
    result = artifacts._work(dict(operation="run", plan=selected, output=str(partial)))
    assert result["status"] == "failed" and "Controlled write fault" in result["error"]
    assert (partial / "data.csv").is_file() and not (partial / "result.json").exists()
