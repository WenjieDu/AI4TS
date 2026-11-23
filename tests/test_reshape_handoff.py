"""Derived references preserve long-cell semantics through existing workflows."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json

import pandas as pd
import pytest

from ai4ts.data_protocols import create_reshape_plan, run_reshape, create_temporal_plan, run_temporal_preprocessing
from ai4ts.data_protocols.analysis import create_analysis_plan
from ai4ts.data_protocols.ingestion import create_ingestion_plan, validate_ingestion_plan


def transformed(tmp_path, b_unit="kPa"):
    source = tmp_path / "long.csv"
    origin = pd.Timestamp("2024-01-01T00:00:00Z")
    source.write_text(
        "id,time,metric,value,unit\n"
        + "".join(
            f"001,{(origin+pd.Timedelta(minutes=5*i+(60 if i>=20 else 0))).isoformat()},{name},{i+j},{unit}\n"
            for i in range(50)
            for j, name, unit in [(0, "a", "C"), (100, "b", b_unit)]
        )
    )
    plan = create_reshape_plan(
        source,
        roles=dict(sample_id_col="id", timestamp_col="time", variable_col="metric", value_col="value", unit_col="unit"),
        variables=["a", "b"],
    )
    root = tmp_path / "reshape"
    assert run_reshape(source, plan, root)["status"] == "success"
    return root, root / "data.csv"


def test_direct_handoff_requires_new_evaluation_and_reuses_units_roles(tmp_path):
    root, csv = transformed(tmp_path)
    value = create_analysis_plan(csv, reshape_result=root)
    assert value["status"] == "ready" and value["selected_rows"] == list(range(50))
    assert value["feature_semantics"]["units"] == {"a": "C", "b": "kPa"}
    assert "split" not in value or value["split"] is None
    pending = create_ingestion_plan(csv, "imputation", reshape_result=root, window_size=3, allow_irregular=True)
    assert pending["status"] == "needs_clarification"
    intake = create_ingestion_plan(
        csv, "imputation", reshape_result=root, split_policy="entity_chronological", window_size=3, allow_irregular=True
    )
    assert intake["status"] == "ready", intake
    assert [len(rows) for rows in intake["split"]["rows"].values()] == [35, 5, 10]
    validate_ingestion_plan(intake, csv)
    subset = create_analysis_plan(
        csv,
        reshape_result=root,
        roles=dict(sample_id_col="id", timestamp_col="time", feature_cols=["b"], label_col=None),
    )
    assert subset["status"] == "ready" and subset["feature_semantics"]["units"] == {"b": "kPa"}
    assert subset["feature_semantics"]["inherited_from_sha256"]


def test_temporal_chain_preserves_new_wide_split_and_long_parent(tmp_path):
    root, csv = transformed(tmp_path)
    donor = create_ingestion_plan(
        csv, "imputation", reshape_result=root, split_policy="entity_chronological", window_size=3, allow_irregular=True
    )
    plan = create_temporal_plan(csv, "regularize", reshape_result=root, split_plan=donor, frequency="5min")
    assert plan["status"] == "ready", plan
    temporal = tmp_path / "temporal"
    assert run_temporal_preprocessing(csv, plan, temporal)["status"] == "success"
    result = create_ingestion_plan(temporal / "data.csv", "imputation", temporal_result=temporal, window_size=3)
    assert result["status"] == "ready", result
    assert [len(rows) for rows in result["split"]["rows"].values()] == [47, 5, 10]
    assert result["feature_semantics"]["units"] == {"a": "C", "b": "kPa"}
    scoped = create_analysis_plan(temporal / "data.csv", temporal_result=temporal, scope="train")
    assert scoped["status"] == "ready" and scoped["selected_rows"] == list(range(47))
    blocked = create_analysis_plan(csv, reshape_result=root, temporal_result=temporal)
    assert blocked["status"] == "blocked"
    # Only a long-to-wide parent is accepted; a temporal result cannot form a deeper chain.
    assert create_temporal_plan(temporal / "data.csv", "normalize", reshape_result=temporal)["status"] == "blocked"
    assert create_analysis_plan(temporal / "data.csv", reshape_result=temporal)["status"] == "blocked"
    from ai4ts.data_protocols import reshape_artifacts

    key = str(root.resolve())
    reshape_artifacts._VERIFY_STACK.add(key)
    try:
        with pytest.raises(ValueError, match="Cyclic"):
            reshape_artifacts._verify(root)
    finally:
        reshape_artifacts._VERIFY_STACK.remove(key)
    # Parent loss cannot silently downgrade the chain to a bare CSV.
    (root / "lineage.jsonl").rename(root / "lost-lineage.jsonl")
    failed = create_analysis_plan(temporal / "data.csv", temporal_result=temporal)
    assert failed["status"] == "blocked"


def test_parent_is_verified_in_same_owned_worker_without_grandchildren(tmp_path, monkeypatch):
    root, csv = transformed(tmp_path)
    plan = create_temporal_plan(csv, "regularize", reshape_result=root, frequency="5min")
    assert plan["status"] == "ready", plan
    from ai4ts.data_protocols import temporal_artifacts as artifacts
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("Nested parent verification spawned another worker")

    monkeypatch.setattr(subprocess, "run", forbidden)
    output = tmp_path / "owned"
    output.mkdir()
    result = artifacts._work(dict(operation="run", source=str(csv), plan=plan, output=str(output)))
    assert result["status"] == "success", result


def test_aggregate_train_right_label_remains_train_with_reshape_parent(tmp_path):
    source = tmp_path / "long.csv"
    origin = pd.Timestamp("2024-01-01T00:00:00Z")
    source.write_text(
        "id,time,metric,value\n"
        + "".join(f"001,{(origin+pd.Timedelta(minutes=5*i)).isoformat()},a,{i}\n" for i in range(40))
    )
    reshape = create_reshape_plan(
        source,
        roles=dict(sample_id_col="id", timestamp_col="time", variable_col="metric", value_col="value"),
        variables=["a"],
    )
    root = tmp_path / "reshape"
    assert run_reshape(source, reshape, root)["status"] == "success"
    csv = root / "data.csv"
    donor = create_ingestion_plan(
        csv, "imputation", reshape_result=root, split_policy="entity_chronological", window_size=2
    )
    temporal = create_temporal_plan(
        csv, "aggregate", reshape_result=root, split_plan=donor, frequency="10min", reducers={"a": "mean"}
    )
    assert temporal["status"] == "ready", temporal
    output = tmp_path / "aggregate"
    assert run_temporal_preprocessing(csv, temporal, output)["status"] == "success"
    lines = [json.loads(s) for s in (output / "lineage.jsonl").read_text().splitlines()]
    assert lines[13]["split"] == "train" and lines[13]["source_rows"] == [26, 27]
    assert lines[14]["split"] == "val"
    crossing = create_temporal_plan(
        csv, "aggregate", reshape_result=root, split_plan=donor, frequency="15min", reducers={"a": "mean"}
    )
    assert crossing["status"] == "blocked"


def test_confirmed_units_fill_unknown_parent_without_overwriting_source_units(tmp_path):
    root, csv = transformed(tmp_path, b_unit="")
    prior = create_analysis_plan(csv, reshape_result=root, feature_units={"b": "kPa"})
    assert prior["status"] == "ready", prior
    temporal = create_temporal_plan(csv, "regularize", reshape_result=root, confirmed_plan=prior, frequency="5min")
    assert temporal["status"] == "ready", temporal
    assert temporal["feature_semantics"]["units"] == {"a": "C", "b": "kPa"}
    conflict = create_temporal_plan(csv, "regularize", reshape_result=root, frequency="5min", feature_units={"a": "K"})
    assert conflict["status"] == "blocked"


def test_wrong_input_roles_reference_and_units_are_rejected(tmp_path):
    root, csv = transformed(tmp_path)
    from ai4ts.data_protocols.reshape import reshape_context

    verified = reshape_context(root, csv)
    wrong = dict(verified["reference"], source_sha256="0" * 64)
    assert create_analysis_plan(csv, reshape_result=wrong)["status"] == "blocked"
    bad_roles = dict(verified["roles"], timestamp_col=None)
    assert create_analysis_plan(csv, reshape_result=root, roles=bad_roles)["status"] == "blocked"
    assert create_analysis_plan(csv, reshape_result=root, feature_units={"a": "K"})["status"] == "blocked"
    csv.write_text(csv.read_text().replace(",0,100", ",9,100", 1))
    assert create_analysis_plan(csv, reshape_result=root)["status"] == "blocked"
