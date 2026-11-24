"""Tests for source-bound ingestion plans and prepared windows."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import pandas as pd
import pytest

from ai4ts.data_protocols import pipeline
from ai4ts.data_protocols.ingestion import (
    create_ingestion_plan,
    validate_ingestion_plan,
    verify_prepared_plan,
    content_sha256,
    render_ingestion_preview,
)

ROLES = dict(sample_id_col="device_id", timestamp_col="time", feature_cols=["temperature"], label_col=None)


def source_file(tmp_path, n=51, entities=("001", "NA")):
    frame = pd.DataFrame(
        [
            {"device_id": sid, "time": str(t), "temperature": np.sin(i / 3) + e, "note": "unchanged"}
            for e, sid in enumerate(entities)
            for i, t in enumerate(pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"))
        ]
    )
    path = tmp_path / "source.csv"
    frame.iloc[::-1].to_csv(path, index=False)
    return path


def plan_for(path, task="imputation", **kw):
    return create_ingestion_plan(path, task, roles=ROLES, split_policy="entity_chronological", window_size=3, **kw)


def test_roles_clarification_and_preserved_ids(tmp_path):
    path = source_file(tmp_path)
    before = path.read_bytes()
    pending = create_ingestion_plan(path, "imputation")
    assert pending["status"] == "needs_clarification"
    assert pending["source"]["candidates"]["device_id"]["role"] == "sample_id_col"
    plan = plan_for(path)
    assert plan["status"] == "ready", plan
    assert {s["sample_id"] for s in plan["sampling"]} == {"001", "NA"}
    assert not plan["unresolved"] and plan["roles"]["feature_cols"] == ["temperature"]
    assert path.read_bytes() == before
    assert "Actual artificial mask counts" in render_ingestion_preview(plan)


@pytest.mark.parametrize("header", ["x,x", "x,", ""])
def test_bad_headers(tmp_path, header):
    p = tmp_path / "source.csv"
    p.write_text(header + "\n1,2\n")
    assert create_ingestion_plan(p, "imputation")["status"] == "blocked"


@pytest.mark.parametrize(
    "task,options",
    [
        ("imputation", {}),
        ("forecasting", {"horizon": 2, "targets": ["temperature"]}),
        ("anomaly_detection", {"anomaly_rate": 0.1}),
    ],
)
def test_real_preparation_matches_manual_counts_and_source_rows(tmp_path, task, options):
    path = source_file(tmp_path)
    plan = plan_for(path, task, **options)
    assert plan["status"] == "ready", plan
    validate_ingestion_plan(plan, path)
    # Each entity: 36 train, 5 validation, 10 test rows. L=3, H=2.
    expected = [64, 2, 12] if task == "forecasting" else [24, 4, 8]
    assert [plan["preview"][r]["windows"] for r in ("train", "val", "test")] == expected
    state = scaler = None
    for role in plan["preview"]:
        data = pipeline.prepare_for_pypots(
            path,
            task=task,
            schema_mapping=ROLES,
            window_size=3,
            missing_rate=0.2,
            set_type=role,
            row_indices=plan["split"]["rows"].get(role),
            split_info=plan["split"],
            preprocessing="standard-v1",
            task_state=state,
            scaler_state=scaler,
            n_pred_steps=options.get("horizon"),
            target_features=options.get("targets"),
        )
        verify_prepared_plan(plan, role, data)
        state, scaler = data.get("task_state"), data.get("scaler_state")
        allowed = set(plan["split"]["rows"].get(role, range(102)))
        for e in data["registry"].entries:
            assert set(e.row_indices + e.target_row_indices).issubset(allowed)
            raw = pd.read_csv(path, dtype=str, keep_default_na=False)
            assert set(raw.iloc[e.row_indices]["device_id"]) == {e.original_sample_id}
        if task == "imputation":
            flat = [i for e in data["registry"].entries for i in e.row_indices]
            assert set(flat) == allowed and len(flat) == len(allowed)
    changed = copy.deepcopy(plan)
    changed["preview"]["train"]["mapping_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from plan"):
        verify_prepared_plan(changed, "train", pipeline.prepare_for_pypots(path, schema_mapping=ROLES, window_size=3))


def test_plan_is_no_fit_no_mask_no_tensor_and_bounded(tmp_path, monkeypatch):
    path = source_file(tmp_path)
    import ai4ts.data_protocols.preprocessing as preprocessing

    def forbidden(*a, **kw):
        pytest.fail("plan cannot prepare/fit/mask")

    for name in ("prepare_for_pypots", "_transform", "_inject_artificial_missing"):
        monkeypatch.setattr(pipeline, name, forbidden)
    monkeypatch.setattr(preprocessing, "fit_scaler", forbidden)
    assert plan_for(path)["status"] == "ready"
    huge = create_ingestion_plan(
        path, "imputation", roles=ROLES, split_policy="entity_chronological", window_size=10_000_001
    )
    assert huge["status"] == "blocked" and "cells" in huge["errors"][0]
    assert set(p.name for p in tmp_path.iterdir()) == {"source.csv"}


def test_plan_tampering_recomputed_hash_and_relocation(tmp_path):
    path = source_file(tmp_path)
    plan = plan_for(path)
    moved = tmp_path / "moved.csv"
    moved.write_bytes(path.read_bytes())
    validate_ingestion_plan(plan, moved)
    for key in ("version", "split", "source"):
        changed = copy.deepcopy(plan)
        if key == "version":
            changed[key] = 2
        elif key == "split":
            changed[key]["rows"]["val"][0] = changed[key]["rows"]["train"][0]
        else:
            changed[key]["columns"].reverse()
        changed["sha256"] = content_sha256({k: v for k, v in changed.items() if k != "sha256"})
        with pytest.raises(ValueError):
            validate_ingestion_plan(changed, path)
    moved.write_bytes(path.read_bytes().replace(b"unchanged", b"changed", 1))
    with pytest.raises(ValueError, match="mismatch"):
        validate_ingestion_plan(plan, moved)


def test_durations_gaps_and_independent_policy(tmp_path):
    path = source_file(tmp_path, n=4000, entities=("001",))
    plan = create_ingestion_plan(
        path,
        "forecasting",
        roles=ROLES,
        split_policy="chronological",
        window_size="24h",
        horizon="2h",
        targets=["temperature"],
    )
    assert plan["status"] == "ready", plan
    assert (plan["parameters"]["window_size"], plan["parameters"]["horizon"]) == (288, 24)
    bad = create_ingestion_plan(
        path,
        "forecasting",
        roles=ROLES,
        split_policy="chronological",
        window_size="1min",
        horizon=1,
        targets=["temperature"],
    )
    assert bad["status"] == "blocked" and "exact" in bad["errors"][0]
    frame = pd.read_csv(path)
    frame = frame.drop(5)
    frame.to_csv(path, index=False)
    bad = create_ingestion_plan(
        path,
        "forecasting",
        roles=ROLES,
        split_policy="chronological",
        window_size="1h",
        horizon=1,
        targets=["temperature"],
        allow_irregular=True,
    )
    assert bad["status"] == "blocked" and "regular frequency" in bad["errors"][0]
    assert plan_for(source_file(tmp_path), "classification")["status"] == "blocked"
    with pytest.raises(ValueError, match="Two entities"):
        pipeline.split_source_rows(source_file(tmp_path), ROLES)


def test_short_and_bad_entity_splits(tmp_path):
    path = source_file(tmp_path, n=5)
    assert plan_for(path)["status"] == "blocked"
    path = source_file(tmp_path)
    plan = plan_for(path)
    source, _ = pipeline._read_source(path, "imputation", ROLES, False)
    for kind in ("omit", "reverse", "duplicate"):
        split = copy.deepcopy(plan["split"])
        if kind == "omit":
            split["rows"]["train"].pop()
        elif kind == "reverse":
            split["rows"]["train"][0], split["rows"]["test"][0] = split["rows"]["test"][0], split["rows"]["train"][0]
        else:
            split["rows"]["val"][0] = split["rows"]["train"][0]
        with pytest.raises(ValueError):
            pipeline.validate_source_split(source, ROLES, split, "imputation", split["rows"]["train"])
    for task in ("classification", "clustering", "representation"):
        with pytest.raises(ValueError, match="Independent"):
            pipeline.validate_source_split(source, ROLES, plan["split"], task, plan["split"]["rows"]["train"])


@pytest.mark.parametrize("task", ["classification", "clustering", "representation"])
def test_whole_sample_padding_plan_consumption_without_models(tmp_path, task):
    path = tmp_path / "samples.csv"
    pd.DataFrame(
        [{"id": str(sid), "label": str(sid % 2), "x": np.sin(t)} for sid in range(12) for t in range(8)]
    ).to_csv(path, index=False)
    roles = dict(sample_id_col="id", timestamp_col=None, feature_cols=["x"], label_col="label")
    plan = create_ingestion_plan(
        path, task, roles=roles, split_policy="entity", n_clusters=2 if task == "clustering" else None
    )
    assert plan["status"] == "ready", plan
    state = None
    for role in ("train", "val", "test"):
        result = pipeline.prepare_for_pypots(
            path, task=task, set_type=role, ingestion_plan=plan, preprocessing="standard-v1", task_state=state
        )
        state = result["task_state"]
        verify_prepared_plan(plan, role, result)
        ids = result["registry"].original_sample_ids
        assert len(ids) == result["X"].shape[0]
        assert result["X"].shape[1:] == (8, 1)
