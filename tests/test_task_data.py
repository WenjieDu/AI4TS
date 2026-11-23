"""Tests for classification and forecasting data and exports."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import pandas as pd
import pytest

from ai4ts.data_protocols.pipeline import prepare_for_pypots
from ai4ts.data_protocols.registry import WindowRegistry


def classification_source(tmp_path):
    source = tmp_path / "class.csv"
    source.write_text("id,label,a,b\n001,zebra,1,2\n001,zebra,3,4\n002,ant,5,6\n002,ant,7,8\n003,zebra,100,200\n")
    roles = {"sample_id_col": "id", "label_col": "label", "feature_cols": ["a", "b"]}
    split = {"policy": "entity", "rows": {"train": [0, 1, 2, 3], "val": [4], "test": []}}
    options = dict(
        task="classification", schema_mapping=roles, preprocessing="standard-v1", split_info=split, missing_rate=0.3
    )
    train = prepare_for_pypots(source, row_indices=split["rows"]["train"], **options)
    return source, options, train


def test_classification_training_state_subset_padding_and_numeric_labels(tmp_path):
    source, options, train = classification_source(tmp_path)
    val = prepare_for_pypots(source, row_indices=[4], set_type="val", task_state=train["task_state"], **options)
    assert train["task_state"]["classes"] == ["ant", "zebra"]
    np.testing.assert_equal(train["y"], [1, 0])
    np.testing.assert_equal(val["y"], [1])
    assert val["X"].shape == (1, 2, 2)
    assert np.isnan(val["X"][0, 1]).all() and not val["valid_mask"][0, 1].any()
    assert train["scaler_state"]["mean"] == [4, 5]
    assert val["registry"].entries[0].original_sample_id == "003"
    registry = WindowRegistry.from_dict(val["registry"].to_dict())
    registry.validate_task_binding(source, registry.prediction_binding(), train["task_state"])
    source.write_text(source.read_text().replace("zebra", "20").replace("ant", "10"))
    numeric = prepare_for_pypots(source, row_indices=[0, 1, 2, 3], **options)
    assert numeric["task_state"]["classes"] == ["10", "20"]
    np.testing.assert_equal(numeric["y"], [1, 0])
    raw = prepare_for_pypots(source, row_indices=[0, 1, 2, 3], **dict(options, preprocessing=None, missing_rate=0.9))
    np.testing.assert_equal(raw["X"], raw["X_ori"])


def test_classification_rejections_and_binding(tmp_path):
    source, options, train = classification_source(tmp_path)
    with pytest.raises(ValueError, match="saved training"):
        prepare_for_pypots(source, set_type="val", row_indices=[4], **options)
    original = source.read_text()
    for replacement, match in (("003,unknown,100,200", "Unknown"), ("003,,100,200", "missing")):
        source.write_text(original.replace("003,zebra,100,200", replacement))
        with pytest.raises(ValueError, match=match):
            prepare_for_pypots(source, set_type="val", row_indices=[4], task_state=train["task_state"], **options)
    source.write_text(original.replace("001,zebra,3,4", "001,ant,3,4"))
    with pytest.raises(ValueError, match="Conflicting"):
        prepare_for_pypots(source, row_indices=[0, 1, 2, 3], **options)
    source.write_text(original)
    wrong = copy.deepcopy(train["task_state"])
    wrong["training_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source"):
        prepare_for_pypots(source, set_type="val", row_indices=[4], task_state=wrong, **options)
    registry = train["registry"]
    wrong = copy.deepcopy(train["task_state"])
    wrong["classes"] = ["zebra", "ant"]
    with pytest.raises(ValueError, match="binding"):
        registry.validate_task_binding(source, registry.prediction_binding(), wrong)
    source.write_text(original + "003,zebra,1,2\n003,zebra,2,3\n")
    opts = dict(options, split_info=None)
    with pytest.raises(ValueError, match="exceeds"):
        prepare_for_pypots(source, set_type="test", row_indices=[4, 5, 6], task_state=train["task_state"], **opts)
    with pytest.raises(ValueError, match="binding"):
        registry.validate_task_binding(source, registry.prediction_binding(), train["task_state"])
    source.write_text(original)
    bad_split = {"rows": {"train": [0, 1, 2, 3], "val": [1, 4], "test": []}}
    with pytest.raises(ValueError, match="overlap"):
        prepare_for_pypots(source, row_indices=[0, 1, 2, 3], **dict(options, split_info=bad_split))


def test_forecast_targets_split_order_benchpots_and_live_future(tmp_path):
    from benchpots.utils.task_type import convert_processed_dataset_by_task_type

    source = tmp_path / "forecast.csv"
    df = pd.DataFrame({"a": np.arange(60), "b": np.arange(60) * 10, "c": np.arange(60) * 100})
    df.loc[45, "c"] = np.nan
    df.to_csv(source, index=False)
    split = {
        "policy": "row_order",
        "rows": {"train": list(range(30)), "val": list(range(30, 45)), "test": list(range(45, 60))},
    }
    options = dict(
        task="forecasting",
        window_size=4,
        n_pred_steps=2,
        target_features=["c", "a"],
        schema_mapping={"feature_cols": ["a", "b", "c"]},
        split_info=split,
        preprocessing="standard-v1",
        missing_rate=0,
    )
    train = prepare_for_pypots(source, row_indices=split["rows"]["train"], **options)
    assert train["scaler_state"]["mean"] == [14.5, 145, 1450]
    state = train["task_state"]
    test = prepare_for_pypots(source, row_indices=split["rows"]["test"], set_type="test", task_state=state, **options)
    assert test["X"].shape == (10, 4, 3) and test["X_pred"].shape == (10, 2, 2)
    assert "indicating_mask" not in test
    e = test["registry"].entries[0]
    assert e.row_indices == [45, 46, 47, 48] and e.target_row_indices == [49, 50]
    assert e.forecast_origin == 48
    standardized = (df.to_numpy() - state["scaler_state"]["mean"]) / state["scaler_state"]["scale"]
    np.testing.assert_allclose(test["X_pred"][0], standardized[49:51, [2, 0]])
    # The same complete history+future block follows BenchPOTS' target conversion.
    block = standardized[:6][None]
    ref_data = {"n_steps": 6, "n_features": 3}
    for role in ("train", "val", "test"):
        ref_data.update({f"{role}_X": block.copy(), f"{role}_X_ori": block.copy()})
    ref = convert_processed_dataset_by_task_type(
        ref_data, task_type="forecasting", n_pred_steps=2, forecast_feature_indices=[2, 0]
    )
    np.testing.assert_allclose(train["X"][0], ref["train_X"][0])
    np.testing.assert_allclose(train["X_pred"][0], ref["train_X_pred"][0])
    live = prepare_for_pypots(source, set_type="full", task_state=state, **options)
    assert not live["X_pred_valid_mask"][-1].any() and np.isnan(live["X_pred"][-1]).all()
    assert live["registry"].entries[-1].row_indices == [56, 57, 58, 59]
    for result in (train, test, live):
        reg = WindowRegistry.from_dict(result["registry"].to_dict())
        reg.validate_task_binding(source, reg.prediction_binding(), state)
        for e in reg.entries:
            assert not set(e.row_indices).intersection(e.target_row_indices)
    for updates in (
        {"target_features": []},
        {"target_features": ["label"]},
        {"n_pred_steps": 0},
        {"target_features": ["a", "c"]},
        {"window_size": 5},
    ):
        with pytest.raises(ValueError):
            prepare_for_pypots(
                source, set_type="test", row_indices=split["rows"]["test"], task_state=state, **dict(options, **updates)
            )
    bad = {"rows": {"train": list(range(30)), "val": list(range(45, 60)), "test": list(range(30, 45))}}
    with pytest.raises(ValueError, match="chronological"):
        prepare_for_pypots(source, row_indices=bad["rows"]["train"], **dict(options, split_info=bad))


@pytest.mark.parametrize("timezone", [None, "Asia/Shanghai"])
def test_forecast_export_preserves_timezone_or_observation_index(tmp_path, timezone):
    from ai4ts.data_protocols.task_data import export_forecasts

    path = tmp_path / "timed.csv"
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=24, tz=timezone).astype(str),
            "a": np.arange(24),
            "b": np.arange(24) * 10,
        }
    )
    frame.to_csv(path, index=False)
    split = {"rows": {"train": list(range(12)), "val": list(range(12, 18)), "test": list(range(18, 24))}}
    options = dict(
        task="forecasting",
        window_size=4,
        n_pred_steps=2,
        target_features=["b", "a"],
        schema_mapping={"timestamp_col": "time", "feature_cols": ["a", "b"]},
        split_info=split,
        preprocessing="standard-v1",
        missing_rate=0,
    )
    train = prepare_for_pypots(path, row_indices=split["rows"]["train"], **options)
    full = prepare_for_pypots(path, set_type="full", task_state=train["task_state"], **options)
    reg = full["registry"]
    table = export_forecasts(np.zeros_like(full["X_pred"]), reg, path, reg.prediction_binding(), train["task_state"])
    final = table.iloc[-4:]
    assert final.origin_step.eq(23).all() and final.target_step.tolist() == [24, 24, 25, 25]
    assert final.time_basis.eq("regular_timestamp").all() and not final.evaluated.any()
    expected = pd.Timestamp(frame.time.iloc[-1]) + pd.Timedelta(days=1)
    assert final.target_time.iloc[0] == expected.isoformat()
    if timezone is None:
        assert "+00:00" not in final.target_time.iloc[0]
    # Explicit index analysis never invents a future clock.
    index_full = prepare_for_pypots(
        path, set_type="full", task_state=train["task_state"], allow_irregular=True, **options
    )
    reg = index_full["registry"]
    table = export_forecasts(
        np.zeros_like(index_full["X_pred"]), reg, path, reg.prediction_binding(), train["task_state"]
    )
    assert table.iloc[-4:].time_basis.eq("observation_index").all() and table.iloc[-4:].target_time.isna().all()
