"""Forecast input order, lead availability and training-only categorical encoding."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import numpy as np
import pandas as pd
import pytest

from ai4ts.data_protocols import (
    WindowRegistry,
    create_ingestion_plan,
    prepare_for_pypots,
)
from ai4ts.data_protocols.preprocessing import transform_targets
from ai4ts.data_protocols.registry import file_sha256
from ai4ts.data_protocols.task_data import export_forecasts


def declaration(role, kind):
    return {"role": role, "kind": kind, "evidence": {"origin": "user-confirmed", "reference": "Fixture"}}


@pytest.mark.parametrize(("preprocessing", "frequency"), [(None, "D"), ("standard-v1", "D"), (None, "MS")])
def test_lead_never_uses_values_after_origin_plus_horizon(tmp_path, preprocessing, frequency):
    dates = pd.date_range("2024-01-01", periods=33, freq=frequency)
    history, future = tmp_path / "history.csv", tmp_path / "future.csv"
    frame = pd.DataFrame({
        "time": dates[:30], "past": np.arange(30) + 100, "promo": np.arange(30) + 1000,
        "units": np.arange(30) * 10, "revenue": np.arange(30) + 400, "capacity": 12.0,
    })
    frame.to_csv(history, index=False)
    pd.DataFrame({"time": dates[30:], "promo": np.arange(30, 33) + 1000}).to_csv(future, index=False)
    rows = {"train": list(range(10)), "val": list(range(10, 20)), "test": list(range(20, 30))}
    options = {
        "task": "forecasting", "window_size": 4, "n_pred_steps": 3, "target_features": ["revenue", "units"],
        "schema_mapping": {"timestamp_col": "time", "feature_cols": list(frame.columns[1:])},
        "covariates": {"promo": declaration("known", "numeric"), "capacity": declaration("static", "numeric")},
        "feature_units": {"promo": "count"}, "split_info": {"rows": rows}, "missing_rate": 0,
        "preprocessing": preprocessing, "allow_irregular": frequency == "MS",
    }
    train = prepare_for_pypots(history, row_indices=rows["train"], **options)
    state = train["task_state"]
    outcomes = [train] + [
        prepare_for_pypots(history, row_indices=rows[role], set_type=role, task_state=state, **options)
        for role in ("val", "test")
    ]
    full = prepare_for_pypots(
        history, set_type="full", task_state=state, future_csv=future, frequency=frequency, **options
    )
    for result in [*outcomes, full]:
        reg = WindowRegistry.from_dict(result["registry"].to_dict())
        assert reg.feature_columns == ["revenue", "units", "past", "promo@lead", "capacity"]
        assert reg.column_roles["feature_cols"] == list(frame.columns[1:])
        reg.validate_task_binding(history, reg.prediction_binding(), state)
        for entry in reg.entries:
            x = result["X"][entry.window_id]
            if preprocessing:
                x = transform_targets(
                    x, state["scaler_state"], ["revenue", "units", "past", "promo", "capacity"], inverse=True
                )
            source_rows = np.asarray(entry.row_indices)
            np.testing.assert_allclose(x[:, :3], frame.loc[source_rows, ["revenue", "units", "past"]])
            np.testing.assert_allclose(x[:, 3], 1000 + source_rows + 3)
            np.testing.assert_allclose(
                x[-3:, 3], 1000 + np.arange(entry.forecast_origin + 1, entry.forecast_origin + 4)
            )
            assert np.isclose(max(x[:, 3] - 1000), entry.forecast_origin + 3, rtol=0, atol=1e-9)
            assert np.all(x[:, 4] == 12)
    assert np.isnan(full["X_pred"][-1]).all() and not full["X_pred_valid_mask"][-1].any()
    reg = full["registry"]
    assert reg.split_info["future_covariates"]["source_sha256"] == file_sha256(future)
    predictions = np.broadcast_to([91, 24], full["X_pred"].shape)
    if preprocessing:
        predictions = transform_targets(predictions, state["scaler_state"], options["target_features"])
    table = export_forecasts(predictions, reg, history, reg.prediction_binding(), state)
    assert table.iloc[-6:].target.tolist() == ["revenue", "units"] * 3
    np.testing.assert_allclose(table.iloc[-6:].prediction, [91, 24] * 3)
    assert table.time_basis.eq("regular_timestamp").all()
    assert not table.iloc[-6:].evaluated.any()
    times = dates[30:].tz_localize("UTC").repeat(2)
    assert pd.to_datetime(table.iloc[-6:].target_time, utc=True).tolist() == times.tolist()
    assert table.iloc[:2].truth.tolist() == [404, 40]
    with pytest.raises(ValueError, match="future"):
        prepare_for_pypots(history, set_type="full", task_state=state, **options)
    with pytest.raises(ValueError, match="future"):
        prepare_for_pypots(history, row_indices=rows["train"], future_csv=future, **options)
    pd.DataFrame({"time": dates[30:] + pd.Timedelta(days=1), "promo": [1030, 1031, 1032]}).to_csv(future, index=False)
    with pytest.raises(ValueError, match="continue"):
        prepare_for_pypots(
            history, set_type="full", task_state=state, future_csv=future, frequency=frequency, **options
        )
    with pytest.raises(ValueError, match="contiguous"):
        prepare_for_pypots(history, row_indices=[*range(5), *range(6, 10)], **dict(options, split_info=None))
    with pytest.raises(ValueError, match="window.*horizon"):
        prepare_for_pypots(history, row_indices=rows["train"], **dict(options, window_size=2))


@pytest.mark.parametrize(("targets", "preprocessing"), [(["y"], None), (["z", "y"], None), (["z", "y"], "standard-v1")])
def test_categorical_encoding_per_target_uses_only_training_rows(tmp_path, targets, preprocessing):
    history, held_out = tmp_path / "history.csv", tmp_path / "held.csv"
    train_data = pd.DataFrame({
        "id": ["a"] * 4 + ["b"] * 6, "tag": ["001", "001", None, "NA"] + ["001"] * 6,
        "y": [2, 6, 8, 4] + [10] * 6, "z": [20, 40, 80, 60] + [100] * 6,
    })
    eval_data = pd.DataFrame({
        "id": ["a"] * 4 + ["new"] * 4, "tag": ["001", None, "unseen", "NA"] * 2,
        "y": [900] * 8, "z": [999] * 8,
    })
    pd.concat([train_data, eval_data]).to_csv(history, index=False)
    eval_data.to_csv(held_out, index=False)
    options = {
        "task": "forecasting", "window_size": 3, "n_pred_steps": 1, "target_features": targets, "missing_rate": 0,
        "schema_mapping": {"sample_id_col": "id", "feature_cols": ["tag", "y", "z"]},
        "covariates": {"tag": declaration("known", "categorical")}, "preprocessing": preprocessing,
    }
    train = prepare_for_pypots(history, row_indices=list(range(len(train_data))), **options)
    state = json.loads(json.dumps(train["task_state"], allow_nan=False))
    features = targets + [f"tag@te:{target}" for target in targets] + [c for c in ("y", "z") if c not in targets]
    assert train["registry"].feature_columns == features
    entity = {"y": [13 / 3, 13 / 2, 5], "z": [110 / 3, 130 / 2, 50]}
    global_means = {"y": 8.0, "z": 80.0}
    if preprocessing:
        assert state["scaler_state"]["feature_columns"] == ["y", "z"]
    for role in ("val", "test", "full"):
        result = prepare_for_pypots(
            held_out, task="forecasting", set_type=role, task_state=state, preprocessing=preprocessing, missing_rate=0
        )
        assert result["task_state"] == state
        reg = result["registry"]
        assert reg.feature_columns == features
        reg.validate_task_binding(held_out, reg.prediction_binding(), state)
        for sid, expected in (("a", entity), ("new", {t: [global_means[t]] * 3 for t in targets})):
            entry = reg.get_entries_for_sample(sid)[0]
            encoded = result["X"][entry.window_id, :, len(targets):2 * len(targets)]
            if preprocessing:
                encoded = transform_targets(encoded, state["scaler_state"], targets, inverse=True)
            np.testing.assert_allclose(
                encoded, np.array([expected[target] for target in targets]).T,
            )
        if role == "full":
            entry = reg.get_entries_for_sample("a")[-1]
            encoded = result["X"][entry.window_id, -1, len(targets):2 * len(targets)]
            if preprocessing:
                encoded = transform_targets(encoded, state["scaler_state"], targets, inverse=True)
            literal_na = {"y": 4.5, "z": 55}
            np.testing.assert_allclose(encoded, [literal_na[target] for target in targets])
    changed = copy.deepcopy(state)
    changed["categorical_encoding"][targets[0]]["global_mean"] += 1
    with pytest.raises(ValueError, match="binding"):
        reg.validate_task_binding(held_out, reg.prediction_binding(), changed)


@pytest.mark.parametrize(("kind", "value", "conflict"), [("numeric", 8, 9), ("categorical", "east", "west")])
def test_static_broadcast_does_not_read_beyond_origin(tmp_path, kind, value, conflict):
    history = tmp_path / "history.csv"
    pd.DataFrame({"y": range(7), "capacity": [np.nan] * 4 + [value, np.nan, value]}).to_csv(history, index=False)
    options = {
        "task": "forecasting", "window_size": 2, "n_pred_steps": 1, "target_features": ["y"],
        "covariates": {"capacity": declaration("static", kind)}, "missing_rate": 0,
    }
    result = prepare_for_pypots(history, **options)
    capacity = result["X"][:, :, 1]
    if kind == "numeric":
        assert np.isnan(capacity[:3]).all()
        np.testing.assert_array_equal(capacity[3:], 8)
    else:
        np.testing.assert_allclose(capacity[:3], 14 / 6)
        np.testing.assert_allclose(capacity[3:], 13 / 3)
    pd.DataFrame({"y": range(7), "capacity": [value] * 6 + [conflict]}).to_csv(history, index=False)
    with pytest.raises(ValueError, match="Static covariate"):
        prepare_for_pypots(history, row_indices=list(range(6)), **options)


def test_forecast_routing_preserves_ingestion_plan_window_mapping(tmp_path):
    path = tmp_path / "source.csv"
    pd.DataFrame({"flag": ["red", "blue"] * 20, "y": range(40), "z": np.arange(40) * 10}).to_csv(path, index=False)
    roles = {"sample_id_col": None, "timestamp_col": None, "label_col": None, "feature_cols": ["flag", "y", "z"]}
    plan = create_ingestion_plan(
        path, "forecasting", roles=roles, split_policy="row_order", window_size=2,
        horizon=1, targets=["z", "y"], missing_rate=0,
        covariates={"flag": declaration("past", "categorical")},
    )
    assert plan["status"] == "ready", plan
    result = prepare_for_pypots(
        path, task="forecasting", missing_rate=0, ingestion_plan=plan,
    )
    assert result["registry"].feature_columns == ["z", "y", "flag@te:z", "flag@te:y"]
    assert list(result["X"].shape) == plan["preview"]["train"]["X_shape"]
    assert result["X"].shape[-1] == 4
    with pytest.raises(ValueError, match="Plan/covariate override"):
        prepare_for_pypots(path, task="forecasting", missing_rate=0, ingestion_plan=plan, covariates={})


def test_short_sample_is_rejected_before_compiling_leads(tmp_path, monkeypatch):
    from ai4ts.data_protocols import task_data

    history = tmp_path / "history.csv"
    pd.DataFrame({"y": range(6), "promo": range(6)}).to_csv(history, index=False)
    monkeypatch.setattr(task_data, "forecast_sample", lambda *args: pytest.fail("compiled before checking the window"))
    with pytest.raises(ValueError, match="Sample too short"):
        prepare_for_pypots(
            history, task="forecasting", window_size=10**9, n_pred_steps=10**9, target_features=["y"],
            covariates={"promo": declaration("known", "numeric")},
        )
