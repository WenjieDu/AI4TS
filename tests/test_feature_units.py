"""Tests for feature-unit declarations, inheritance, and registry bindings."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import pytest

from ai4ts.data_protocols.pipeline import prepare_for_pypots
from ai4ts.data_protocols.preprocessing import state_sha256
from ai4ts.data_protocols.registry import WindowRegistry, file_sha256
from ai4ts.data_protocols.semantics import prepare_feature_semantics, validate_semantic_binding


def test_partial_units_source_and_training_inheritance():
    state = prepare_feature_semantics({"a": "ug/m3"}, ["a", "b"], "a" * 64)
    assert state["units"] == {"a": "ug/m3", "b": "unknown"}
    assert state["declaration"]["origin"] == "user-confirmed"
    inherited = prepare_feature_semantics(None, ["a", "b"], "b" * 64, state)
    assert inherited["units"] == state["units"]
    assert inherited["source_sha256"] == "b" * 64
    assert inherited["inherited_from_sha256"] == state_sha256(state)
    assert prepare_feature_semantics({"a": "ug/m3"}, ["a", "b"], "b" * 64, state)["units"] == state["units"]
    with pytest.raises(ValueError, match="saved training"):
        prepare_feature_semantics({"a": "mg/m3"}, ["a", "b"], "b" * 64, state)
    assert prepare_feature_semantics(None, ["a", "b"], "a" * 64) is None


@pytest.mark.parametrize(
    "units",
    [
        [],
        {"z": "m"},
        {"a": None},
        {"a": ""},
        {"a": "m\n"},
        {"a": "m\x00"},
        {"a": "x" * 65},
        {"a": 42},
        {"units": {"a": "m"}, "origin": "source", "reference": "author"},
        {"units": {"a": "m"}, "origin": "source", "source_sha256": "b" * 64, "reference": "author"},
    ],
)
def test_invalid_declarations_fail(units):
    with pytest.raises(ValueError):
        prepare_feature_semantics(units, ["a", "b"], "a" * 64)


@pytest.mark.parametrize(
    "task", ["imputation", "forecasting", "classification", "clustering", "anomaly_detection", "representation"]
)
def test_semantic_pipeline_does_not_change_numbers_and_rejects_tampering(tmp_path, task):
    source = tmp_path / "source.csv"
    source.write_text("id,a,b,label\n" + "".join(f"{i//8},{i},{i*2},{i//8%2}\n" for i in range(32)))
    roles = {"sample_id_col": "id", "feature_cols": ["a", "b"]}
    if task in ("classification", "clustering", "representation"):
        roles["label_col"] = "label"
    opts = dict(
        task=task,
        window_size=4 if task == "forecasting" else 8,
        schema_mapping=roles,
        preprocessing="standard-v1",
        missing_rate=0.25,
    )
    if task == "forecasting":
        opts.update(n_pred_steps=2, target_features=["b", "a"])
    raw = prepare_for_pypots(source, **opts)
    declared = prepare_for_pypots(source, feature_units={"a": "m", "b": "s"}, **opts)
    for key in ("X", "X_ori", "valid_mask", "indicating_mask", "X_pred", "y", "anomaly_y"):
        if key in raw:
            np.testing.assert_equal(raw[key], declared[key])
    assert declared["scaler_state"] == raw["scaler_state"]
    registry = declared["registry"]
    assert WindowRegistry.from_dict(registry.to_dict()).prediction_binding() == registry.prediction_binding()
    damaged = copy.deepcopy(registry.to_dict())
    damaged["feature_semantics"]["units"]["a"] = "km"
    with pytest.raises(ValueError, match="binding"):
        WindowRegistry.from_dict(damaged)
    for key in ("feature_semantics", "feature_semantics_sha256"):
        damaged = copy.deepcopy(registry.to_dict())
        del damaged[key]
        with pytest.raises(ValueError):
            WindowRegistry.from_dict(damaged)
    if task != "imputation":
        test = prepare_for_pypots(source, set_type="test", task_state=declared["task_state"], **opts)
        assert test["feature_semantics"]["units"] == {"a": "m", "b": "s"}
        with pytest.raises(ValueError, match="saved training"):
            prepare_for_pypots(
                source, set_type="test", task_state=declared["task_state"], feature_units={"a": "km"}, **opts
            )
    else:
        test = prepare_for_pypots(
            source,
            set_type="test",
            scaler_state=declared["scaler_state"],
            unit_state=declared["feature_semantics"],
            **opts,
        )
        assert test["feature_semantics"]["units"] == {"a": "m", "b": "s"}
    validate_semantic_binding(
        test["feature_semantics"], state_sha256(test["feature_semantics"]), ["a", "b"], file_sha256(source)
    )
    with pytest.raises(ValueError, match="order"):
        validate_semantic_binding(test["feature_semantics"], state_sha256(test["feature_semantics"]), ["b", "a"])
