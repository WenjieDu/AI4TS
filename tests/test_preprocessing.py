"""Tests for training-only scaling, scaler state, and semantic export."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from ai4ts.data_protocols.preprocessing import fit_scaler, load_scaler, save_scaler, transform_values, validate_scaler


def test_scaler_matches_reference_and_roundtrip(tmp_path):
    values = np.array([[1, 7], [3, 7], [np.nan, 7], [5, 7]], dtype=float)
    reference = StandardScaler().fit(values)
    state = fit_scaler(values, ["a", "constant"])
    path = tmp_path / "scaler.json"
    save_scaler(state, path)
    restored = load_scaler(path, ["a", "constant"], state["training_sha256"])
    assert restored == state
    assert state["scale"][1] == 1
    np.testing.assert_allclose(transform_values(values, restored), reference.transform(values), equal_nan=True)
    np.testing.assert_allclose(transform_values(transform_values(values, restored), restored, inverse=True), values)
    np.testing.assert_allclose(transform_values(np.zeros((1, 2)), state, inverse=True), reference.mean_[None])


@pytest.mark.parametrize(
    "key,value",
    [("scale", [0, 1]), ("scale", [1]), ("mean", [np.inf, 1]), ("variance", [-1, 0]), ("n_samples_seen", [0, 3])],
)
def test_invalid_scaler_rejected(key, value):
    state = fit_scaler([[1, 2], [3, 4]], ["a", "b"])
    state[key] = value
    with pytest.raises(ValueError):
        validate_scaler(state)


def test_scaler_bindings_and_empty_training():
    state = fit_scaler([[1, 2], [3, 4]], ["a", "b"])
    with pytest.raises(ValueError, match="order"):
        validate_scaler(state, ["b", "a"])
    with pytest.raises(ValueError, match="source"):
        validate_scaler(copy.deepcopy(state), training_sha="0" * 64)
    with pytest.raises(ValueError, match="observed"):
        fit_scaler([[np.nan, 1], [np.nan, 2]], ["a", "b"])


def test_standard_pipeline_training_only_masks_and_semantic_export(tmp_path):
    from ai4ts.data_protocols.pipeline import prepare_for_pypots, reconstruct_semantic_csv, split_source_rows
    from ai4ts.data_protocols.registry import WindowRegistry

    source = tmp_path / "source.csv"
    source.write_text("x,b,note\n" + "\n".join(f"{i if i != 3 else ''},7,keep-{i}" for i in range(100)) + "\n")
    roles = {"feature_cols": ["x", "b"]}
    split = split_source_rows(source, roles)
    options = dict(
        schema_mapping=roles, window_size=8, missing_rate=0.25, preprocessing="standard-v1", split_info=split
    )
    train = prepare_for_pypots(source, row_indices=split["rows"]["train"], **options)
    state = train["scaler_state"]
    assert train["indicating_mask"].any()
    for name in ("val", "test", "full"):
        kwargs = dict(options, set_type=name, row_indices=split["rows"].get(name), scaler_state=state)
        result = prepare_for_pypots(source, **kwargs)
        repeat = prepare_for_pypots(source, **kwargs)
        np.testing.assert_equal(result["X"], repeat["X"])
        assert not (result["indicating_mask"] & ~np.isfinite(result["X_ori"])).any()
        assert not (result["indicating_mask"] & ~result["valid_mask"]).any()
        if name != "full":
            changed = prepare_for_pypots(source, seed=19, **kwargs)
            assert not np.array_equal(result["indicating_mask"], changed["indicating_mask"])
            assert result["registry"].split_info["masking"]["seed"] != train["registry"].split_info["masking"]["seed"]
        else:
            assert not result["indicating_mask"].any()
            registry = WindowRegistry.from_dict(result["registry"].to_dict())
            zero = np.zeros_like(result["X"])
            output = reconstruct_semantic_csv(zero, registry, source, registry.prediction_binding(), state)
            assert output.loc[0, "x"] == "0" and output.loc[3, "note"] == "keep-3"
            assert float(output.loc[3, "x"]) == state["mean"][0]
            altered = copy.deepcopy(state)
            altered["mean"][0] += 1
            with pytest.raises(ValueError, match="binding"):
                reconstruct_semantic_csv(zero, registry, source, registry.prediction_binding(), altered)
            with pytest.raises(ValueError, match="scaler"):
                reconstruct_semantic_csv(zero, registry, source, registry.prediction_binding())
    # Held-out values do not participate in fitting or the training fingerprint.
    text = source.read_text().splitlines()
    for i in split["rows"]["test"] + split["rows"]["val"]:
        text[i + 1] = f"999999,7,keep-{i}"
    source.write_text("\n".join(text) + "\n")
    assert prepare_for_pypots(source, row_indices=split["rows"]["train"], **options)["scaler_state"] == state
    wrong = copy.deepcopy(state)
    wrong["training_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source"):
        prepare_for_pypots(source, set_type="full", scaler_state=wrong, **options)
    with pytest.raises(ValueError, match="saved training"):
        prepare_for_pypots(source, set_type="val", **options)


def test_import_with_encodingless_stdout():
    import contextlib
    import importlib
    import io
    import ai4ts
    with contextlib.redirect_stdout(io.StringIO()):
        importlib.reload(ai4ts)
