"""Tests for preparing and reconstructing PyPOTS inputs."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from ai4ts.data_protocols.registry import WindowEntry, WindowRegistry, file_sha256
from ai4ts.data_protocols.pipeline import prepare_for_pypots, split_by_sample_id, split_source_rows
import pandas as pd

ROLES = {"sample_id_col": "id", "timestamp_col": "time", "feature_cols": ["x"], "label_col": "CLAF_TARGET"}


@pytest.mark.parametrize("header", ["x,x", "x,"])
def test_ambiguous_raw_csv_headers_rejected(tmp_path, header):
    source = tmp_path / "source.csv"
    source.write_text(header + "\n1,2\n3,4\n")
    with pytest.raises(ValueError, match="unique nonempty columns"):
        prepare_for_pypots(source)


def test_na_like_identifiers_remain_original_strings(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("id,x\nNA,1\nNA,2\nNULL,3\nNULL,4\n001,5\n001,6\n")
    roles = {"sample_id_col": "id", "feature_cols": ["x"]}
    prepared = prepare_for_pypots(source, schema_mapping=roles)
    assert prepared["registry"].original_sample_ids == ["001", "NA", "NULL"]
    assert all(len(rows) == 2 for rows in split_source_rows(source, roles)["rows"].values())
    source.write_text("id,x\n,1\n,2\n")
    with pytest.raises(ValueError, match="identifiers cannot be missing"):
        prepare_for_pypots(source, schema_mapping=roles)


def test_confirmed_roles_sorting_and_passthrough(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text(
        "id,time,x,CLAF_TARGET,note\n002,2024-01-02,2,A,two\n001,2024-01-02,,B,tail\n"
        "001,2024-01-01,1,B,first\n002,2024-01-01,4,A,start\n"
    )
    result = prepare_for_pypots(source, schema_mapping=ROLES)
    reg = result["registry"]
    assert "y" not in result
    assert reg.original_sample_ids == ["001", "002"]
    assert reg.entries[0].row_indices == [2, 1]
    np.testing.assert_equal(result["X_ori"][0, :, 0], [1, np.nan])
    assert reg.column_roles == ROLES
    no_time = dict(ROLES, timestamp_col=None)
    assert prepare_for_pypots(source, schema_mapping=no_time)["registry"].ordering == "row_order"


@pytest.mark.parametrize(
    "rows,message",
    [
        ("001,bad,1,A,a\n001,2024-01-02,2,A,b\n", None),
        ("001,2024-01-01,1,A,a\n001,2024-01-01,2,A,b\n", "Duplicate"),
        ("001,2024-01-01,,A,a\n001,2024-01-02,,A,b\n", "observed"),
        ("001,2024-01-01,inf,A,a\n001,2024-01-02,2,A,b\n", "infinite"),
    ],
)
def test_invalid_source_rejected(tmp_path, rows, message):
    path = tmp_path / "input.csv"
    path.write_text("id,time,x,CLAF_TARGET,note\n" + rows)
    with pytest.raises(ValueError, match=message):
        prepare_for_pypots(path, schema_mapping=ROLES)


def test_irregular_sampling_requires_consent(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("id,time,x,CLAF_TARGET\n001,2024-01-01,1,A\n001,2024-01-02,2,A\n001,2024-01-04,3,A\n")
    with pytest.raises(ValueError, match="Irregular"):
        prepare_for_pypots(path, schema_mapping=ROLES)
    assert prepare_for_pypots(path, schema_mapping=ROLES, allow_irregular=True)["registry"].ordering == "timestamp"


def test_splits_precede_windows(tmp_path):
    path = tmp_path / "input.csv"
    pd.DataFrame({"id": ["001"] * 12, "x": range(12)}).to_csv(path, index=False)
    roles = {"sample_id_col": "id", "feature_cols": ["x"]}
    split = split_source_rows(path, roles)
    assert split["policy"] == "row_order"
    rows = split["rows"]
    assert max(rows["train"]) < min(rows["val"]) < min(rows["test"])
    assert sorted(sum(rows.values(), [])) == list(range(12))
    result = prepare_for_pypots(path, schema_mapping=roles, row_indices=rows["train"], split_info=split)
    assert result["registry"].source_row_count == 12
    assert result["registry"].split_info == split
    assert set(sum([e.row_indices for e in result["registry"].entries], [])) == set(rows["train"])
    for n in (3, 5):
        pd.DataFrame({"x": range(n)}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="too short"):
            split_source_rows(path)
    for n in (2, 3):
        frame = pd.DataFrame({"id": np.repeat(range(n), 5), "x": range(n * 5)})
        frame.to_csv(path, index=False)
        if n == 2:
            with pytest.raises(ValueError, match="Two entities"):
                split_source_rows(path, roles)
        else:
            entity_split = split_source_rows(path, roles)
            assert all(len(r) == 5 for r in entity_split["rows"].values())
            assert all(len(ids) == 1 for ids in split_by_sample_id(frame, sample_id_col="id"))


def test_fixed_features_windows_and_mask_categories(tmp_path):
    train, val = tmp_path / "train.csv", tmp_path / "val.csv"
    pd.DataFrame({"a": range(12), "b": range(12)}).to_csv(train, index=False)
    pd.DataFrame({"b": [9, 8, 7, 6, 5], "a": [1, np.nan, 3, 4, 5]}).to_csv(val, index=False)
    roles = {"feature_cols": ["a", "b"]}
    prepared = prepare_for_pypots(train, window_size=8, schema_mapping=roles)
    first = prepare_for_pypots(val, window_size=8, schema_mapping=roles, set_type="val", missing_rate=0.5)
    again = prepare_for_pypots(val, window_size=8, schema_mapping=roles, set_type="val", missing_rate=0.5)
    assert prepared["X"].shape == (2, 8, 2)
    assert first["X"].shape == (1, 8, 2)
    assert first["registry"].feature_columns == ["a", "b"]
    np.testing.assert_equal(first["X_ori"][0, :5, 0], [1, np.nan, 3, 4, 5])
    np.testing.assert_equal(first["X"], again["X"])
    assert not first["indicating_mask"][0, 1, 0]  # natural missingness
    assert not first["valid_mask"][0, 5:].any()  # padding
    assert not first["indicating_mask"][0, 5:].any()
    full = prepare_for_pypots(val, window_size=8, schema_mapping=roles, set_type="full")
    assert not full["indicating_mask"].any()
    with pytest.raises(ValueError, match="Column roles"):
        prepare_for_pypots(val, schema_mapping={"feature_cols": ["missing", "a"]})


def test_registry_legacy_and_semantic_binding(tmp_path):
    legacy = {
        "window_size": 2,
        "n_features": 1,
        "entries": [{"window_id": 0, "original_sample_id": 1, "start_step": 0, "end_step": 2}],
    }
    assert WindowRegistry.from_dict(legacy).format_version == 1
    source = tmp_path / "source.csv"
    source.write_text("id,x\n001,1\n001,\n", encoding="utf-8")
    registry = WindowRegistry(
        entries=[WindowEntry(0, "001", 0, 2, row_indices=[0, 1])],
        window_size=2,
        n_features=1,
        task_type="imputation",
        format_version=2,
        source_sha256=file_sha256(source),
        source_row_count=2,
        columns=["id", "x"],
        column_roles={"sample_id_col": "id", "feature_cols": ["x"]},
        feature_columns=["x"],
    )
    path = tmp_path / "registry.json"
    registry.to_json(path)
    restored = WindowRegistry.from_json(path)
    assert restored.to_dict() == registry.to_dict()
    predictions = np.ones((1, 2, 1))
    restored.validate_binding(predictions, source, registry.prediction_binding())
    other = WindowRegistry.from_dict(registry.to_dict())
    other.entries[0].original_sample_id = "another"
    with pytest.raises(ValueError, match="binding mismatch"):
        other.validate_binding(predictions, source, registry.prediction_binding())
    source.write_text("id,x\n001,2\n001,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="binding mismatch"):
        restored.validate_binding(predictions, source, registry.prediction_binding())
