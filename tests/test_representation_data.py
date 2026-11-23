"""Tests for representation preparation and bound sample-vector export."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import pytest

from ai4ts.data_protocols.pipeline import prepare_for_pypots
from ai4ts.data_protocols.task_data import export_representations


@pytest.mark.parametrize("labelled", [True, False])
def test_representation_isolated_mapping_padding_export_and_rejections(tmp_path, labelled):
    source = tmp_path / "source.csv"
    source.write_text("id,label,a,b\n001,zebra,1,2\n001,zebra,3,4\n002,ant,5,6\n002,ant,7,8\n003,zebra,100,200\n")
    roles = dict(sample_id_col="id", label_col="label" if labelled else None, feature_cols=["a", "b"])
    options = dict(task="representation", schema_mapping=roles, preprocessing="standard-v1", missing_rate=0.1)
    train = prepare_for_pypots(source, row_indices=[0, 1, 2, 3], **options)
    state = train["task_state"]
    val = prepare_for_pypots(source, set_type="val", row_indices=[4], task_state=state, **options)
    assert val["registry"].entries[0].original_sample_id == "003" and val["X"].shape == (1, 2, 2)
    assert not val["valid_mask"][0, 1].any() and np.isnan(val["X"][0, 1]).all()
    assert ("y" in val) == labelled and ("classes" in state) == labelled
    if labelled:
        assert state["classes"] == ["ant", "zebra"]
        np.testing.assert_equal(train["y"], [1, 0])
        np.testing.assert_equal(val["y"], [1])
    reg = val["registry"]
    vectors = np.ones((1, 8))
    table = export_representations(
        vectors, reg, source, reg.prediction_binding(), state, np.array([[0.2, 0.8]]) if labelled else None
    )
    assert table.sample_id.tolist() == ["003"] and table.valid_length.tolist() == [1]
    if labelled:
        assert table.predicted_class.tolist() == ["zebra"]
    with pytest.raises(ValueError, match="representations"):
        export_representations(np.full((1, 8), np.nan), reg, source, reg.prediction_binding(), state)
    wrong = copy.deepcopy(state)
    wrong["roles"]["feature_cols"].reverse()
    with pytest.raises(ValueError):
        export_representations(vectors, reg, source, reg.prediction_binding(), wrong)
    if labelled:
        source.write_text(source.read_text().replace("003,zebra", "003,unknown"))
        with pytest.raises(ValueError, match="Unknown"):
            prepare_for_pypots(source, set_type="val", row_indices=[4], task_state=state, **options)
