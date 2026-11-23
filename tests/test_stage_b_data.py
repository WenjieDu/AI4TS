"""Tests for clustering and point-anomaly data preparation."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import pytest

from ai4ts.data_protocols.pipeline import prepare_for_pypots
from ai4ts.data_protocols.task_data import export_clusters, export_anomalies


def test_cluster_reference_labels_not_encoded_and_unlabelled_state(tmp_path):
    p = tmp_path / "samples.csv"
    p.write_text("id,label,x\n001,类别甲,1\n001,类别甲,2\n002,other,3\n002,other,4\n003,unseen,99\n")
    roles = {"sample_id_col": "id", "label_col": "label", "feature_cols": ["x"]}
    options = dict(task="clustering", schema_mapping=roles, preprocessing="standard-v1", missing_rate=0)
    train = prepare_for_pypots(p, row_indices=[0, 1, 2, 3], **options)
    val = prepare_for_pypots(p, row_indices=[4], set_type="val", task_state=train["task_state"], **options)
    assert "classes" not in train["task_state"] and val["scaler_state"] == train["scaler_state"]
    assert train["y"][0].decode() == "类别甲" and val["y"][0] == b"unseen"
    table = export_clusters([2], val["registry"], p, val["registry"].prediction_binding(), val["task_state"])
    assert table.iloc[0].to_dict() == {"sample_id": "003", "cluster": 2, "reference_label": "unseen"}
    bad = copy.deepcopy(val["registry"])
    bad.entries[0].row_indices = [0]
    with pytest.raises(ValueError, match="position|coverage"):
        export_clusters([2], bad, p, bad.prediction_binding(), val["task_state"])
    unlabelled = prepare_for_pypots(
        p, row_indices=[0, 1, 2, 3], **dict(options, schema_mapping={"sample_id_col": "id", "feature_cols": ["x"]})
    )
    assert "y" not in unlabelled


def test_anomaly_point_truth_padding_isolation_and_export(tmp_path):
    p = tmp_path / "stream.csv"
    p.write_text(
        "id,x,anomaly\n000,1,0\n000,2,1\n000,,0\n000,4,1\n000,100,0\n000,200,1\n000,300,0\n000,400,1\n000,500,0\n"
    )
    roles = {"sample_id_col": "id", "feature_cols": ["x"], "label_col": "anomaly"}
    split = {"rows": {"train": [0, 1, 2, 3], "val": [4, 5], "test": [6, 7, 8]}}
    options = dict(
        task="anomaly_detection",
        window_size=3,
        schema_mapping=roles,
        preprocessing="standard-v1",
        missing_rate=0,
        split_info=split,
    )
    train = prepare_for_pypots(p, row_indices=split["rows"]["train"], **options)
    np.testing.assert_equal(train["anomaly_y"], [[0, 1, 0], [1, np.nan, np.nan]])
    assert train["scaler_state"]["mean"] == [7 / 3]
    val = prepare_for_pypots(p, row_indices=[4, 5], set_type="val", task_state=train["task_state"], **options)
    np.testing.assert_equal(val["anomaly_y"], [[0, 1, np.nan]])
    registry = train["registry"]
    registry.validate_task_binding(p, registry.prediction_binding(), train["task_state"])
    predictions = {
        "anomaly_score": np.array([[0.0, 2, np.nan], [2, np.nan, np.nan]]),
        "anomaly_valid_mask": np.array([[1, 1, 0], [1, 0, 0]], dtype=bool),
        "anomaly_detection": np.array([[0, 1, -1], [1, -1, -1]]),
        "threshold": 1.0,
    }
    table = export_anomalies(predictions, registry, p, registry.prediction_binding(), train["task_state"])
    assert table.source_row.tolist() == [0, 1, 2, 3]
    assert table.truth.tolist() == [0, 1, 0, 1] and table.evaluated.tolist() == [True, True, False, True]
    assert table.iloc[2]["status"] == "unscorable" and table.iloc[2]["anomaly"] == -1
    val_prediction = {
        "anomaly_score": np.array([[1.0, 1.0, np.nan]]),
        "anomaly_valid_mask": np.array([[1, 1, 0]], dtype=bool),
        "anomaly_detection": np.array([[0, 0, -1]]),
        "threshold": 2.0,
    }
    exported_val = export_anomalies(
        val_prediction, val["registry"], p, val["registry"].prediction_binding(), val["task_state"]
    )
    assert exported_val.source_row.tolist() == [4, 5] and exported_val.sample_step.tolist() == [4, 5]
    predictions["threshold"] = 3
    with pytest.raises(ValueError, match="threshold"):
        export_anomalies(predictions, registry, p, registry.prediction_binding(), train["task_state"])
    with pytest.raises(ValueError, match="chronological"):
        prepare_for_pypots(
            p, row_indices=[0, 2], **dict(options, split_info={"rows": {"train": [0, 2], "val": [1], "test": [3]}})
        )
