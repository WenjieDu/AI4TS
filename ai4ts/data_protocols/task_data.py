"""Training-bound classification and future-target data for the CSV bridge."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .preprocessing import fit_scaler, state_sha256, transform_values, validate_scaler
from .registry import WindowEntry, WindowRegistry, file_sha256
from .semantics import prepare_feature_semantics, validate_semantic_binding


def validate_task_state(state, task=None):
    """Validate portable training state before using its roles or parameters."""
    if (
        not isinstance(state, dict)
        or state.get("version") != 1
        or state.get("task")
        not in ("classification", "forecasting", "clustering", "anomaly_detection", "representation")
    ):
        raise ValueError("Unsupported task state")
    if task is not None and state["task"] != task:
        raise ValueError("Task state/task mismatch")
    features = state.get("roles", {}).get("feature_cols", [])
    if not features or len(set(features)) != len(features):
        raise ValueError("Invalid task state feature order")
    if type(state.get("window_size")) is not int or state["window_size"] < 1:
        raise ValueError("Invalid task state window")
    if state.get("preprocessing") not in (None, "standard-v1"):
        raise ValueError("Invalid task preprocessing")
    if state["preprocessing"]:
        validate_scaler(state.get("scaler_state"), features)
    elif state.get("scaler_state") is not None:
        raise ValueError("Raw task state cannot contain scaler")
    if state["task"] == "classification" or (state["task"] == "representation" and state["roles"].get("label_col")):
        classes = state.get("classes", [])
        if (
            not classes
            or any(not isinstance(c, str) or not c.strip() for c in classes)
            or len(set(classes)) != len(classes)
        ):
            raise ValueError("Invalid training classes")
        if not state["roles"].get("sample_id_col") or not state["roles"].get("label_col"):
            raise ValueError("Classification requires confirmed sample and label roles")
    elif state["task"] == "forecasting":
        targets = state.get("target_features", [])
        if not targets or len(set(targets)) != len(targets) or not set(targets).issubset(features):
            raise ValueError("Invalid ordered forecast targets")
        if type(state.get("n_pred_steps")) is not int or state["n_pred_steps"] < 1:
            raise ValueError("Invalid forecast horizon")
    elif state["task"] in ("clustering", "representation") and not state["roles"].get("sample_id_col"):
        raise ValueError("Clustering/representation requires confirmed independent sample roles")
    if state["task"] == "representation" and not state["roles"].get("label_col") and "classes" in state:
        raise ValueError("Unlabelled representation cannot contain training classes")
    for key in ("training_sha256", "training_source_sha256"):
        value = state.get(key, "")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Invalid training source fingerprint")
    if ("feature_semantics" in state or "feature_semantics_sha256" in state) and (
        state.get("feature_semantics") is None or not state.get("feature_semantics_sha256")
    ):
        raise ValueError("Incomplete task state feature semantics")
    validate_semantic_binding(
        state.get("feature_semantics"), state.get("feature_semantics_sha256"), features,
        state["training_source_sha256"],
    )
    state_sha256(state)
    return state


def _training_hash(df, roles):
    columns = [c for c in (roles["sample_id_col"], roles["timestamp_col"], roles["label_col"]) if c]
    columns += roles["feature_cols"]
    return hashlib.sha256(df[columns].to_csv(index=False).encode()).hexdigest()


def _labels(df, id_col, label_col):
    groups = df.groupby(id_col, sort=True)[label_col]
    if df[label_col].isna().any() or df[label_col].str.strip().eq("").any():
        raise ValueError("Sample labels cannot be missing")
    if groups.nunique().ne(1).any():
        raise ValueError("Conflicting labels within a sample")
    return groups.first().to_dict()


def prepare_task(
    csv_path,
    task,
    window_size,
    missing_rate,
    set_type,
    seed,
    schema_mapping,
    allow_irregular,
    row_indices,
    split_info,
    preprocessing,
    scaler_state,
    task_state,
    n_pred_steps,
    target_features,
    feature_units=None,
    unit_state=None,
):
    """Prepare a confirmed task, fitting state exclusively for training."""
    from .pipeline import _read_source, _extract_samples, _transform, _inject_artificial_missing

    if preprocessing not in (None, "standard-v1") or not 0 <= missing_rate < 1:
        raise ValueError("Invalid task preprocessing or missing rate")
    if task_state is not None:
        validate_task_state(task_state, task)
        if schema_mapping is None:
            schema_mapping = task_state["roles"]
        if window_size is None:
            window_size = task_state["window_size"]
        if task == "forecasting":
            n_pred_steps = task_state["n_pred_steps"] if n_pred_steps is None else n_pred_steps
            target_features = task_state["target_features"] if target_features is None else target_features
        if preprocessing != task_state["preprocessing"]:
            raise ValueError("Task state preprocessing mismatch")
        if scaler_state is not None and scaler_state != task_state["scaler_state"]:
            raise ValueError("Task state/scaler mismatch")
        scaler_state = task_state["scaler_state"]
        saved_units = task_state.get("feature_semantics")
        if unit_state is not None and unit_state != saved_units:
            raise ValueError("Task state/units mismatch")
        unit_state = saved_units
    elif set_type != "train":
        raise ValueError("Classification/forecasting requires saved training task state for val/test/full")
    source, profile = _read_source(csv_path, task, schema_mapping, allow_irregular)
    df = source
    if row_indices is not None:
        if not row_indices or len(set(row_indices)) != len(row_indices) or not set(row_indices).issubset(source.index):
            raise ValueError("Split rows must be nonempty, unique and in source")
        df = source.loc[source.index.isin(row_indices)]
    schema = profile.schema_mapping
    roles = {k: getattr(schema, k) for k in ("sample_id_col", "timestamp_col", "label_col", "feature_cols")}
    features = roles["feature_cols"]
    semantics = prepare_feature_semantics(feature_units, features, file_sha256(csv_path), unit_state)
    id_col, label_col = roles["sample_id_col"], roles["label_col"]
    if set_type == "train" and df[features].isna().all().any():
        raise ValueError("Training features must contain observed values")
    if split_info:
        from .pipeline import validate_source_split

        validate_source_split(source, roles, split_info, task, row_indices, set_type)
    labels = None
    if task in ("classification", "clustering", "representation"):
        if not id_col or (task == "classification" and not label_col):
            raise ValueError("Classification requires confirmed sample and label roles")
        if label_col:
            labels = _labels(df, id_col, label_col)
    samples, ids = _extract_samples(df, id_col, features)
    if task in ("classification", "clustering", "representation"):
        window_size = max(map(len, samples)) if window_size is None else window_size
        if any(len(s) > window_size for s in samples):
            raise ValueError("Sample exceeds training window; truncation is forbidden")
    if type(window_size) is not int or window_size < 1:
        raise ValueError("Task requires a positive history/window size")
    if task == "forecasting":
        if label_col:
            raise ValueError("Forecast targets are numeric feature columns, not a label role")
        if type(n_pred_steps) is not int or n_pred_steps < 1:
            raise ValueError("Forecasting requires a positive horizon")
        if (
            not target_features
            or len(set(target_features)) != len(target_features)
            or not set(target_features).issubset(features)
        ):
            raise ValueError("Forecasting requires a nonempty ordered feature subset")
    training_rows = (split_info or {}).get("rows", {}).get("train")
    training = df if set_type == "train" else source.loc[source.index.isin(training_rows)] if training_rows else None
    if task_state is None:
        if preprocessing:
            fitted = fit_scaler(df[features].to_numpy(), features)
            if scaler_state is not None and scaler_state != fitted:
                raise ValueError("Provided scaler differs from training observations")
            scaler_state = fitted
        elif scaler_state is not None:
            raise ValueError("Scaler requires standard-v1")
        task_state = {
            "version": 1,
            "task": task,
            "roles": roles,
            "window_size": window_size,
            "preprocessing": preprocessing,
            "scaler_state": scaler_state,
            "training_sha256": _training_hash(df, roles),
            "training_source_sha256": file_sha256(csv_path),
        }
        if semantics is not None:
            task_state.update(feature_semantics=semantics, feature_semantics_sha256=state_sha256(semantics))
        if task == "classification" or (task == "representation" and labels is not None):
            task_state["classes"] = sorted(set(labels.values()))
        elif task == "forecasting":
            task_state.update(n_pred_steps=n_pred_steps, target_features=list(target_features))
    validate_task_state(task_state, task)
    if roles != task_state["roles"] or window_size != task_state["window_size"]:
        raise ValueError("Task state feature/role/window mismatch")
    if training is not None and _training_hash(training, roles) != task_state["training_sha256"]:
        raise ValueError("Task state training source mismatch")
    if training is not None and (task == "classification" or (task == "representation" and labels is not None)):
        if sorted(set(_labels(training, id_col, label_col).values())) != task_state["classes"]:
            raise ValueError("Task state training class mapping mismatch")
    if task == "forecasting" and (
        n_pred_steps != task_state["n_pred_steps"] or list(target_features) != task_state["target_features"]
    ):
        raise ValueError("Task state forecast target/horizon mismatch")
    if task == "classification" or (task == "representation" and labels is not None):
        mapping = {c: i for i, c in enumerate(task_state["classes"])}
        if not set(labels.values()).issubset(mapping):
            raise ValueError("Unknown class outside training mapping")
    if preprocessing:
        df = df.copy()
        df[features] = transform_values(df[features].to_numpy(), scaler_state)
        samples, ids = _extract_samples(df, id_col, features)
    source_rows = {sid: list(g.index) for sid, g in df.groupby(id_col, sort=True)} if id_col else {0: list(df.index)}
    original_rows = (
        {sid: list(g.index) for sid, g in source.groupby(id_col, sort=True)} if id_col else {0: list(source.index)}
    )
    original_positions = {
        sid: {row: position for position, row in enumerate(rows)} for sid, rows in original_rows.items()
    }
    if task in ("classification", "clustering", "representation"):
        original, registry = _transform(samples, ids, window_size, "pad_only", True)
        for entry in registry.entries:
            entry.row_indices = source_rows[entry.original_sample_id]
        targets = None
    elif task == "anomaly_detection":
        registry = WindowRegistry(window_size=window_size, strategy="sliding_window")
        windows = []
        for sample, sid in zip(samples, ids):
            for start in range(0, len(sample), window_size):
                end = min(start + window_size, len(sample))
                pad = window_size - (end - start)
                registry.add(WindowEntry(len(windows), sid, start, end, bool(pad), pad, source_rows[sid][start:end]))
                window = np.full((window_size, len(features)), np.nan)
                window[: end - start] = sample[start:end]
                windows.append(window)
        original, targets = np.stack(windows), None
    else:
        registry = WindowRegistry(window_size=window_size, strategy="forecast_sliding")
        history, future = [], []
        indices = [features.index(c) for c in target_features]
        for sample, sid in zip(samples, ids):
            if len(sample) < window_size + (0 if set_type == "full" else n_pred_steps):
                raise ValueError("Sample too short for history and future within its split")
            starts = list(range(len(sample) - window_size - n_pred_steps + 1))
            if set_type == "full":
                starts.append(len(sample) - window_size)
            for start in starts:
                end = start + window_size
                target_rows = source_rows[sid][end : end + n_pred_steps]
                entry = WindowEntry(
                    len(history),
                    sid,
                    start,
                    end,
                    row_indices=source_rows[sid][start:end],
                    target_row_indices=target_rows,
                    forecast_origin=original_positions[sid][source_rows[sid][end - 1]],
                )
                registry.add(entry)
                history.append(sample[start:end])
                target = np.full((n_pred_steps, len(indices)), np.nan)
                target[: len(target_rows)] = sample[end : end + len(target_rows), indices]
                future.append(target)
        original, targets = np.stack(history), np.stack(future)
        registry.target_features, registry.n_pred_steps = list(target_features), n_pred_steps
    registry.n_features, registry.task_type, registry.format_version = len(features), task, 2
    registry.source_sha256, registry.source_row_count = file_sha256(csv_path), len(source)
    registry.columns, registry.column_roles, registry.feature_columns = list(df.columns), roles, list(features)
    if semantics is not None:
        registry.feature_semantics, registry.feature_semantics_sha256 = semantics, state_sha256(semantics)
    registry.ordering = "timestamp" if schema.timestamp_col else "row_order"
    registry.split_info = dict(split_info or {}, role=set_type)
    if "rows" not in registry.split_info:
        registry.split_info["rows"] = {set_type: list(df.index)}
    registry.split_info["time_basis"] = (
        "regular_timestamp" if schema.timestamp_col and not allow_irregular else "observation_index"
    )
    registry.task_state_sha256 = state_sha256(task_state)
    if preprocessing:
        registry.preprocessing, registry.scaler_sha256 = preprocessing, state_sha256(scaler_state)
    valid = np.zeros_like(original, dtype=bool)
    for e in registry.entries:
        valid[e.window_id, : e.actual_length] = True
    split_seed = int(
        np.random.SeedSequence([seed, ("train", "val", "test", "full").index(set_type)]).generate_state(1)[0]
    )
    mask_rate = 0 if set_type == "full" or (set_type == "train" and preprocessing is None) else missing_rate
    inputs = _inject_artificial_missing(original, mask_rate, split_seed)
    registry.split_info["masking"] = {
        "pattern": "point",
        "seed": split_seed,
        "rate": mask_rate,
    }
    result = dict(X=inputs, X_ori=original, valid_mask=valid, registry=registry, profile=profile, task_state=task_state)
    if semantics is not None:
        result["feature_semantics"] = semantics
    if preprocessing:
        result["scaler_state"] = scaler_state
    if labels is not None:
        result["y"] = (
            np.array([mapping[labels[e.original_sample_id]] for e in registry.entries], dtype=np.int64)
            if task in ("classification", "representation")
            else np.array([labels[e.original_sample_id].encode("utf-8") for e in registry.entries])
        )
    if task == "anomaly_detection" and label_col:
        raw = pd.to_numeric(source[label_col].replace("", np.nan), errors="raise")
        if not raw.dropna().isin([0, 1]).all():
            raise ValueError("Anomaly truth must be point labels 0/1 or unavailable")
        truth = np.full(original.shape[:2], np.nan)
        for e in registry.entries:
            truth[e.window_id, : e.actual_length] = raw.loc[e.row_indices].to_numpy()
        result["anomaly_y"] = truth
    if targets is not None:
        target_valid = np.zeros_like(targets, dtype=bool)
        for e in registry.entries:
            target_valid[e.window_id, : len(e.target_row_indices)] = True
        result.update(X_pred=targets, X_pred_valid_mask=target_valid, target_features=list(target_features))
    return result


def export_representations(vectors, registry, source_csv, binding, task_state, probabilities=None):
    """Export finite sample vectors and optional fixed-probe predictions with bound identities."""
    if registry.task_type != "representation":
        raise ValueError("Representation export requires representation registry")
    registry.validate_task_binding(source_csv, binding, task_state)
    vectors = np.asarray(vectors)
    if (
        vectors.ndim != 2
        or vectors.shape[0] != registry.n_windows
        or not vectors.shape[1]
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Invalid bound sample representations")
    records = [
        {
            "sample_id": e.original_sample_id,
            "valid_length": e.actual_length,
            **{f"representation:{j}": float(vectors[i, j]) for j in range(vectors.shape[1])},
        }
        for i, e in enumerate(registry.entries)
    ]
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        classes = task_state.get("classes", [])
        if (
            probabilities.shape != (len(records), len(classes))
            or not classes
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-6)
        ):
            raise ValueError("Invalid bound representation probe probabilities")
        for i, record in enumerate(records):
            record.update(
                predicted_class=classes[int(probabilities[i].argmax())],
                **{f"probability:{c}": float(probabilities[i, j]) for j, c in enumerate(classes)},
            )
    return pd.DataFrame(records)


def export_classifications(probabilities, registry, source_csv, binding, task_state):
    """Return decoded sample predictions after verifying their source binding."""
    if registry.task_type != "classification":
        raise ValueError("Classification export requires classification registry")
    registry.validate_task_binding(source_csv, binding, task_state)
    probabilities = np.asarray(probabilities)
    classes = task_state["classes"]
    if (
        probabilities.shape != (registry.n_windows, len(classes))
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Invalid bound class probabilities")
    return pd.DataFrame(
        [
            {
                "sample_id": e.original_sample_id,
                "predicted_class": classes[int(probabilities[i].argmax())],
                **{f"probability:{c}": float(probabilities[i, j]) for j, c in enumerate(classes)},
            }
            for i, e in enumerate(registry.entries)
        ]
    )


def export_forecasts(predictions, registry, source_csv, binding, task_state):
    """Return raw-scale long forecasts, preserving origins and unknown futures."""
    from .pipeline import _read_source
    from .preprocessing import transform_targets

    if registry.task_type != "forecasting":
        raise ValueError("Forecast export requires forecasting registry")
    registry.validate_task_binding(source_csv, binding, task_state)
    values = np.asarray(predictions)
    if (
        values.shape != (registry.n_windows, registry.n_pred_steps, len(registry.target_features))
        or not np.isfinite(values).all()
    ):
        raise ValueError("Invalid bound future predictions")
    if task_state["preprocessing"]:
        values = transform_targets(values, task_state["scaler_state"], registry.target_features, inverse=True)
    source, _ = _read_source(source_csv, "forecasting", task_state["roles"], True)
    ts_col, id_col = registry.column_roles["timestamp_col"], registry.column_roles["sample_id_col"]
    records = []
    for i, e in enumerate(registry.entries):
        frequency = None
        origin_time = source.at[e.row_indices[-1], ts_col] if ts_col else None
        if ts_col and registry.split_info.get("time_basis") == "regular_timestamp":
            group = source[source[id_col] == e.original_sample_id] if id_col else source
            times = pd.to_datetime(group[ts_col], format="mixed", utc=True)
            differences = times.diff().dropna()
            if len(differences) and differences.nunique() == 1:
                frequency = differences.iloc[0]
        for h in range(registry.n_pred_steps):
            row = e.target_row_indices[h] if h < len(e.target_row_indices) else None
            future_time = source.at[row, ts_col] if row is not None and ts_col else None
            if row is None and frequency is not None:
                future_time = (pd.Timestamp(origin_time) + (h + 1) * frequency).isoformat()
            for j, target in enumerate(registry.target_features):
                truth = source.at[row, target] if row is not None else np.nan
                records.append(
                    {
                        "sample_id": e.original_sample_id,
                        "origin_step": e.forecast_origin,
                        "origin_source_row": e.row_indices[-1],
                        "horizon": h + 1,
                        "target_step": e.forecast_origin + h + 1,
                        "target_source_row": row,
                        "origin_time": origin_time,
                        "target_time": future_time,
                        "time_basis": "regular_timestamp" if frequency is not None else "observation_index",
                        "target": target,
                        "prediction": float(values[i, h, j]),
                        "truth": float(truth) if np.isfinite(truth) else None,
                        "evaluated": bool(np.isfinite(truth)),
                    }
                )
    return pd.DataFrame(records)


def export_clusters(clusters, registry, source_csv, binding, task_state):
    """Export sample IDs and cluster numbers; reference labels never name clusters."""
    if registry.task_type != "clustering":
        raise ValueError("Cluster export requires clustering state")
    registry.validate_task_binding(source_csv, binding, task_state)
    clusters = np.asarray(clusters)
    if (
        clusters.shape != (registry.n_windows,)
        or not np.isfinite(clusters).all()
        or (clusters < 0).any()
        or not np.equal(clusters, np.floor(clusters)).all()
    ):
        raise ValueError("Invalid sample cluster assignments")
    from .pipeline import _read_source

    source, _ = _read_source(source_csv, "clustering", task_state["roles"], True)
    roles = task_state["roles"]
    labels = _labels(source, roles["sample_id_col"], roles["label_col"]) if roles["label_col"] else {}
    return pd.DataFrame(
        [
            {
                "sample_id": e.original_sample_id,
                "cluster": int(clusters[e.window_id]),
                "reference_label": labels.get(e.original_sample_id),
            }
            for e in registry.entries
        ]
    )


def export_anomalies(predictions, registry, source_csv, binding, task_state):
    """Export each real source point, including explicit unscorable positions."""
    if registry.task_type != "anomaly_detection":
        raise ValueError("Anomaly export requires anomaly state")
    registry.validate_task_binding(source_csv, binding, task_state)
    from .pipeline import _read_source

    source, _ = _read_source(source_csv, "anomaly_detection", task_state["roles"], True)
    shape = (registry.n_windows, registry.window_size)
    scores, valid, decisions = [
        np.asarray(predictions[k]) for k in ("anomaly_score", "anomaly_valid_mask", "anomaly_detection")
    ]
    threshold = float(predictions["threshold"])
    if (
        any(v.shape != shape for v in (scores, valid, decisions))
        or not np.isfinite(threshold)
        or threshold < 0
        or not np.isfinite(scores[valid.astype(bool)]).all()
    ):
        raise ValueError("Invalid anomaly score/decision/threshold geometry")
    if not np.array_equal(decisions[valid.astype(bool)], scores[valid.astype(bool)] > threshold):
        raise ValueError("Anomaly decisions differ from saved threshold")
    rows = []
    roles = task_state["roles"]
    groups = source.groupby(roles["sample_id_col"], sort=True) if roles["sample_id_col"] else [(0, source)]
    positions = {row: step for _, group in groups for step, row in enumerate(group.index)}
    for e in registry.entries:
        if valid[e.window_id, e.actual_length :].any():
            raise ValueError("Anomaly padding cannot be scored")
        for t, row in enumerate(e.row_indices):
            available = bool(valid[e.window_id, t])
            label = (
                pd.to_numeric(source.loc[row, roles["label_col"]], errors="coerce") if roles["label_col"] else np.nan
            )
            rows.append(
                {
                    "sample_id": e.original_sample_id,
                    "source_row": row,
                    "sample_step": positions[row],
                    "time": source.loc[row, roles["timestamp_col"]] if roles["timestamp_col"] else None,
                    "score": float(scores[e.window_id, t]) if available else np.nan,
                    "threshold": threshold,
                    "anomaly": int(decisions[e.window_id, t]) if available else -1,
                    "status": "scorable" if available else "unscorable",
                    "truth": label,
                    "evaluated": available and bool(np.isfinite(label)),
                }
            )
    return pd.DataFrame(rows)
