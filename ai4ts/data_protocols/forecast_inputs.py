"""Model column layout and training-derived covariate transforms for forecasting."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .covariates import _read_csv, validate_future_covariates
from .preprocessing import transform_targets


def categorical_columns(covariates):
    return [
        column for column, item in (covariates.items() if isinstance(covariates, dict) else ())
        if isinstance(item, dict) and item.get("kind") == "categorical"
    ]


def input_layout(features, targets, covariates):
    """List output name, source column, and optional encoding target in model order."""
    layout = [(target, target, None) for target in targets]
    for feature in features:
        if feature in targets:
            continue
        item = covariates[feature]
        if item["kind"] == "categorical":
            layout.extend((f"{feature}@te:{target}", feature, target) for target in targets)
        else:
            name = f"{feature}@lead" if item["role"] == "known" else feature
            layout.append((name, feature, None))
    names = [name for name, _, _ in layout]
    if len(names) != len(set(names)) or any(name != source and name in features for name, source, _ in layout):
        raise ValueError("Generated forecast input names collide with source or model columns")
    return layout


def entity_groups(frame, id_col):
    return frame.groupby(id_col, sort=True) if id_col else [(0, frame)]


def fit_categorical_encoding(frame, id_col, columns, targets):
    """Fit smooth=1 on observed training targets, counting missing labels separately."""
    encoding = {}
    if not columns:
        return encoding
    groups = entity_groups(frame, id_col)
    for target in targets:
        global_mean = float(frame[target].mean())
        entities = {}
        for sid, group in groups:
            observed = group.loc[group[target].notna()]
            mean = float(observed[target].mean()) if len(observed) else global_mean
            category_tables = {}
            for column in columns:
                totals = observed.groupby(column, dropna=False)[target].agg(["sum", "count"])
                means = (mean + totals["sum"]) / (1 + totals["count"])
                category_tables[column] = {
                    "values": {str(label): float(value) for label, value in means.items() if not pd.isna(label)},
                    "missing": next((float(value) for label, value in means.items() if pd.isna(label)), None),
                }
            entities[str(sid)] = {"mean": mean, "columns": category_tables}
        encoding[target] = {"global_mean": global_mean, "entities": entities}
    return encoding


def _finite_mean(value):
    return type(value) is float and math.isfinite(value)


def validate_categorical_encoding(encoding, columns, targets):
    if not isinstance(encoding, dict) or set(encoding) != (set(targets) if columns else set()):
        raise ValueError("Invalid forecast categorical encoding targets")
    for target in encoding.values():
        if (
            not isinstance(target, dict)
            or not _finite_mean(target.get("global_mean"))
            or not isinstance(target.get("entities"), dict)
            or not target["entities"]
            or any(not isinstance(sid, str) for sid in target["entities"])
        ):
            raise ValueError("Invalid forecast categorical encoding target statistics")
        for entity in target["entities"].values():
            if (
                not isinstance(entity, dict)
                or not _finite_mean(entity.get("mean"))
                or not isinstance(entity.get("columns"), dict)
                or set(entity["columns"]) != set(columns)
            ):
                raise ValueError("Invalid forecast categorical encoding entity statistics")
            for table in entity["columns"].values():
                if (
                    not isinstance(table, dict)
                    or set(table) != {"values", "missing"}
                    or not isinstance(table.get("values"), dict)
                    or any(not isinstance(k, str) or not _finite_mean(v) for k, v in table["values"].items())
                    or (table.get("missing") is not None and not _finite_mean(table["missing"]))
                ):
                    raise ValueError("Invalid forecast categorical encoding lookup")


def _encode(values, target_state, sid, column):
    entity = target_state["entities"].get(str(sid))
    if entity is None:
        return np.full(len(values), target_state["global_mean"])
    mean = entity["mean"]
    lookup = entity["columns"][column]
    labels = values.astype(object)
    missing = mean if lookup["missing"] is None else lookup["missing"]
    return labels.map(lookup["values"]).fillna(mean).mask(labels.isna(), missing).to_numpy(dtype=float)


def prepare_live_future(history, roles, targets, horizon, covariates, future_csv, frequency):
    """Validate the external horizon against the selected history before using any values."""
    timestamp = roles["timestamp_col"]
    if not timestamp:
        raise ValueError("A timestamp column is required to validate future covariates")
    future, sha256 = _read_csv(future_csv)
    user_roles = {col: item for col, item in covariates.items() if item.get("evidence", {}).get("origin") != "default"}
    validated = validate_future_covariates(
        history, future, sample_id_col=roles["sample_id_col"], timestamp_col=timestamp,
        feature_columns=roles["feature_cols"], target_features=targets, n_pred_steps=horizon,
        covariates=user_roles, frequency=frequency,
    )
    groups = dict(entity_groups(validated, roles["sample_id_col"]))
    return groups, {
        "source_sha256": sha256,
        "timestamps": {str(sid): [time.isoformat() for time in part[timestamp]] for sid, part in groups.items()},
    }


def forecast_sample(group, sid, layout, covariates, horizon, encoding, scaler, future):
    """Compile one ordered entity from its selected rows and optional final horizon."""
    columns = []
    for name, source, target in layout:
        values = group[source]
        if covariates.get(source, {}).get("role") == "static":
            values = values.ffill()
        if target is not None:
            array = _encode(values, encoding[target], sid, source)
            if scaler is not None:
                array = transform_targets(array[:, None], scaler, [target]).ravel()
        else:
            array = values.to_numpy(dtype=float, na_value=np.nan)
            if name != source:
                tail = np.full(horizon, np.nan)
                if future is not None:
                    tail = pd.to_numeric(future[source], errors="raise").to_numpy(dtype=float, na_value=np.nan)
                    if scaler is not None:
                        tail = transform_targets(tail[:, None], scaler, [source]).ravel()
                array = np.concatenate((array, tail))[horizon:]
        columns.append(array)
    return np.column_stack(columns)
