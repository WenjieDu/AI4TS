"""Bounded, provider-free CSV ingestion plans; no fitting or tensor preparation."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import _read_source, _determine_strategy, split_source_rows, validate_source_split
from .registry import file_sha256

TASKS = ("imputation", "classification", "forecasting", "clustering", "anomaly_detection", "representation")
INDEPENDENT = ("classification", "clustering", "representation")
MAX_CSV_BYTES = 32 * 1024 * 1024
MAX_FEATURE_CELLS = 10_000_000


def content_sha256(value):
    """Hash finite JSON data, preserving ordered lists (not a signature)."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def inspect_csv(csv_path):
    """Inspect a UTF-8 wide CSV and return bounded role evidence without confirming it.

    Parameters
    ----------
    csv_path : str or Path
        Unmodified source CSV.

    Returns
    -------
    dict
        Source identity, candidate roles, and evidence. Candidates are not consent.
    """
    path = Path(csv_path)
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("CSV exceeds 32 MiB; reduce input before planning")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader, [])
        if not columns or any(not c.strip() for c in columns) or len(set(columns)) != len(columns):
            raise ValueError("CSV must have unique nonempty columns")
        n = 0
        for row in reader:
            if len(row) != len(columns):
                raise ValueError("CSV rows must have the header width")
            n += 1
            if n * len(columns) > MAX_FEATURE_CELLS:
                raise ValueError("Source cells exceed 10000000; reduce input")
    if not n:
        raise ValueError("CSV must contain data")
    frame = pd.read_csv(path)
    candidates = {}
    for column in columns:
        name = column.casefold()
        numeric = pd.api.types.is_numeric_dtype(frame[column])
        if name in ("sample_id", "device_id", "sensor_id", "id", "sensor_number", "sensor_no"):
            role, reason = "sample_id_col", "identifier name; confirm independent sample versus continuous entity"
        elif name in ("timestamp", "time", "datetime", "date"):
            role, reason = "timestamp_col", "time name; confirm format and sampling meaning"
        elif "claf_target" in name or name in ("label", "class", "target", "status", "anomaly"):
            role, reason = "label_or_target", "possible task truth; confirm meaning before selecting features"
        elif numeric and (frame[column].nunique() == len(frame) and frame[column].is_monotonic_increasing):
            role, reason = "feature_or_index", "unique increasing numbers may be a row index"
        elif numeric and set(frame[column].dropna().unique()).issubset({0, 1}):
            role, reason = "feature_or_label", "binary values may be a label"
        elif numeric:
            role, reason = "feature_cols", "numeric observation candidate; explicit feature selection required"
        else:
            role, reason = "passthrough", "not a numeric feature; preserve without encoding"
        candidates[column] = {"role": role, "reason": reason}
    return {
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": n,
        "columns": columns,
        "candidates": candidates,
    }


def _groups(source, roles):
    column = roles.get("sample_id_col")
    return list(source.groupby(column, sort=True)) if column else [(0, source)]


def _sampling(source, roles):
    column = roles.get("timestamp_col")
    result = []
    for sid, group in _groups(source, roles):
        step = None
        if column:
            raw = group[column]
            if pd.to_numeric(raw, errors="coerce").notna().all():
                raise ValueError("Numeric timestamps need explicit upstream date/unit conversion")
            times = pd.to_datetime(raw, format="mixed", utc=True)
            differences = times.diff().dropna().dt.total_seconds()
            if len(differences) and differences.nunique() == 1 and differences.iloc[0] > 0:
                step = float(differences.iloc[0])
        result.append({"sample_id": sid, "rows": len(group), "regular_step_seconds": step})
    return result


def _steps(value, sampling, allow_irregular):
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str) and value.isdecimal() and int(value) > 0:
        return int(value)
    if not isinstance(value, str):
        raise ValueError("History/window and horizon require positive steps or an exact duration")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(s|min|h|d)", value)
    if not match:
        raise ValueError("Use positive steps or a duration such as 30min, 24h, 2d")
    frequencies = {item["regular_step_seconds"] for item in sampling}
    if allow_irregular or None in frequencies or len(frequencies) != 1:
        raise ValueError(
            "Duration requires one confirmed regular frequency; use explicit observation steps or preprocess"
        )
    seconds = float(match[1]) * {"s": 1, "min": 60, "h": 3600, "d": 86400}[match[2]]
    count = seconds / frequencies.pop()
    if count <= 0 or not math.isfinite(count) or not count.is_integer():
        raise ValueError("Duration must be an exact positive multiple of the sampling interval")
    return int(count)


def _mapping_digest(entries):
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
    return digest.hexdigest()


def _descriptors(source, roles, selected, task, length, horizon, full):
    for sid, original in _groups(source, roles):
        rows = [int(i) for i in original.index if i in selected]
        if not rows:
            continue
        n = len(rows)
        if task in INDEPENDENT:
            starts = [0]
        elif task == "forecasting":
            starts = range(n - length - horizon + 1)
        else:
            starts = range(0, n, length)
        if task == "forecasting" and full:
            starts = [*starts, n - length]
        positions = {int(row): i for i, row in enumerate(original.index)}
        for start in starts:
            end = min(start + length, n)
            history = rows[start:end]
            future = rows[end : end + horizon] if task == "forecasting" else []
            yield {
                "sample_id": sid,
                "start": start,
                "end": end,
                "rows": history,
                "targets": future,
                "origin": positions[history[-1]] if task == "forecasting" else None,
            }


def registry_mapping_sha256(registry):
    """Compute a source-position digest independently of tensor values."""
    return _mapping_digest(
        {
            "sample_id": e.original_sample_id,
            "start": e.start_step,
            "end": e.end_step,
            "rows": e.row_indices,
            "targets": e.target_row_indices,
            "origin": e.forecast_origin if registry.task_type == "forecasting" else None,
        }
        for e in registry.entries
    )


def _preview(source, roles, task, parameters, split):
    length, horizon = parameters["window_size"], parameters.get("horizon") or 0
    features, targets = len(roles["feature_cols"]), len(parameters.get("targets") or [])
    result = {}
    for role in ("train", "val", "test", "full"):
        if role == "full" and task not in ("imputation", "forecasting"):
            continue
        selected = set(source.index if role == "full" else split["rows"][role])
        groups = [(sid, g.loc[g.index.isin(selected)]) for sid, g in _groups(source, roles)]
        lengths = [len(g) for _, g in groups if len(g)]
        for sid, group in groups:
            if not len(group):
                continue
            minimum = length + (0 if role == "full" else horizon) if task == "forecasting" else 1
            if len(group) < minimum:
                raise ValueError(f"Entity {sid!r} {role} too short: need at least {minimum} rows")
            if task in INDEPENDENT and len(group) > length:
                raise ValueError("Sample exceeds training window; truncation is forbidden")
        windows = sum(
            (
                1
                if task in INDEPENDENT
                else n - length - horizon + 1 + int(role == "full") if task == "forecasting" else math.ceil(n / length)
            )
            for n in lengths
        )
        input_cells = windows * length * features
        target_cells = windows * horizon * targets
        if input_cells + target_cells > MAX_FEATURE_CELLS:
            raise ValueError(f"Prepared window cells exceed 10000000 in {role}; reduce input/window/horizon")
        # Existing task scripts also enforce a conservative source-row expansion bound.
        conservative = (
            len(source) * (length * features + horizon * targets)
            if task not in INDEPENDENT + ("imputation",)
            else input_cells
        )
        if conservative > MAX_FEATURE_CELLS:
            raise ValueError("Prepared window conservative bound exceeds 10000000; reduce input/window/horizon")
        padding = 0 if task == "forecasting" else windows * length - sum(lengths)
        data = source.loc[source.index.isin(selected), roles["feature_cols"]]
        result[role] = {
            "source_rows": sum(lengths),
            "samples": len(lengths),
            "windows": windows,
            "X_shape": [windows, length, features],
            "X_pred_shape": [windows, horizon, targets] if task == "forecasting" else None,
            "target_values": target_cells,
            "unknown_future_windows": len(lengths) if role == "full" and task == "forecasting" else 0,
            "padding_steps": padding,
            "natural_missing_cells": int(data.isna().sum().sum()),
            "observed_source_cells": int(data.notna().sum().sum()),
            "input_cells": input_cells,
            "array_bytes_estimate": input_cells * (8 * 2 + 1 * 2) + target_cells * 9,
            "mapping_sha256": _mapping_digest(
                _descriptors(source, roles, selected, task, length, horizon, role == "full")
            ),
        }
    return result


def create_ingestion_plan(
    csv_path,
    task,
    roles=None,
    split_policy=None,
    window_size=None,
    horizon=None,
    targets=None,
    n_clusters=None,
    clustering_model="VaDER",
    anomaly_rate=None,
    allow_irregular=False,
    seed=2024,
    missing_rate=0.1,
    confirmed_by=None,
    temporal_result=None,
    reshape_result=None,
    time_selection=None,
    merge_result=None,
):
    """Create a content-bound CSV plan without fitting, masking or preparing tensors.

    Parameters
    ----------
    csv_path : str or Path
        Source wide CSV; never modified.
    task : str
        Explicit task from ``TASKS``.
    roles : dict or None
        Confirmed sample/time/label roles and ordered feature columns.
    split_policy : str or None
        Explicit evaluation intent: entity, chronological, row_order or entity_chronological.
    window_size, horizon : int, str or None
        Positive observation counts or exact durations (e.g. ``24h``).
    targets : list of str or None
        Ordered forecast feature subset.
    n_clusters : int or None
        Confirmed cluster count.
    clustering_model : str
        VaDER or CRLI.
    anomaly_rate : float or None
        Confirmed validation calibration rate.
    allow_irregular : bool
        Explicit observation-index interpretation.
    seed : int
        Reproducible split and missingness seed.
    missing_rate : float
        Input artificial missingness rate; actual mask counts are not previewed.
    confirmed_by : dict or None
        Field-level references to supplied declarations, not authentication.
    temporal_result : str, dict or None
        Explicit verified derived-source reference; inherits exact split rows.
    reshape_result : str, dict or None
        Verified long-to-wide source/roles/units; a new split intent is required.
    time_selection : dict or None
        Opt-in source-bound time representation; only this path creates version 2.
    merge_result : str, dict or None
        Unique verified merge parent; requires a new explicit evaluation intent.

    Returns
    -------
    dict
        Ready, needs_clarification or blocked plan with a finite JSON digest.
    """
    request = dict(
        task=task,
        roles=roles,
        split_policy=split_policy,
        window_size=window_size,
        horizon=horizon,
        targets=targets,
        n_clusters=n_clusters,
        clustering_model=clustering_model,
        anomaly_rate=anomaly_rate,
        allow_irregular=allow_irregular,
        seed=seed,
        missing_rate=missing_rate,
        confirmed_by=confirmed_by,
    )
    if time_selection is not None:
        from .time_basis import create_time_selection_plan

        return create_time_selection_plan(
            csv_path,
            time_selection,
            dict(
                request,
                temporal_result=temporal_result,
                reshape_result=reshape_result,
                **({"merge_result": merge_result} if merge_result is not None else {}),
            ),
        )
    plan = {"version": 1, "status": "needs_clarification", "request": request, "unresolved": [], "errors": []}
    try:
        plan["source"] = inspect_csv(csv_path)
        derived = None
        if sum(ref is not None for ref in (temporal_result, reshape_result, merge_result)) > 1:
            raise ValueError("Choose one top-level derived reference")
        if any(ref is not None for ref in (temporal_result, reshape_result, merge_result)):
            from .time_basis import _derived

            key = (
                "merge_result"
                if merge_result is not None
                else "temporal_result" if temporal_result is not None else "reshape_result"
            )
            derived = _derived(csv_path, roles, temporal_result, reshape_result, merge_result)
            roles = derived["roles"]
            request[key] = derived["reference"]
            plan[key] = derived["reference"]
            if derived["feature_semantics"]:
                plan["feature_semantics"] = derived["feature_semantics"]
            if derived["split"]:
                if split_policy is not None and split_policy != derived["split"]["policy"]:
                    raise ValueError("Inherited split policy override conflict")
                split_policy = derived["split"]["policy"]
        if task not in TASKS:
            raise ValueError("Unsupported task")
        if type(seed) is not int or type(allow_irregular) is not bool or not 0 <= missing_rate < 1:
            raise ValueError("Invalid seed, time interpretation or missing rate")
        if roles is None:
            plan["unresolved"].append(
                "Confirm sample/entity, time, ordered feature and label/target roles; see column candidates"
            )
        if split_policy is None:
            plan["unresolved"].append("Confirm independent/new-entity evaluation versus the same entities' future")
        if task in ("forecasting", "anomaly_detection") and window_size is None:
            plan["unresolved"].append("Confirm history/window length")
        if task == "forecasting" and (horizon is None or not targets):
            plan["unresolved"].append("Confirm horizon and ordered forecast targets")
        if task == "clustering" and (type(n_clusters) is not int or n_clusters < 2):
            plan["unresolved"].append("Confirm cluster count >=2")
        if task == "anomaly_detection" and anomaly_rate is None:
            plan["unresolved"].append("Confirm validation calibration anomaly_rate")
        if plan["unresolved"]:
            plan["sha256"] = content_sha256(plan)
            return plan
        source, profile = _read_source(csv_path, task, roles, allow_irregular)
        roles = {
            k: getattr(profile.schema_mapping, k)
            for k in ("sample_id_col", "timestamp_col", "label_col", "feature_cols")
        }
        if task in INDEPENDENT and not roles["sample_id_col"]:
            raise ValueError("Independent samples require confirmed sample identifiers, not continuous stream windows")
        if task == "classification" and not roles["label_col"]:
            raise ValueError("Classification requires a confirmed sample label")
        if task == "forecasting" and (
            roles["label_col"]
            or not targets
            or len(set(targets)) != len(targets)
            or not set(targets).issubset(roles["feature_cols"])
        ):
            raise ValueError("Forecast targets must be an ordered numeric feature subset, not a label")
        if clustering_model not in ("VaDER", "CRLI"):
            raise ValueError("Unsupported clustering model")
        if task == "anomaly_detection" and (not np.isfinite(anomaly_rate) or not 0 < anomaly_rate < 1):
            raise ValueError("anomaly_rate must be in (0,1)")
        sampling = _sampling(source, roles)
        split = (
            derived["split"]
            if derived and derived["split"]
            else split_source_rows(csv_path, roles, seed, allow_irregular, policy=split_policy)
        )
        validate_source_split(source, roles, split, task, split["rows"]["train"], strict=True)
        training = source.loc[source.index.isin(split["rows"]["train"])]
        if training[roles["feature_cols"]].isna().all().any():
            raise ValueError("Training features must contain observed values")
        lengths = [len(g) for _, g in _groups(training, roles)]
        length = (
            _steps(window_size, sampling, allow_irregular)
            if window_size is not None
            else (max(lengths) if task in INDEPENDENT else _determine_strategy(lengths, False, None)[1])
        )
        steps = _steps(horizon, sampling, allow_irregular) if task == "forecasting" else None
        if task in INDEPENDENT and roles["label_col"]:
            from .task_data import _labels

            labels = _labels(source, roles["sample_id_col"], roles["label_col"])
            if task in ("classification", "representation"):
                train_ids = {sid for sid, _ in _groups(training, roles)}
                known = {labels[sid] for sid in train_ids}
                if set(labels.values()) - known:
                    raise ValueError("Unknown training class in validation/test")
        if task == "anomaly_detection" and roles["label_col"]:
            truth = pd.to_numeric(source[roles["label_col"]].replace("", np.nan), errors="raise")
            if not truth.dropna().isin([0, 1]).all():
                raise ValueError("Anomaly truth must be point labels 0/1 or unavailable")
        parameters = dict(
            window_size=length,
            horizon=steps,
            targets=targets,
            n_clusters=n_clusters,
            clustering_model=clustering_model,
            anomaly_rate=anomaly_rate,
            seed=seed,
            missing_rate=missing_rate,
            allow_irregular=allow_irregular,
        )
        plan.update(
            roles=roles,
            parameters=parameters,
            split=split,
            sampling=sampling,
            time_basis="regular_timestamp" if roles["timestamp_col"] and not allow_irregular else "observation_index",
            window_policy=(
                "whole_sample_padding"
                if task in INDEPENDENT
                else "forecast_stride_1" if task == "forecasting" else "non_overlapping_padding"
            ),
            confirmation=confirmed_by
            or {"roles": "explicit declaration", "split_policy": "explicit evaluation intent"},
            preview=_preview(source, roles, task, parameters, split),
            status="ready",
        )
    except (ValueError, TypeError, KeyError, UnicodeError, pd.errors.ParserError) as error:
        plan.update(status="blocked", errors=[str(error)])
    plan["sha256"] = content_sha256(plan)
    return plan


def validate_ingestion_plan(plan, csv_path):
    """Recompute a ready plan from the actual source; a hash alone is insufficient."""
    if (
        not isinstance(plan, dict)
        or type(plan.get("version")) is not int
        or plan["version"] not in (1, 2)
        or plan.get("status") != "ready"
    ):
        raise ValueError("Plan must be a supported ready plan with no unresolved semantics")
    data = {k: v for k, v in plan.items() if k not in ("sha256", "execution_identity")}
    if plan.get("sha256") != content_sha256(data):
        raise ValueError("Plan content fingerprint mismatch")
    actual = create_ingestion_plan(csv_path, **plan["request"])
    if actual != {k: v for k, v in plan.items() if k != "execution_identity"}:
        raise ValueError("Plan/source/roles/split/window preview mismatch; generate a new plan")
    return plan


def verify_prepared_plan(plan, role, result):
    """Compare real prepared tensors/registry with the preview before model fitting."""
    expected = plan["preview"][role]
    registry = result["registry"]
    target_shape = list(result["X_pred"].shape) if "X_pred" in result else None
    if (
        list(result["X"].shape) != expected["X_shape"]
        or target_shape != expected["X_pred_shape"]
        or sum(e.pad_length for e in registry.entries) != expected["padding_steps"]
        or registry_mapping_sha256(registry) != expected["mapping_sha256"]
    ):
        raise ValueError(f"Prepared {role} shape/window/padding/source mapping differs from plan")


def render_ingestion_preview(plan):
    """Render bounded human-readable facts; actual masks and model quality are unknown."""
    lines = [f"# CSV ingestion: {plan['status']}", "", f"Task: {plan['request']['task']}"]
    if "source" in plan:
        lines += [f"Source rows: {plan['source']['rows']}; SHA256: {plan['source']['sha256']}"]
        lines += [
            f"- {name}: {facts['role']} — {facts['reason']}"
            for name, facts in list(plan["source"]["candidates"].items())[:40]
        ]
    lines += [f"- Required: {item}" for item in plan["unresolved"]]
    lines += [f"- Blocked: {item}" for item in plan["errors"]]
    if plan.get("version") == 2 and plan.get("time_selection"):
        selected = plan["time_selection"]
        lines += [
            "",
            f"Confirmed time representation: {selected['kind']}",
            f"Reason/consequences: {selected['reason']}",
            f"Origin source: {selected['origin_source_sha256']}; current input: {selected['input_sha256']}",
            "History/horizon count records. Future physical timestamps and model quality remain unverified.",
            "No fitted scaler/model/threshold is inherited. No natural/new missing cell becomes evaluation truth.",
        ]
        if plan.get("next_step"):
            lines += [f"Preparation required: {json.dumps(plan['next_step'],ensure_ascii=False)}"]
    if plan["status"] == "ready":
        lines += [
            "",
            f"Confirmed roles: {json.dumps(plan['roles'], ensure_ascii=False)}",
            f"Evaluation: {plan['split']['policy']}; time basis: {plan['time_basis']}",
            f"Window policy: {plan['window_policy']}; L={plan['parameters']['window_size']}; H={plan['parameters']['horizon']}",
            "Default L is an engineering rule, not an inferred signal period. Duration means observation count, not endpoint span.",
            "",
            "|Split|Windows|X shape|Future shape|Padding steps|Natural missing cells|Estimated array bytes|",
            "|---|---:|---|---|---:|---:|---:|",
        ]
        for role, facts in plan["preview"].items():
            lines.append(
                f"|{role}|{facts['windows']}|{facts['X_shape']}|{facts['X_pred_shape']}|{facts['padding_steps']}|{facts['natural_missing_cells']}|{facts['array_bytes_estimate']}|"
            )
        lines += [
            "",
            "Full-source inference is separate from held-out evaluation; terminal unknown futures are unscored.",
            "Natural missing, eligible observed truth and padding are distinct. Actual artificial mask counts are recorded only after preparation.",
            "Array bytes are an estimate for float64 inputs/truth and boolean masks; model/framework peak memory and training time are unknown.",
            "No model/scaler fitting, artificial masking, prediction or resampling occurred. Full per-entity sampling facts and mappings are in plan.json.",
        ]
    return "\n".join(lines) + "\n"
