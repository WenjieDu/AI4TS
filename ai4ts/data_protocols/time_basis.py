"""Explicit time meaning, structural preview and opt-in ingestion v2 (no estimators)."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .ingestion import content_sha256, inspect_csv
from .registry import file_sha256
from .temporal_preprocessing import (
    Clarification,
    _duration,
    _source,
    create_temporal_plan,
    declaration,
)


def _derived(csv_path, roles, temporal_result=None, reshape_result=None, merge_result=None):
    if sum(ref is not None for ref in (temporal_result, reshape_result, merge_result)) > 1:
        raise ValueError("Choose one top-level derived source")
    if merge_result is not None:
        from .merge import merge_context

        return merge_context(merge_result, csv_path, roles)
    if temporal_result is not None:
        from .temporal_preprocessing import temporal_context

        return temporal_context(temporal_result, csv_path, roles)
    if reshape_result is not None:
        from .reshape import reshape_context

        return reshape_context(reshape_result, csv_path, roles)
    return None


def _grid_support(source, plan):
    step, anchor, times = plan["frequency_ns"], plan["origin_ns"], plan["times_ns"]
    features = plan["roles"]["feature_cols"]
    counts = {feature: dict(finite=0, missing=0, longest_unsupported_run=0) for feature in features}
    for group in plan["groups"]:
        members = {}
        for row in group["retained"]:
            point = (
                times[row] if plan["request"]["mode"] == "regularize" else anchor + (times[row] - anchor) // step * step
            )
            index = (point - group["first"]) // step
            members.setdefault(index, []).append(row)
        for feature in features:
            supported = []
            for index, rows in members.items():
                values = source.loc[rows, feature]
                reducer = (plan["request"]["reducers"] or {}).get(feature)
                finite = bool(pd.notna(values.iloc[-1])) if reducer == "last" else bool(values.notna().any())
                if finite:
                    supported.append(index)
            supported.sort()
            edges = [-1, *supported, group["count"]]
            longest = max(right - left - 1 for left, right in zip(edges, edges[1:]))
            counts[feature]["finite"] += len(supported)
            counts[feature]["missing"] += group["count"] - len(supported)
            counts[feature]["longest_unsupported_run"] = max(counts[feature]["longest_unsupported_run"], longest)
    return dict(
        output_rows=plan["preview"]["output_rows"],
        feature_cells=plan["preview"]["feature_cells"],
        features=counts,
        frequency_ns=step,
        origin_ns=anchor,
        mode=plan["request"]["mode"],
        estimated_csv_bytes_upper_bound=plan["preview"]["estimated_csv_bytes_upper_bound"],
        limits=plan["limits"],
        bucket_rule=plan["preview"]["bucket_rule"],
    )


def preview_time_basis(
    csv_path,
    roles=None,
    *,
    timezone=None,
    frequency=None,
    origin="epoch",
    mode="regularize",
    reducers=None,
    drop_columns=None,
    temporal_result=None,
    reshape_result=None,
    merge_result=None,
):
    """Preview physical intervals and optional fixed-grid support without producing data.

    Parameters
    ----------
    csv_path : str or Path
        One raw or verified derived wide CSV, under the existing CSV limits.
    roles : dict, optional
        Confirmed entity, timestamp, measurement and label roles. No inference.
    timezone : str, optional
        Explicit timezone for naive clocks; mixed/ambiguous clocks are rejected.
    frequency, origin : str, optional
        Candidate fixed seconds/minutes/hours and aware anchor (or ``epoch``).
    mode : str
        Exact ``regularize`` (default) or explicitly reduced ``aggregate``.
    reducers, drop_columns : dict, list, optional
        Existing temporal aggregation/exclusion declarations.
    temporal_result, reshape_result, merge_result : str or dict, optional
        At most one explicit independently verified source reference.

    Returns
    -------
    dict
        Bounded ready/needs_clarification/blocked structure. Unknown missing
        instants stay None without a schedule. No grid values are generated.
    """
    value = dict(
        schema="timeseries-time-preview-v1",
        status="needs_clarification",
        unresolved=[],
        errors=[],
        entities=[],
        finite_support={},
        grid=None,
        roles=None,
        limitations=[
            "Interval variation alone does not establish dropped samples.",
            "Grid support is not imputation truth or a prediction-quality estimate.",
            "No resampling, interpolation, fitting or model execution occurred.",
        ],
    )
    try:
        facts = inspect_csv(csv_path)
        value["source"] = facts
        derived = _derived(csv_path, roles, temporal_result, reshape_result, merge_result)
        if derived:
            roles = derived["roles"]
            value["derived_reference"] = derived["reference"]
            value["inherited_split"] = derived["split"]
        if not roles or not roles.get("feature_cols"):
            raise Clarification("Confirm entity, timestamp and ordered feature roles")
        source, roles, observed, times = _source(csv_path, roles, timezone)
        if facts["sha256"] != observed["sha256"]:
            raise ValueError("Source changed while previewing time")
        value["roles"] = roles
        value["roles_sha256"] = content_sha256(roles)
        naive = all(pd.Timestamp(item).tzinfo is None for item in source[roles["timestamp_col"]])
        value["requires_normalization"] = naive
        groups = source.groupby(roles["sample_id_col"], sort=False) if roles["sample_id_col"] else [("0", source)]
        for entity, frame in groups:
            rows = frame.index.tolist()
            points = [times[row] for row in rows]
            differences = [right - left for left, right in zip(points, points[1:])]
            positive = [item / 1e9 for item in differences if item > 0]
            counts = Counter(points)
            duplicates = [row for row in rows if counts[times[row]] > 1]
            reversed_rows = [rows[index + 1] for index, difference in enumerate(differences) if difference < 0]
            regular = bool(differences) and len(set(differences)) == 1 and differences[0] > 0
            value["entities"].append(
                dict(
                    entity=str(entity),
                    rows=rows,
                    times_ns=points,
                    interval_seconds=(
                        dict(
                            min=min(positive),
                            median=float(np.median(positive)),
                            max=max(positive),
                        )
                        if positive
                        else None
                    ),
                    regular_step_seconds=differences[0] / 1e9 if regular else None,
                    missing_time_points=None,
                    duplicate_rows=duplicates,
                    reversed_rows=reversed_rows,
                )
            )
            value["requires_normalization"] |= bool(reversed_rows)
        value["finite_support"] = {
            feature: dict(
                finite=int(source[feature].notna().sum()),
                missing=int(source[feature].isna().sum()),
            )
            for feature in roles["feature_cols"]
        }
        if frequency is not None:
            if mode not in ("regularize", "aggregate"):
                raise ValueError("A grid requires regularize or aggregate semantics")
            grid_plan = create_temporal_plan(
                csv_path,
                mode=mode,
                roles=roles,
                timezone=timezone,
                frequency=frequency,
                origin=origin,
                reducers=reducers,
                drop_columns=drop_columns,
            )
            if grid_plan["status"] == "needs_clarification":
                raise Clarification("; ".join(grid_plan["unresolved"]))
            if grid_plan["status"] != "ready":
                raise ValueError("; ".join(grid_plan["errors"]))
            value["grid"] = _grid_support(source, grid_plan)
        elif reducers or drop_columns or mode != "regularize":
            raise Clarification("Declare a target frequency for the candidate grid")
        if file_sha256(csv_path) != facts["sha256"]:
            raise ValueError("Source changed during time preview")
        value["status"] = "ready"
    except Clarification as error:
        value["unresolved"] = [str(error)]
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        value.update(status="blocked", errors=[str(error)])
    value["sha256"] = content_sha256(value)
    return value


def _selection(csv_path, supplied, preview, arguments):
    from .ingestion import _steps

    task = arguments["task"]
    temporal_result = arguments.get("temporal_result")
    record = declaration(supplied)
    if record.get("schema", "timeseries-time-selection-v1") != "timeseries-time-selection-v1":
        raise ValueError("Unsupported time selection schema")
    digest = record.pop("sha256", None)
    if digest is not None and digest != content_sha256(record):
        raise ValueError("Time selection content changed; confirm material differences")
    if record.get("kind") not in ("observation_index", "regular_grid"):
        raise Clarification("Choose observation_index or regular_grid")
    if (
        record.get("impact_acknowledged") is not True
        or not isinstance(record.get("reason"), str)
        or not record["reason"].strip()
    ):
        raise Clarification("Record the confirmed representation and its known consequences")
    if len(record["reason"]) > 2000:
        raise ValueError("Time confirmation exceeds bounded text length")
    context = record.get("context", {})
    if not isinstance(context, dict) or context.get("task", task) != task:
        raise ValueError("Time choice belongs to a different task context")
    context = dict(context, task=task)
    index = record["kind"] == "observation_index"
    frequency = (record.get("grid") or {}).get("frequency")
    if not index and frequency is None:
        raise Clarification("Declare the regular-grid frequency and origin")
    sampling = [dict(regular_step_seconds=_duration(frequency) / 1e9 if frequency else None)]
    inherited = preview.get("inherited_split") or {}
    operation = dict(
        window_size=_steps(arguments["window_size"], sampling, index) if arguments["window_size"] is not None else None,
        horizon=_steps(arguments["horizon"], sampling, index) if arguments["horizon"] is not None else None,
        targets=arguments["targets"],
        split_policy=arguments["split_policy"] or inherited.get("policy"),
        clustering_model=arguments["clustering_model"] if task == "clustering" else None,
        n_clusters=arguments["n_clusters"] if task == "clustering" else None,
        anomaly_rate=arguments["anomaly_rate"] if task == "anomaly_detection" else None,
    )
    if context.get("operation", operation) != operation:
        raise ValueError("Time selection operation context changed; confirm material differences")
    context["operation"] = operation
    current = preview["source"]["sha256"]
    role_hash = preview["roles_sha256"]
    if record.get("roles_sha256", role_hash) != role_hash:
        raise ValueError("Time selection roles changed")
    origin = record.get("origin_source_sha256", current)
    previous = record.get("input_sha256", current)
    if origin != current:
        if temporal_result is None:
            raise ValueError("Changed input requires the verified temporal transition from the original decision")
        reference = preview["derived_reference"]
        temporal = declaration(Path(reference["path"]) / "plan.json")
        if temporal["source"]["sha256"] != origin or previous not in {origin, current}:
            raise ValueError("Time selection origin/input does not match the verified temporal transition")
        expected = record.get("grid")
        actual = temporal["request"]
        if record["kind"] == "regular_grid":
            if (
                not expected
                or actual["mode"] != expected["mode"]
                or _duration(actual["frequency"]) != _duration(expected["frequency"])
            ):
                raise ValueError("Verified transformation differs from the confirmed grid")
            expected_anchor = (
                pd.Timestamp("1970-01-01T00:00:00Z")
                if expected.get("origin", "epoch") == "epoch"
                else pd.Timestamp(expected["origin"])
            )
            if (
                expected_anchor.tzinfo is None
                or int(expected_anchor.tz_convert("UTC").as_unit("ns").value) != temporal["origin_ns"]
            ):
                raise ValueError("Verified transformation uses another grid origin")
            if (actual.get("reducers") or None) != (expected.get("reducers") or None) or (
                actual.get("drop_columns") or None
            ) != (expected.get("drop_columns") or None):
                raise ValueError("Verified aggregation/exclusion differs from the confirmed choice")
        elif actual["mode"] != "normalize":
            raise ValueError("Observation-index choice does not authorize a grid transformation")
        if actual.get("timezone") != record.get("timezone"):
            raise ValueError("Verified timezone differs from the confirmed interpretation")
    elif previous != current:
        raise ValueError("Time selection input does not match this source")
    record.update(
        schema="timeseries-time-selection-v1",
        origin_source_sha256=origin,
        input_sha256=current,
        roles_sha256=role_hash,
        context=context,
        timezone=record.get("timezone"),
        horizon_unit="record_steps",
        future_wall_clock="unknown",
        inherited_fitted_state="none",
    )
    record["sha256"] = content_sha256(record)
    return record


def create_time_selection_plan(csv_path, supplied, legacy_arguments):
    """Compose a v2 decision with the unchanged v1 split/window implementation."""
    from .ingestion import create_ingestion_plan

    request = copy.deepcopy(legacy_arguments)
    request["time_selection"] = declaration(supplied)
    value = dict(
        version=2,
        status="needs_clarification",
        request=request,
        unresolved=[],
        errors=[],
    )
    try:
        declared = request["time_selection"]
        grid = declared.get("grid") or {}
        if not isinstance(grid, dict):
            raise ValueError("Grid declaration must be an object")
        # After verified aggregation the output is already labelled on the grid;
        # do not propose aggregation for a second time just to inspect its clock.
        transitioned = declared.get("origin_source_sha256") not in (
            None,
            file_sha256(csv_path),
        )
        candidate = dict(grid)
        if transitioned and declared.get("kind") == "regular_grid":
            candidate = {key: item for key, item in candidate.items() if key not in ("reducers", "drop_columns")}
            candidate["mode"] = "regularize"
        preview = preview_time_basis(
            csv_path,
            request.get("roles"),
            timezone=declared.get("timezone"),
            temporal_result=request.get("temporal_result"),
            reshape_result=request.get("reshape_result"),
            merge_result=request.get("merge_result"),
            **candidate,
        )
        value["time_preview"] = preview
        if "source" in preview:
            value["source"] = preview["source"]
        if preview["status"] != "ready":
            value.update(
                status=preview["status"],
                unresolved=preview["unresolved"],
                errors=preview["errors"],
            )
            value["sha256"] = content_sha256(value)
            return value
        selected = _selection(csv_path, declared, preview, request)
        request.update(roles=preview["roles"], time_selection=selected)
        value["time_selection"] = selected
        value["roles"] = preview["roles"]
        index = selected["kind"] == "observation_index"
        if not index and request["allow_irregular"]:
            raise ValueError("allow_irregular=True conflicts with regular_grid")
        if any(group["duplicate_rows"] for group in preview["entities"]):
            raise ValueError("Duplicate instants require a separately confirmed temporal duplicate policy")
        if not index and not grid.get("frequency"):
            raise Clarification("Declare the regular-grid frequency and origin")
        needs_normalize = preview["requires_normalization"]
        needs_grid = not index and (
            preview["grid"]["output_rows"] != preview["source"]["rows"]
            or (grid.get("mode", "regularize") == "aggregate" and not transitioned)
        )
        if needs_normalize or needs_grid:
            value.update(
                status="needs_preparation",
                next_step=dict(
                    operation="temporal",
                    mode=("normalize" if needs_normalize and index else grid.get("mode", "regularize")),
                    timezone=selected["timezone"],
                    grid=selected.get("grid"),
                    reason="Use the existing plan/run/inspect, then pass the verified derived source; do not repeat the time choice.",
                ),
            )
        else:
            arguments = {key: item for key, item in request.items() if key != "time_selection"}
            arguments["allow_irregular"] = index
            value = create_ingestion_plan(csv_path, **arguments)
            value.update(version=2, time_selection=selected, time_preview=preview)
            value["request"]["time_selection"] = selected
    except Clarification as error:
        value.update(status="needs_clarification", unresolved=[str(error)])
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        value.update(status="blocked", errors=[str(error)])
    value.pop("sha256", None)
    value["sha256"] = content_sha256(value)
    return value
