"""Explicit, source-bound physical-time preparation; no imputation or estimators."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .ingestion import content_sha256, inspect_csv, validate_ingestion_plan
from .registry import file_sha256
from .semantics import prepare_feature_semantics

PLAN_SCHEMA = "timeseries-temporal-plan-v1"
RESULT_SCHEMA = "timeseries-temporal-result-v1"
LIMITS = {"output_rows": 100_000, "feature_cells": 1_000_000, "contribution_references": 10_000_000}
ROLES = ("feature_cols", "sample_id_col", "timestamp_col", "label_col")


class Clarification(ValueError):
    """A scientific declaration is missing, rather than an invalid source."""


def implementation_identity():
    """Return the exact temporal implementation identity, without importing backends."""
    digest = hashlib.sha256()
    root = Path(__file__).parents[1]
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {"schema": PLAN_SCHEMA, "code_sha256": digest.hexdigest(), "pandas": pd.__version__, "numpy": np.__version__}


def declaration(value):
    """Read a bounded JSON declaration or copy an already supplied mapping."""
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Invalid or oversized declaration file")
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("A JSON object declaration is required")
    return json.loads(json.dumps(value, allow_nan=False))


def _confirmed(value, source):
    value = declaration(value)
    body = {k: v for k, v in value.items() if k not in ("sha256", "execution_identity")}
    # Analysis plans include execution identity in their content hash.
    if value.get("schema") == "timeseries-analysis-plan-v1":
        body = {k: v for k, v in value.items() if k != "sha256"}
    if value.get("sha256") != content_sha256(body) or value.get("source", {}).get("sha256") != source["sha256"]:
        raise ValueError("Confirmed plan does not match its content/source")
    if not value.get("roles"):
        raise ValueError("Prior plan has no confirmed roles")
    return value


def _duration(value):
    match = re.fullmatch(r"([1-9][0-9]*)\s*(s|sec|seconds?|min|minutes?|h|hours?)", str(value))
    if not match:
        raise ValueError("frequency must be a positive integer duration in seconds, minutes or hours")
    factor = 3600 if match[2].startswith("h") else 60 if match[2].startswith("min") else 1
    nanos = int(match[1]) * factor * 1_000_000_000
    if nanos > np.iinfo(np.int64).max:
        raise ValueError("frequency overflows supported physical time")
    return nanos


def _timestamps(values, timezone):
    if pd.to_numeric(values.replace("", np.nan), errors="coerce").notna().any():
        raise ValueError("Numeric epoch timestamps are not supported; declare an upstream conversion")
    stamps = [pd.Timestamp(value) for value in values]
    if any(pd.isna(stamp) for stamp in stamps):
        raise ValueError("Invalid or empty physical timestamp")
    naive = [stamp.tzinfo is None for stamp in stamps]
    if any(naive) and not all(naive):
        raise ValueError("Mixed timezone-aware and naive timestamps are unsupported")
    if all(naive):
        if not timezone:
            raise Clarification("Declare timezone for naive timestamps")
        try:
            stamps = [stamp.tz_localize(timezone, ambiguous="raise", nonexistent="raise") for stamp in stamps]
        except Exception as error:
            raise ValueError(f"Invalid/ambiguous/nonexistent timezone interpretation: {error}") from error
    return [int(stamp.tz_convert("UTC").as_unit("ns").value) for stamp in stamps]


def iso(nanos):
    """Render a physical instant without discarding subsecond precision."""
    return pd.Timestamp(nanos, unit="ns", tz="UTC").isoformat().replace("+00:00", "Z")


def _source(csv_path, roles, timezone):
    from .analysis import read_analysis_source

    path = Path(csv_path)
    if path.is_symlink():
        raise ValueError("Source symlinks are unsupported")
    source, roles, facts = read_analysis_source(path, roles)
    if not roles["timestamp_col"]:
        raise Clarification("Confirm a physical timestamp column")
    times = _timestamps(source[roles["timestamp_col"]], timezone)
    return source, roles, facts, times


def _partition(source, roles, times, split_plan, csv_path):
    if split_plan is None:
        return None, {}, {}
    donor = declaration(split_plan)
    validate_ingestion_plan(donor, csv_path)
    if donor["roles"] != roles or donor["split"]["policy"] not in ("entity", "chronological", "entity_chronological"):
        raise ValueError("Prior split roles/policy conflict with physical-time preprocessing")
    assignment = {row: role for role, rows in donor["split"]["rows"].items() for row in rows}
    if set(assignment) != set(source.index):
        raise ValueError("Prior split must cover every original row")
    boundaries = {}
    groups = source.groupby(roles["sample_id_col"], sort=False) if roles["sample_id_col"] else [("0", source)]
    for entity, group in groups:
        present = {assignment[row] for row in group.index}
        if donor["split"]["policy"] == "entity":
            if len(present) != 1:
                raise ValueError("Entity isolation was lost")
            boundaries[str(entity)] = {"entity_role": next(iter(present))}
        else:
            if present != {"train", "val", "test"}:
                raise ValueError("Temporal split requires all three roles per entity")
            boundaries[str(entity)] = {
                role: min(times[row] for row in group.index if assignment[row] == role) for role in ("val", "test")
            }
            for row in group.index:
                if assignment[row] != _role(times[row], boundaries[str(entity)]):
                    raise ValueError("Prior physical-time split is not chronological")
    return donor, assignment, boundaries


def _role(time, boundary):
    if "entity_role" in boundary:
        return boundary["entity_role"]
    return "train" if time < boundary["val"] else "val" if time < boundary["test"] else "test"


def create_temporal_plan(
    csv_path,
    mode=None,
    roles=None,
    frequency=None,
    timezone=None,
    origin="epoch",
    duplicate_policy="error",
    reducers=None,
    drop_columns=None,
    feature_units=None,
    confirmed_plan=None,
    split_plan=None,
    execution_identity=None,
    reshape_result=None,
    merge_result=None,
):
    """Plan explicit time operations without producing transformed numerical data.

    Parameters
    ----------
    csv_path : str or Path
        Bounded source wide CSV; never modified.
    mode : str or None
        normalize, regularize, or aggregate.
    roles : dict or None
        Confirmed entity, timestamp, ordered features and optional label.
    frequency, timezone, origin : str or None
        Fixed duration, naive-clock interpretation and shared grid anchor.
    duplicate_policy : str
        error, keep-first, or keep-last, applied to complete records.
    reducers : dict or None
        Explicit feature-to-mean/sum/last mapping for aggregation.
    drop_columns : list or None
        Explicitly excluded non-role metadata columns.
    feature_units : dict or None
        Source-bound unit declaration; absent units remain unknown.
    confirmed_plan, split_plan : dict, str or None
        Source-matching semantic declaration and optional ready ingestion split.
    execution_identity : dict or None
        Optional caller identity, in addition to the temporal code identity.
    reshape_result : str, dict or None
        Verified long-to-wide parent; no fitted state or old split is inherited.
    merge_result : str, dict or None
        Unique verified unsplit merge parent, preserved by the temporal result.

    Returns
    -------
    dict
        Content-bound ready, needs_clarification or blocked plan.
    """
    request = dict(
        mode=mode,
        roles=copy.deepcopy(roles),
        frequency=frequency,
        timezone=timezone,
        origin=origin,
        duplicate_policy=duplicate_policy,
        reducers=copy.deepcopy(reducers),
        drop_columns=copy.deepcopy(drop_columns),
        feature_units=copy.deepcopy(feature_units),
        confirmed_plan=declaration(confirmed_plan) if confirmed_plan is not None else None,
        split_plan=declaration(split_plan) if split_plan is not None else None,
        execution_identity=copy.deepcopy(execution_identity),
    )
    plan = dict(
        schema=PLAN_SCHEMA,
        status="needs_clarification",
        request=request,
        unresolved=[],
        errors=[],
        limits=dict(LIMITS),
        budget={"run_seconds": 60, "inspect_seconds": 30},
        implementation=implementation_identity(),
    )
    try:
        facts = inspect_csv(csv_path)
        plan["source"] = {k: facts[k] for k in ("sha256", "bytes", "rows", "columns")}
        prior = _confirmed(request["confirmed_plan"], facts) if confirmed_plan is not None else None
        if roles is None and prior:
            roles = prior["roles"]
        derived = None
        if reshape_result is not None and merge_result is not None:
            raise ValueError("Choose one top-level derived reference")
        if reshape_result is not None or merge_result is not None:
            from .time_basis import _derived

            derived = _derived(csv_path, roles, reshape_result=reshape_result, merge_result=merge_result)
            roles = derived["roles"]
            key = "merge_result" if merge_result is not None else "reshape_result"
            request[key] = derived["reference"]
            plan[key] = derived["reference"]
        if not roles or roles.get("feature_cols") is None:
            raise Clarification("Confirm entity, physical timestamp and ordered measurement roles")
        if timezone is None and prior:
            timezone = prior.get("request", {}).get("timezone")
            request["timezone"] = timezone
        if mode is None:
            raise Clarification("Choose normalize, regularize or aggregate")
        if mode not in ("normalize", "regularize", "aggregate"):
            raise ValueError("Unsupported temporal mode")
        if duplicate_policy not in ("error", "keep-first", "keep-last"):
            raise ValueError("Unsupported duplicate policy")
        source, roles, observed, times = _source(csv_path, roles, timezone)
        if observed["sha256"] != facts["sha256"]:
            raise ValueError("Source changed while planning")
        plan["roles"] = roles
        if mode != "normalize" and frequency is None:
            raise Clarification("Declare the target fixed-duration frequency")
        step = _duration(frequency) if mode != "normalize" else None
        if mode == "normalize" and (frequency is not None or reducers):
            raise ValueError("normalize does not accept frequency or reducers")
        anchor = pd.Timestamp("1970-01-01T00:00:00Z") if origin == "epoch" else pd.Timestamp(origin)
        if anchor.tzinfo is None or pd.isna(anchor):
            raise ValueError("origin must be epoch or an explicit timezone-aware instant")
        anchor = int(anchor.tz_convert("UTC").as_unit("ns").value)
        if mode == "aggregate" and reducers is None:
            raise Clarification("Declare mean, sum or last for each selected feature")
        if mode == "aggregate" and (
            not isinstance(reducers, dict)
            or set(reducers) != set(roles["feature_cols"])
            or any(v not in ("mean", "sum", "last") for v in reducers.values())
        ):
            raise ValueError("reducers must name every selected feature with mean, sum or last")
        if mode != "aggregate" and reducers:
            raise ValueError("reducers require aggregate mode")
        dropped = drop_columns or []
        special = [roles[k] for k in ROLES if k != "feature_cols" and roles[k] is not None]
        if (
            not isinstance(dropped, list)
            or len(set(dropped)) != len(dropped)
            or not set(dropped).issubset(source.columns)
            or set(dropped) & set(special + roles["feature_cols"])
        ):
            raise ValueError("drop_columns may only explicitly exclude non-role metadata")
        output_columns = [c for c in source.columns if c not in dropped]
        metadata = [c for c in source.columns if c not in special + roles["feature_cols"] + dropped]
        if roles["label_col"]:
            metadata.append(roles["label_col"])
        donor, assignment, boundaries = _partition(source, roles, times, request["split_plan"], csv_path)
        groups = []
        total = 0
        estimated_bytes = sum(2 * len(c.encode("utf-8")) + 3 for c in output_columns)
        refs = 0
        rows_by_role = {name: [] for name in ("train", "val", "test")}
        grouped = source.groupby(roles["sample_id_col"], sort=False) if roles["sample_id_col"] else [("0", source)]
        for entity, frame in grouped:
            ordered = sorted(frame.index.tolist(), key=lambda i: (times[i], i))
            by_time = {}
            for row in ordered:
                by_time.setdefault(times[row], []).append(row)
            if duplicate_policy == "error" and any(len(rows) > 1 for rows in by_time.values()):
                raise ValueError(f"Duplicate physical timestamps in entity {entity!r}; declare a duplicate policy")
            retained = [rows[-1] if duplicate_policy == "keep-last" else rows[0] for rows in by_time.values()]
            discarded = [
                dict(row=row, retained=keep, reason="unselected-duplicate")
                for rows, keep in zip(by_time.values(), retained)
                for row in rows
                if row != keep
            ]
            constant = {
                c: source.loc[retained[0], c] for c in metadata if source.loc[retained, c].nunique(dropna=False) == 1
            }
            if mode == "aggregate" and set(constant) != set(metadata):
                raise ValueError(
                    f"Varying metadata require explicit exclusion: {sorted(set(metadata) - set(constant))}"
                )
            low, high = times[retained[0]], times[retained[-1]]
            if mode == "regularize" and any((times[i] - anchor) % step for i in retained):
                raise ValueError("Observation is off the declared grid; no rounding or snapping performed")
            first = low if mode == "regularize" else anchor + ((low - anchor) // step) * step if step else low
            last = high if mode == "regularize" else anchor + ((high - anchor) // step) * step if step else high
            count = len(retained) if mode == "normalize" else (last - first) // step + 1
            if mode == "aggregate" and not pd.Timestamp.min.value <= last + step <= pd.Timestamp.max.value:
                raise ValueError("Bucket end overflows supported physical time")
            total += count
            refs += len(retained) * len(roles["feature_cols"])
            metadata_size = sum(
                max(len(str(source.at[i, c]).encode("utf-8")) for i in retained) * 2 + 3 for c in metadata
            )
            estimated_bytes += count * (
                64
                + 27 * len(roles["feature_cols"])
                + metadata_size
                + 2 * len(str(entity).encode("utf-8"))
                + len(source.columns)
                + 4
            )
            if (
                total > LIMITS["output_rows"]
                or total * len(roles["feature_cols"]) > LIMITS["feature_cells"]
                or refs > LIMITS["contribution_references"]
                or total * len(output_columns) > 10_000_000
                or estimated_bytes > 32 * 1024 * 1024
            ):
                raise ValueError("Temporal output rows/feature cells/contribution references exceed limits")
            boundary = boundaries.get(str(entity))
            if (
                boundary
                and mode == "aggregate"
                and "entity_role" not in boundary
                and any((cut - anchor) % step for cut in boundary.values())
            ):
                raise ValueError("Existing split boundary falls inside an aggregation bucket")
            group = dict(
                entity=str(entity),
                rows=frame.index.tolist(),
                retained=retained,
                discarded=discarded,
                first=first,
                last=last,
                count=count,
                constants=constant,
                original_first=low,
                original_last=high,
                boundary=boundary,
            )
            groups.append(group)
            if donor:
                for offset in range(count):
                    point = times[retained[offset]] if mode == "normalize" else first + offset * step
                    role = _role(point, boundary)
                    rows_by_role[role].append(total - count + offset)
        if donor and not all(rows_by_role.values()):
            raise ValueError("Temporal operation emptied an inherited split")
        units = feature_units
        if units is None and prior and prior.get("feature_semantics"):
            unit_state = prior["feature_semantics"]
        else:
            unit_state = None
        if derived and derived["feature_semantics"]:
            inherited = derived["feature_semantics"]
            supplements = {}
            if unit_state:
                for feature in roles["feature_cols"]:
                    old, new = unit_state["units"][feature], inherited["units"][feature]
                    if old != "unknown" and new not in ("unknown", old):
                        raise ValueError("Confirmed units conflict with reshape source units")
                    if new == "unknown" and old != "unknown":
                        supplements[feature] = old
            unit_state = inherited
            if supplements:
                unit_state = prepare_feature_semantics(
                    supplements, roles["feature_cols"], facts["sha256"], unit_state=inherited
                )
        semantic = prepare_feature_semantics(units, roles["feature_cols"], facts["sha256"], unit_state=unit_state)
        plan.update(
            groups=groups,
            times_ns=times,
            metadata=metadata,
            columns=output_columns,
            frequency_ns=step,
            origin_ns=anchor,
            feature_semantics=semantic,
            split=(
                {"policy": donor["split"]["policy"], "seed": donor["split"]["seed"], "rows": rows_by_role}
                if donor
                else None
            ),
            preview=dict(
                input_rows=len(source),
                output_rows=total,
                feature_cells=total * len(roles["feature_cols"]),
                estimated_csv_bytes_upper_bound=estimated_bytes,
                contribution_references=refs,
                original_missing=int(source[roles["feature_cols"]].isna().sum().sum()),
                retained_rows=sum(len(g["retained"]) for g in groups),
                unselected_duplicates=sum(len(g["discarded"]) for g in groups),
                grid_added_rows=total - sum(len(g["retained"]) for g in groups) if mode == "regularize" else 0,
                dropped_columns=dropped,
                split_status="inherited" if donor else "unsplit",
                bucket_rule="[start,end), labelled end" if mode == "aggregate" else None,
            ),
            status="ready",
        )
        if file_sha256(csv_path) != facts["sha256"]:
            raise ValueError("Source changed while planning")
    except Clarification as error:
        plan["unresolved"] = [str(error)]
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        plan.update(status="blocked", errors=[str(error)])
    plan["sha256"] = content_sha256(plan)
    return plan


def validate_temporal_plan(plan, csv_path):
    """Rebuild ready plan semantics from the actual unchanged source.

    Parameters
    ----------
    plan : dict
        Ready temporal plan.
    csv_path : str or Path
        Actual source to validate against all recorded declarations.

    Returns
    -------
    dict
        The validated plan.
    """
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "ready":
        raise ValueError("A ready supported temporal plan is required")
    if create_temporal_plan(csv_path, **plan["request"]) != plan:
        raise ValueError("Temporal plan/source/implementation changed; create a new plan")
    return plan


def run_temporal_preprocessing(csv_path, plan, output, timeout=60):
    """Execute a ready temporal plan into a new directory within its time budget.

    Parameters
    ----------
    csv_path : str or Path
        Source matching the ready plan.
    plan : dict
        Confirmed temporal plan.
    output : str or Path
        New directory; existing results are never overwritten.
    timeout : float
        Worker budget in (0, 60] seconds.

    Returns
    -------
    dict
        Execution status and real output paths.
    """
    from .temporal_artifacts import supervise

    return supervise("run", csv_path, plan, output, timeout)


def inspect_temporal_result(result, timeout=30):
    """Independently verify existing temporal data without a new preparation run.

    Parameters
    ----------
    result : str or Path
        Saved temporal result directory.
    timeout : float
        Read-only worker budget in (0, 30] seconds.

    Returns
    -------
    dict
        Verified saved facts and existing artifact paths.
    """
    from .temporal_artifacts import supervise

    return supervise("inspect", result, None, None, timeout)


def temporal_context(result, csv_path, roles=None):
    """Verify an explicit derived-source reference for downstream plan creation."""
    from . import temporal_artifacts

    supplied = declaration(result) if isinstance(result, dict) else None
    root = Path(supplied["path"] if supplied else result).absolute()
    if root.is_symlink():
        raise ValueError("Temporal result symlinks are unsupported")
    result_path = root / "result.json"
    if result_path.is_symlink() or not result_path.is_file():
        raise ValueError("Missing or unsafe temporal result")
    digest = file_sha256(result_path)
    if supplied and (
        set(supplied) != {"path", "result_sha256", "derived_sha256"} or supplied["result_sha256"] != digest
    ):
        raise ValueError("Temporal result reference changed")
    # A supervised verifier/producer may validate a prior derived source without
    # spawning grandchildren that could outlive its own deadline.
    verified = temporal_artifacts._verify(root) if temporal_artifacts._IN_WORKER else inspect_temporal_result(root)
    if verified["status"] != "success":
        raise ValueError("Temporal result verification failed: " + verified.get("error", "unknown"))
    if file_sha256(csv_path) != verified["derived_sha256"] or (
        supplied and supplied["derived_sha256"] != verified["derived_sha256"]
    ):
        raise ValueError("Temporal result is bound to a different derived CSV")
    if file_sha256(result_path) != digest:
        raise ValueError("Temporal result changed while reading")
    bound_roles = verified["roles"]
    if roles is not None:
        if set(roles) - set(ROLES):
            raise ValueError("Unsupported derived roles")
        requested = {key: copy.deepcopy(roles.get(key)) for key in ROLES}
        features = requested["feature_cols"]
        if (
            not isinstance(features, list)
            or not features
            or len(set(features)) != len(features)
            or not set(features).issubset(bound_roles["feature_cols"])
            or any(requested[k] != bound_roles[k] for k in ROLES if k != "feature_cols")
        ):
            raise ValueError("Derived roles conflict with verified temporal semantics")
        bound_roles = requested
    semantics = verified["feature_semantics"]
    if semantics and bound_roles["feature_cols"] != semantics["feature_columns"]:
        from .preprocessing import state_sha256

        parent = state_sha256(semantics)
        semantics = copy.deepcopy(semantics)
        semantics.update(
            feature_columns=bound_roles["feature_cols"],
            units={f: semantics["units"][f] for f in bound_roles["feature_cols"]},
            inherited_from_sha256=parent,
        )
    return dict(
        reference=dict(path=str(root), result_sha256=digest, derived_sha256=verified["derived_sha256"]),
        roles=bound_roles,
        split=verified["split"],
        feature_semantics=semantics,
    )
