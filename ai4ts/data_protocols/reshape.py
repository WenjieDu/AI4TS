"""Bounded explicit long-to-wide plans; no estimator or implicit aggregation."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ingestion import content_sha256, inspect_csv, MAX_CSV_BYTES, MAX_FEATURE_CELLS
from .registry import file_sha256
from .semantics import prepare_feature_semantics
from .temporal_preprocessing import Clarification, _timestamps, declaration, implementation_identity

PLAN_SCHEMA = "timeseries-reshape-plan-v1"
RESULT_SCHEMA = "timeseries-reshape-result-v1"
ROLES = ("sample_id_col", "timestamp_col", "variable_col", "value_col", "unit_col", "label_col")
LIMITS = dict(output_rows=100_000, feature_cells=1_000_000, contribution_references=10_000_000)
MAX_PLAN_BYTES = 32 * 1024 * 1024


def _read_long(path, roles, timezone):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Source symlinks are unsupported")
    facts = inspect_csv(path)
    if not isinstance(roles, dict):
        raise Clarification("Confirm entity, timestamp, variable and value columns")
    if set(roles) - set(ROLES):
        raise ValueError("Unsupported long-table roles")
    roles = {key: copy.deepcopy(roles.get(key)) for key in ROLES}
    if any(not roles[key] for key in ("timestamp_col", "variable_col", "value_col")):
        raise Clarification("Confirm timestamp, variable and value columns")
    columns = [value for value in roles.values() if value is not None]
    if any(not isinstance(c, str) for c in columns) or len(set(columns)) != len(columns):
        raise ValueError("Role columns must be distinct strings")
    if not set(columns).issubset(facts["columns"]):
        raise ValueError("Role names must be source columns")
    source = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if roles["sample_id_col"] and source[roles["sample_id_col"]].str.strip().eq("").any():
        raise ValueError("Entity identifiers cannot be empty")
    if source[roles["variable_col"]].str.strip().eq("").any():
        raise ValueError("Measurement variable names cannot be empty")
    times = _timestamps(source[roles["timestamp_col"]], timezone)
    if file_sha256(path) != facts["sha256"]:
        raise ValueError("Source changed while reading")
    return source, roles, facts, times


def _units(source, roles, rows, variables, supplied, source_sha):
    declared = prepare_feature_semantics(supplied, variables, source_sha) if supplied is not None else None
    units, evidence = {}, {}
    for variable in variables:
        selected = [row["measurements"][variable]["selected_row"] for row in rows if variable in row["measurements"]]
        labels = [source.at[index, roles["unit_col"]] for index in selected] if roles["unit_col"] else []
        known = set(labels) - {""}
        if len(known) > 1:
            raise ValueError(f"Conflicting source units for {variable}")
        value = next(iter(known), "unknown")
        explicit = declared["units"][variable] if declared else "unknown"
        if known and explicit not in ("unknown", value):
            raise ValueError(f"Explicit unit conflicts with source unit for {variable}")
        if known and "" in labels and explicit != value:
            raise Clarification(f"Confirm blank-unit records for {variable}; other records declare {value}")
        units[variable] = value if known else explicit
        evidence[variable] = dict(source_rows=selected, source_units=labels, explicit_unit=explicit)
    if not roles["unit_col"] and declared is None:
        return None, evidence
    declaration_record = dict(
        units=units,
        origin="user-confirmed" if declared else "source",
        source_sha256=source_sha,
        reference=("long measurement unit column " + roles["unit_col"]) if roles["unit_col"] else "explicit units",
    )
    return prepare_feature_semantics(declaration_record, variables, source_sha), evidence


def create_reshape_plan(
    csv_path,
    roles=None,
    variables=None,
    timezone=None,
    duplicate_policy=None,
    drop_columns=None,
    feature_units=None,
    confirmed_plan=None,
    execution_identity=None,
):
    """Plan long-to-wide reshaping without producing a dense table or fitting state.

    Parameters
    ----------
    csv_path : str or Path
        Bounded source long CSV, retained unchanged.
    roles : dict or None
        Confirmed timestamp/variable/value and optional entity/unit/label columns.
    variables : list of str or None
        Explicit selected measurement names in output order.
    timezone : str or None
        Interpretation required for naive clocks.
    duplicate_policy : str or None
        error (default), keep-first or keep-last; selects one complete record.
    drop_columns : list of str or None
        Explicitly excluded non-role metadata columns.
    feature_units : dict or None
        Source-bound or user-confirmed units; never a unit conversion.
    confirmed_plan : dict, str or None
        Same-source previous reshape declaration to reuse.
    execution_identity : dict or None
        Optional caller identity in addition to the implementation identity.

    Returns
    -------
    dict
        A ready, needs_clarification or blocked content-bound plan.
    """
    plan = dict(
        schema=PLAN_SCHEMA,
        status="needs_clarification",
        unresolved=[],
        errors=[],
        limits=dict(LIMITS),
        budget=dict(
            run_seconds=60,
            inspect_seconds=30,
            model_fit=0,
            model_predict=0,
            statistical_estimation=0,
            scaler_fit=0,
            provider=0,
            replay=0,
        ),
    )
    try:
        if Path(csv_path).is_symlink():
            raise ValueError("Source symlinks are unsupported")
        initial = inspect_csv(csv_path)
        plan.update(source=initial, implementation=dict(implementation_identity(), schema=PLAN_SCHEMA))
        prior = declaration(confirmed_plan) if confirmed_plan is not None else None
        if prior:
            expected = content_sha256({k: v for k, v in prior.items() if k != "sha256"})
            if prior.get("schema") != PLAN_SCHEMA or expected != prior.get("sha256"):
                raise ValueError("Invalid confirmed reshape plan")
            if prior.get("source", {}).get("sha256") != file_sha256(csv_path):
                raise ValueError("Confirmed reshape plan belongs to another source")
            previous = prior["request"]
            roles = previous["roles"] if roles is None else roles
            variables = previous["variables"] if variables is None else variables
            timezone = previous["timezone"] if timezone is None else timezone
            duplicate_policy = previous["duplicate_policy"] if duplicate_policy is None else duplicate_policy
            drop_columns = previous["drop_columns"] if drop_columns is None else drop_columns
            feature_units = previous["feature_units"] if feature_units is None else feature_units
        policy = duplicate_policy or "error"
        drops = [] if drop_columns is None else drop_columns
        plan["request"] = declaration(
            dict(
                roles=roles,
                variables=variables,
                timezone=timezone,
                duplicate_policy=policy,
                drop_columns=drops,
                feature_units=feature_units,
                confirmed_plan=prior,
                execution_identity=execution_identity,
            )
        )
        if policy not in ("error", "keep-first", "keep-last"):
            raise ValueError("Unsupported duplicate policy")
        source, roles, facts, times = _read_long(csv_path, roles, timezone)
        if initial != facts:
            raise ValueError("Source changed while planning")
        if variables is None:
            plan["candidates"] = list(dict.fromkeys(source[roles["variable_col"]]))
            raise Clarification("Confirm the selected measurement variables and their order")
        if (
            not isinstance(variables, list)
            or not variables
            or any(not isinstance(v, str) or not v.strip() for v in variables)
            or len(set(variables)) != len(variables)
        ):
            raise ValueError("Select distinct nonempty variable names")
        if not set(variables).issubset(set(source[roles["variable_col"]])):
            raise ValueError("Selected variables are absent from the source")
        if not isinstance(drops, list) or len(set(drops)) != len(drops):
            raise ValueError("drop_columns must be a list of unique source columns")
        structural = {roles[k] for k in ("variable_col", "value_col", "unit_col") if roles[k]}
        special = {value for value in roles.values() if value}
        if not set(drops).issubset(set(facts["columns"]) - special):
            raise ValueError("Only non-role metadata columns can be explicitly dropped")
        metadata = [
            c
            for c in facts["columns"]
            if c not in structural | set(drops) and c not in (roles["sample_id_col"], roles["timestamp_col"])
        ]
        prefix = [roles[k] for k in ("sample_id_col", "timestamp_col") if roles[k]]
        columns = prefix + variables + metadata
        if len(set(columns)) != len(columns):
            raise ValueError("Variable names conflict with output role or metadata headers")
        selected = source[roles["variable_col"]].isin(variables)
        numeric = pd.to_numeric(source.loc[selected, roles["value_col"]].replace("", np.nan), errors="raise").astype(
            float
        )
        raw = source.loc[selected, roles["value_col"]]
        if np.isinf(numeric).any() or (numeric.isna() & raw.ne("")).any():
            raise ValueError("Nonempty readings must be finite numbers")
        entities, keys, excluded = {}, {}, []
        chosen_names = set(variables)
        for index in source.index:
            variable = source.at[index, roles["variable_col"]]
            if variable not in chosen_names:
                excluded.append(int(index))
                continue
            entity = source.at[index, roles["sample_id_col"]] if roles["sample_id_col"] else "0"
            entities.setdefault(entity, len(entities))
            key = (entity, times[index])
            keys.setdefault(key, {}).setdefault(variable, []).append(int(index))
        count, width = len(keys), len(variables)
        if count > LIMITS["output_rows"] or count * width > LIMITS["feature_cells"]:
            raise ValueError("Wide output row/feature-cell limit exceeded")
        if count * len(columns) > MAX_FEATURE_CELLS or int(selected.sum()) > LIMITS["contribution_references"]:
            raise ValueError("Wide output/source-reference cell limit exceeded")
        rows, discarded = [], []
        missing, present, finite = 0, 0, 0
        for key in sorted(keys, key=lambda item: (entities[item[0]], item[1])):
            entity = key[0]
            measurements, retained = {}, []
            for variable, indexes in keys[key].items():
                if len(indexes) > 1 and policy == "error":
                    raise ValueError(f"Duplicate measurement key: {entity}/{variable}")
                chosen = indexes[-1] if policy == "keep-last" else indexes[0]
                measurements[variable] = dict(source_rows=indexes, selected_row=chosen)
                retained.append(chosen)
                present += 1
                missing += int(pd.isna(numeric.at[chosen]))
                finite += int(pd.notna(numeric.at[chosen]))
                discarded.extend(
                    dict(row=i, retained=chosen, reason="unselected-duplicate") for i in indexes if i != chosen
                )
            values = {}
            for column in metadata:
                labels = {source.at[i, column] for i in retained}
                if len(labels) != 1:
                    raise ValueError(f"Conflicting metadata or label {column}; explicitly drop non-role metadata")
                values[column] = next(iter(labels))
            rows.append(dict(entity=entity, time_ns=key[1], measurements=measurements, metadata=values))
        semantics, evidence = _units(source, roles, rows, variables, feature_units, facts["sha256"])
        # CSV quoting may double every source character; include headers and all metadata.
        byte_bound = 2 * sum(len(c.encode()) + 3 for c in columns)
        for row in rows:
            byte_bound += 3 * len(columns) + 1 + 40 + width * 26
            byte_bound += 2 * (len(row["entity"].encode()) + sum(len(v.encode()) for v in row["metadata"].values())) + 8
        if byte_bound > MAX_CSV_BYTES:
            raise ValueError("Wide CSV byte limit exceeded")
        request = dict(
            roles=roles,
            variables=list(variables),
            timezone=timezone,
            duplicate_policy=policy,
            drop_columns=list(drops),
            feature_units=copy.deepcopy(feature_units),
            confirmed_plan=prior,
            execution_identity=copy.deepcopy(execution_identity),
        )
        plan.update(
            status="ready",
            request=request,
            source=facts,
            roles=roles,
            wide_roles=dict(
                sample_id_col=roles["sample_id_col"],
                timestamp_col=roles["timestamp_col"],
                feature_cols=list(variables),
                label_col=roles["label_col"],
            ),
            columns=columns,
            metadata=metadata,
            rows=rows,
            discarded=discarded,
            excluded=excluded,
            feature_semantics=semantics,
            unit_evidence=evidence,
            split=None,
            preview=dict(
                input_rows=len(source),
                output_rows=count,
                feature_cells=count * width,
                absent_measurements=count * width - present,
                recorded_missing=missing,
                finite_measurements=finite,
                unselected_duplicates=len(discarded),
                excluded_variable_records=len(excluded),
                dropped_columns=list(drops),
                estimated_csv_bytes_upper_bound=byte_bound,
                split_status="unsplit",
            ),
        )
        if file_sha256(csv_path) != facts["sha256"]:
            raise ValueError("Source changed while planning")
    except Clarification as error:
        plan.update(status="needs_clarification", unresolved=[str(error)])
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        plan.update(status="blocked", errors=[str(error)])
    plan["sha256"] = content_sha256(plan)
    # The exact CLI serialization must fit the declaration reader as well as the CSV.
    if len((json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()) > MAX_PLAN_BYTES:
        plan = dict(
            schema=PLAN_SCHEMA,
            status="blocked",
            unresolved=[],
            errors=["Reshape plan declaration byte limit exceeded"],
            limits=dict(LIMITS, declaration_bytes=MAX_PLAN_BYTES),
            budget=plan["budget"],
            source={"sha256": plan.get("source", {}).get("sha256")},
        )
        plan["sha256"] = content_sha256(plan)
    return plan


def validate_reshape_plan(plan, csv_path):
    """Rebuild a ready plan from the unchanged source and confirmed declarations.

    Parameters
    ----------
    plan : dict
        Ready reshape plan.
    csv_path : str or Path
        Actual long source.

    Returns
    -------
    dict
        The validated plan, unchanged.
    """
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "ready":
        raise ValueError("A ready supported reshape plan is required")
    if create_reshape_plan(csv_path, **plan["request"]) != plan:
        raise ValueError("Reshape plan/source/implementation changed; create a new plan")
    return plan


def run_reshape(csv_path, plan, output, timeout=60):
    """Execute a ready long-to-wide plan into a new directory.

    Parameters
    ----------
    csv_path : str or Path
        Unchanged long source.
    plan : dict
        Ready reshape plan.
    output : str or Path
        New directory; existing artifacts are never overwritten.
    timeout : float
        Owned worker limit in (0, 60] seconds.

    Returns
    -------
    dict
        Success or failure with actual output paths.
    """
    from .temporal_artifacts import supervise

    return supervise("run", csv_path, plan, output, timeout, kind="reshape")


def inspect_reshape_result(result, timeout=30):
    """Independently read and verify an existing long-to-wide result.

    Parameters
    ----------
    result : str or Path
        Existing result directory.
    timeout : float
        Read-only worker limit in (0, 30] seconds.

    Returns
    -------
    dict
        Verified saved facts or explicit failure; no artifacts are written.
    """
    from .temporal_artifacts import supervise

    return supervise("inspect", result, None, None, timeout, kind="reshape")


def reshape_context(result, csv_path, roles=None):
    """Verify an explicit long-source reference before a downstream plan."""
    from . import reshape_artifacts, temporal_artifacts
    from .preprocessing import state_sha256

    supplied = declaration(result) if isinstance(result, dict) else None
    root = Path(supplied["path"] if supplied else result).absolute()
    path = root / "result.json"
    if root.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("Missing or unsafe reshape result")
    digest = file_sha256(path)
    fields = {"path", "result_sha256", "source_sha256", "derived_sha256"}
    if supplied and (set(supplied) != fields or supplied["result_sha256"] != digest):
        raise ValueError("Reshape result reference changed")
    verified = (
        reshape_artifacts._verify(root)
        if reshape_artifacts._IN_WORKER or temporal_artifacts._IN_WORKER
        else inspect_reshape_result(root)
    )
    if verified["status"] != "success":
        raise ValueError("Reshape result verification failed: " + verified.get("error", "unknown"))
    if file_sha256(csv_path) != verified["derived_sha256"] or file_sha256(path) != digest:
        raise ValueError("Reshape result/input identity differs")
    if supplied and any(supplied[k] != verified[k] for k in ("source_sha256", "derived_sha256")):
        raise ValueError("Reshape parent source identity differs")
    bound = verified["roles"]
    if roles is not None:
        if set(roles) - set(bound):
            raise ValueError("Unsupported derived roles")
        requested = copy.deepcopy(bound)
        requested.update(copy.deepcopy(roles))
        features = requested["feature_cols"]
        if (
            not isinstance(features, list)
            or not features
            or len(set(features)) != len(features)
            or not set(features).issubset(bound["feature_cols"])
            or any(requested[k] != bound[k] for k in bound if k != "feature_cols")
        ):
            raise ValueError("Derived roles conflict with reshape semantics")
        bound = requested
    semantics = verified["feature_semantics"]
    if semantics and semantics["feature_columns"] != bound["feature_cols"]:
        parent = state_sha256(semantics)
        semantics = copy.deepcopy(semantics)
        semantics.update(
            feature_columns=bound["feature_cols"],
            units={f: semantics["units"][f] for f in bound["feature_cols"]},
            inherited_from_sha256=parent,
        )
    return dict(
        reference=dict(
            path=str(root),
            result_sha256=digest,
            source_sha256=verified["source_sha256"],
            derived_sha256=verified["derived_sha256"],
        ),
        roles=bound,
        feature_semantics=semantics,
        split=None,
    )
