"""Source-bound two-dimensional analysis plans; no statistical backend imports."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .ingestion import content_sha256, inspect_csv, validate_ingestion_plan
from .pipeline import split_source_rows
from .registry import file_sha256
from .semantics import prepare_feature_semantics

SCHEMA = "timeseries-analysis-plan-v1"
METHODS = ("acf", "pacf", "adf", "kpss", "stl", "lombscargle")


def read_analysis_source(csv_path, roles):
    """Read bounded CSV data without sorting or repairing temporal defects.

    Parameters
    ----------
    csv_path : str or Path
        Unmodified UTF-8 wide CSV.
    roles : dict
        Explicit feature and optional identifier/time/label columns.

    Returns
    -------
    tuple
        Source dataframe, normalized roles, and bounded source inspection.
    """
    facts = inspect_csv(csv_path)
    if not isinstance(roles, dict) or set(roles) - {"feature_cols", "sample_id_col", "timestamp_col", "label_col"}:
        raise ValueError("Explicit supported column roles are required")
    roles = {
        key: copy.deepcopy(roles.get(key)) for key in ("feature_cols", "sample_id_col", "timestamp_col", "label_col")
    }
    features = roles["feature_cols"]
    special = [roles[key] for key in ("sample_id_col", "timestamp_col", "label_col") if roles[key] is not None]
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(c, str) for c in features)
        or len(set(features + special)) != len(features + special)
        or not set(features + special).issubset(facts["columns"])
    ):
        raise ValueError("Feature and special roles must be unique source columns")
    source = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for column in features:
        values = source[column].str.strip().replace({"": np.nan, "NaN": np.nan, "nan": np.nan})
        source[column] = pd.to_numeric(values, errors="raise").astype(float)
    if np.isinf(source[features].to_numpy()).any():
        raise ValueError("Features contain infinite values")
    if roles["sample_id_col"] and source[roles["sample_id_col"]].str.strip().eq("").any():
        raise ValueError("Entity identifiers cannot be empty")
    if file_sha256(csv_path) != facts["sha256"]:
        raise ValueError("Source changed while reading")
    return source, roles, facts


def analysis_groups(source, roles):
    """Return separate entities in original row order."""
    column = roles["sample_id_col"]
    return list(source.groupby(column, sort=True)) if column else [("0", source)]


def time_profile(group, roles, expected_frequency=None):
    """Describe real timestamps without filling, sorting, or deduplicating."""
    column = roles["timestamp_col"]
    result = {
        "time_basis": "observation_index",
        "rows": group.index.tolist(),
        "valid": False,
        "regular_step_seconds": None,
        "missing_time_points": None,
        "seconds": None,
        "duplicate_rows": [],
        "reversed_rows": [],
        "unparseable_rows": [],
    }
    if not column:
        return result
    result["time_basis"] = "timestamp"
    raw = group[column]
    if pd.to_numeric(raw.replace("", np.nan), errors="coerce").notna().any():
        result["reason"] = "Numeric timestamps require explicit upstream epoch/unit conversion"
        return result
    times = pd.to_datetime(raw, format="mixed", errors="coerce", utc=True)
    result["unparseable_rows"] = group.index[times.isna()].tolist()
    result["duplicate_rows"] = group.index[times.duplicated(keep=False) & times.notna()].tolist()
    differences = times.diff().dt.total_seconds()
    result["reversed_rows"] = group.index[differences.lt(0)].tolist()
    if result["unparseable_rows"] or result["duplicate_rows"] or result["reversed_rows"]:
        result["reason"] = "Invalid, duplicate or reversed time; no repair performed"
        return result
    seconds = (times - times.iloc[0]).dt.total_seconds().to_numpy()
    result.update(valid=True, seconds=seconds.tolist())
    delta = np.diff(seconds)
    result["interval_seconds"] = (
        {"min": float(delta.min()), "median": float(np.median(delta)), "max": float(delta.max())}
        if len(delta)
        else None
    )
    if len(delta) and np.all(delta == delta[0]) and delta[0] > 0:
        result["regular_step_seconds"] = float(delta[0])
    if expected_frequency is not None:
        cadence = float(pd.Timedelta(expected_frequency).total_seconds())
        if not np.isfinite(cadence) or cadence <= 0:
            raise ValueError("Expected frequency must be a positive fixed duration")
        positions = seconds / cadence
        aligned = bool(np.allclose(positions, np.rint(positions), rtol=0, atol=1e-6))
        result["declared_grid"] = {
            "step_seconds": cadence,
            "origin": "first observed timestamp",
            "within": "observed first-to-last span",
            "aligned": aligned,
        }
        if aligned:
            result["missing_time_points"] = int(round(positions[-1])) + 1 - len(group)
    return result


def _applicability(method, values, temporal, period, period_range):
    finite = np.isfinite(values)
    minimum = (2 * (period or 2)) if method == "stl" else 16 if method in ("acf", "pacf") else 32
    if not temporal["valid"]:
        return "unsupported", "Valid strictly increasing timestamps required"
    if finite.sum() < minimum or np.ptp(values[finite]) == 0:
        return "skipped", "Insufficient finite observations or constant series"
    if method == "lombscargle":
        if period_range is None:
            return "unsupported", "Explicit physical period search range required"
        if period_range[1] > temporal["seconds"][-1]:
            return "unsupported", "Maximum period exceeds selected time span"
    else:
        if temporal["regular_step_seconds"] is None:
            return "unsupported", "Regular temporal grid required"
        if method != "acf" and not finite.all():
            return "unsupported", "Complete observations required; no implicit filling or dropping"
        if method == "stl" and (period is None or len(values) < 2 * period):
            return "skipped", "Explicit period and at least two cycles required"
    return "ready", None


def create_analysis_plan(
    csv_path,
    kind="diagnostics",
    roles=None,
    methods=None,
    period=None,
    period_range=None,
    scope="descriptive_all",
    scope_plan=None,
    rows=None,
    expected_frequency=None,
    time_basis=None,
    targets=None,
    split_policy=None,
    seasonal_period=None,
    candidates=None,
    feature_units=None,
    confirmation=None,
    execution_identity=None,
    temporal_result=None,
    reshape_result=None,
    merge_result=None,
    diagnostic_compute_seconds=None,
):
    """Create a source-bound analysis preview without fitting or backend calls.

    Parameters
    ----------
    csv_path : str or Path
        Original wide CSV.
    kind : str
        ``diagnostics`` or ``forecast-baselines``.
    roles : dict or None
        Explicit source roles; None returns clarification.
    methods : list of str or None
        Requested diagnostic methods; empty means profile only.
    period : int or None
        Declared STL observation period.
    period_range : list of float or None
        Physical search period bounds, in seconds.
    scope : str
        ``descriptive_all``, train, val, or test.
    scope_plan : dict or None
        Source-bound existing ingestion plan for a selected split.
    rows : list of int or None
        Explicit original row subset, never implicit sampling.
    expected_frequency : str or None
        Declared fixed cadence for gap counting within observed span.
    time_basis : str or None
        Explicit ``observation_index`` when no time column is supplied.
    targets : list of str or None
        Ordered forecast targets.
    split_policy : str or None
        Chronological evaluation intent.
    seasonal_period : int or None
        Declared seasonal baseline period, in observations.
    candidates : list of str or None
        Fixed last-value and optional seasonal-naive names.
    feature_units : dict or None
        Explicit units or source-bound unit declaration.
    confirmation : dict or None
        Field-level declaration provenance, not authentication.
    execution_identity : dict or None
        Calling analysis implementation identity.
    temporal_result : str, dict or None
        Verified derived-source reference for units and exact inherited splits.
    reshape_result : str, dict or None
        Verified unsplit long-to-wide source reference for confirmed roles/units.
    merge_result : str, dict or None
        Unique independently verified unsplit multi-source parent.
    diagnostic_compute_seconds : int or None
        Explicit diagnostic worker budget: 60 or 120 seconds. The default
        None retains the original 60-second plan and its source-bound hash.
        Forecast-baseline budgets are not overridable.

    Returns
    -------
    dict
        Ready, needs_clarification or blocked finite JSON plan.
    """
    request = copy.deepcopy(
        dict(
            kind=kind,
            roles=roles,
            methods=methods,
            period=period,
            period_range=period_range,
            scope=scope,
            scope_plan=scope_plan,
            rows=rows,
            expected_frequency=expected_frequency,
            time_basis=time_basis,
            targets=targets,
            split_policy=split_policy,
            seasonal_period=seasonal_period,
            candidates=candidates,
            feature_units=feature_units,
            confirmation=confirmation,
            execution_identity=execution_identity,
        )
    )
    plan = dict(schema=SCHEMA, status="needs_clarification", request=request, unresolved=[], errors=[])
    try:
        # Preserve old source-bound hashes unless a bounded diagnostic budget is
        # explicitly selected. Unknown, invalid and other-kind tiers fail closed.
        if diagnostic_compute_seconds is not None:
            if kind != "diagnostics":
                raise ValueError("Diagnostic compute budget applies only to diagnostics")
            if type(diagnostic_compute_seconds) is not int or diagnostic_compute_seconds not in (60, 120):
                raise ValueError("Diagnostic compute budget must be 60 or 120 seconds")
            request["diagnostic_compute_seconds"] = diagnostic_compute_seconds
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
            if derived["split"]:
                if split_policy is not None and split_policy != derived["split"]["policy"]:
                    raise ValueError("Inherited analysis split policy override conflict")
                split_policy = derived["split"]["policy"]
        if kind not in ("diagnostics", "forecast-baselines"):
            raise ValueError("Unsupported analysis kind")
        if roles is None:
            plan["unresolved"].append("Confirm entity, time and selected feature roles; candidates are not consent")
            return _seal(plan)
        source, roles, facts = read_analysis_source(csv_path, roles)
        plan["roles"] = roles
        if not roles["timestamp_col"] and time_basis != "observation_index":
            plan["unresolved"].append("Declare a timestamp or explicitly choose observation_index")
        selected = source.index.tolist() if rows is None else rows
        if (
            not isinstance(selected, list)
            or not selected
            or any(type(i) is not int for i in selected)
            or selected != sorted(set(selected))
            or not set(selected).issubset(source.index)
        ):
            raise ValueError("Selected source rows must be unique, ordered and in bounds")
        if scope != "descriptive_all":
            if scope not in ("train", "val", "test"):
                raise ValueError("A source-bound existing split plan is required")
            inherited = derived["split"] if derived else None
            if scope_plan is not None:
                validate_ingestion_plan(scope_plan, csv_path)
                if roles != scope_plan["roles"]:
                    raise ValueError("Analysis roles differ from the existing split plan")
                if inherited and scope_plan["split"] != inherited:
                    raise ValueError("Explicit analysis scope conflicts with inherited partitions")
                inherited = scope_plan["split"]
            if inherited is None:
                raise ValueError("A source-bound existing split plan is required")
            allowed = set(inherited["rows"][scope])
            selected = [i for i in selected if i in allowed]
        if not selected:
            raise ValueError("Selected analysis scope is empty")
        if period is not None and (type(period) is not int or period < 2):
            raise ValueError("Period must be an integer >= 2")
        if period_range is not None and (
            len(period_range) != 2
            or any(type(v) not in (int, float) for v in period_range)
            or not np.isfinite(period_range).all()
            or not 0 < period_range[0] < period_range[1]
        ):
            raise ValueError("Period range must be two increasing positive seconds")
        units = prepare_feature_semantics(
            feature_units,
            roles["feature_cols"],
            facts["sha256"],
            unit_state=derived["feature_semantics"] if derived else None,
        )
        subset = source.loc[selected]
        groups = analysis_groups(subset, roles)
        if len(groups) * len(roles["feature_cols"]) > 8 or any(len(g) > 4096 for _, g in groups):
            raise ValueError("Analysis exceeds 8 series or 4096 rows per entity; explicitly reduce scope")
        plan.update(
            selected_rows=selected,
            feature_semantics=units,
            confirmation=confirmation or {"roles": "explicit declaration"},
            scope=scope,
            groups=[],
            applicability=[],
        )
        for sid, group in groups:
            temporal = time_profile(group, roles, expected_frequency)
            plan["groups"].append(dict(entity=str(sid), time=temporal))
        if kind == "diagnostics":
            requested = methods or []
            if (
                not isinstance(requested, list)
                or len(set(requested)) != len(requested)
                or not set(requested).issubset(METHODS)
            ):
                raise ValueError("Unknown or duplicate diagnostic method")
            if "stl" in requested and period is None:
                plan["unresolved"].append("Declare the STL period in observations")
            if "lombscargle" in requested and period_range is None:
                plan["unresolved"].append("Declare physical period search bounds in seconds")
            for entity, (_, group) in zip(plan["groups"], groups):
                for feature in roles["feature_cols"]:
                    for method in requested:
                        status, reason = _applicability(
                            method, group[feature].to_numpy(), entity["time"], period, period_range
                        )
                        plan["applicability"].append(
                            dict(entity=entity["entity"], feature=feature, method=method, status=status, reason=reason)
                        )
            plan["budget"] = dict(
                compute_seconds=60 if diagnostic_compute_seconds is None else diagnostic_compute_seconds,
                max_frequency_points=256,
                neural_fit=0,
                neural_predict=0,
                baseline_prepare=0,
                baseline_predict=0,
                provider=0,
                replay=0,
            )
        else:
            if scope != "descriptive_all" or rows is not None:
                raise ValueError("Baseline comparison requires the full explicitly supplied source")
            if split_policy not in ("chronological", "entity_chronological"):
                plan["unresolved"].append("Confirm chronological or entity_chronological evaluation intent")
            if not targets:
                plan["unresolved"].append("Confirm ordered forecast target features")
            elif (
                not isinstance(targets, list)
                or len(set(targets)) != len(targets)
                or not set(targets).issubset(roles["feature_cols"])
            ):
                raise ValueError("Targets must be an ordered unique subset of selected features")
            requested = candidates if candidates is not None else ["last-value", "seasonal-naive"]
            if (
                not requested
                or len(set(requested)) != len(requested)
                or not set(requested).issubset({"last-value", "seasonal-naive"})
            ):
                raise ValueError("Unsupported baseline candidates")
            if "seasonal-naive" in requested and seasonal_period is None:
                plan["unresolved"].append("Declare seasonal_period in observations")
            if seasonal_period is not None and (type(seasonal_period) is not int or seasonal_period < 2):
                raise ValueError("Seasonal period must be an integer >= 2")
            if any(not g["time"]["valid"] or g["time"]["regular_step_seconds"] is None for g in plan["groups"]):
                raise ValueError("Baselines require valid, ordered, regular timestamps per entity")
            if not plan["unresolved"]:
                split = (
                    derived["split"]
                    if derived and derived["split"]
                    else split_source_rows(csv_path, roles, policy=split_policy, allow_irregular=True)
                )
                plan["split"] = split
                for entity, (_, group) in zip(plan["groups"], groups):
                    partitions = {
                        name: [i for i in group.index if i in set(indices)] for name, indices in split["rows"].items()
                    }
                    if not all(partitions.values()) or len(partitions["val"]) + len(partitions["test"]) > 128:
                        raise ValueError("Nonempty splits and at most 128 post-training forecast steps required")
                    entity["partitions"] = partitions
                    entity["origin_row"] = partitions["train"][-1]
                    entity["validation_fh"] = list(range(1, len(partitions["val"]) + 1))
                    entity["test_fh"] = list(
                        range(len(partitions["val"]) + 1, len(partitions["val"]) + len(partitions["test"]) + 1)
                    )
                    for feature in targets:
                        train = source.loc[partitions["train"], feature].to_numpy()
                        for method in requested:
                            sp = 1 if method == "last-value" else seasonal_period
                            good = len(train) >= 2 * sp and np.isfinite(train[-sp:]).all()
                            plan["applicability"].append(
                                dict(
                                    entity=entity["entity"],
                                    feature=feature,
                                    method=method,
                                    status="ready" if good else "unsupported",
                                    reason=None if good else "Required final training context is missing or too short",
                                )
                            )
                pairs = len(groups) * len(targets)
                plan["budget"] = dict(
                    compute_seconds=120,
                    baseline_prepare=pairs * len(requested),
                    baseline_predict=pairs * (len(requested) + 1),
                    neural_fit=0,
                    neural_predict=0,
                    provider=0,
                    replay=0,
                )
        if file_sha256(csv_path) != facts["sha256"]:
            raise ValueError("Source changed while planning")
        plan["status"] = "needs_clarification" if plan["unresolved"] else "ready"
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        plan.update(status="blocked", errors=[str(error)])
    return _seal(plan)


def _seal(plan):
    plan["sha256"] = content_sha256(plan)
    return plan


def validate_analysis_plan(plan, csv_path):
    """Rebuild and check all analysis semantics before execution.

    Parameters
    ----------
    plan : dict
        Ready source-bound analysis plan.
    csv_path : str or Path
        Current source; the path may change but content cannot.

    Returns
    -------
    dict
        Original validated plan.
    """
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA or plan.get("status") != "ready":
        raise ValueError("A ready supported analysis plan is required")
    rebuilt = create_analysis_plan(csv_path, **plan["request"])
    if rebuilt != plan:
        raise ValueError("Analysis source, roles, scope, parameters or identity changed; generate a new plan")
    return plan
