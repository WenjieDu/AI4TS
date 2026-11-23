"""Explicit, bounded exact-time joins; no resampling, fitted state or estimators."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import numpy as np

from .ingestion import content_sha256
from .registry import file_sha256
from .semantics import prepare_feature_semantics
from .temporal_preprocessing import Clarification, _source, declaration, implementation_identity, iso

PLAN_SCHEMA = "timeseries-merge-plan-v1"
RESULT_SCHEMA = "timeseries-merge-result-v1"
LIMITS = dict(
    input_bytes=64 * 1024**2,
    source_cells=10_000_000,
    output_rows=100_000,
    feature_cells=1_000_000,
    output_bytes=32 * 1024**2,
    output_cells=10_000_000,
    contribution_references=1_000_000,
    declaration_bytes=32 * 1024**2,
)
FIELDS = {
    "source_id",
    "path",
    "roles",
    "timezone",
    "feature_units",
    "retain_columns",
    "drop_columns",
    "reshape_result",
    "temporal_result",
}


def safe_path(value):
    """Reject symlinks in the complete explicitly supplied local path."""
    path = Path(value).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Source/result path symlinks are unsupported")
    return path.resolve()


def _parents(sources):
    """Check the fixed four-branch shape before any inspector can recurse."""
    from .temporal_artifacts import _json

    seen = set()
    for source in sources:
        refs = [key for key in ("reshape_result", "temporal_result") if source.get(key) is not None]
        if len(refs) > 1:
            raise ValueError("Choose one derived parent per source")
        if not refs:
            continue
        kind = refs[0]
        ref = source[kind]
        root = safe_path(ref["path"] if isinstance(ref, dict) else ref)
        if str(root) in seen:
            raise ValueError("Duplicate/cyclic parent directory")
        seen.add(str(root))
        plan = _json(root, "plan.json")
        if plan.get("merge_result") or plan.get("request", {}).get("merge_result"):
            raise ValueError("Nested merge parents are unsupported")
        expected = "timeseries-temporal-plan-v1" if kind == "temporal_result" else "timeseries-reshape-plan-v1"
        if plan.get("schema") != expected:
            raise ValueError("Unsupported parent shape")
        prior = plan.get("reshape_result")
        if prior:
            ancestor = safe_path(prior["path"])
            if str(ancestor) in seen:
                raise ValueError("Duplicate/cyclic parent directory")
            seen.add(str(ancestor))
            if _json(ancestor, "plan.json").get("schema") != "timeseries-reshape-plan-v1":
                raise ValueError("Unsupported reshape ancestor")
    if len(seen) > 8:
        raise ValueError("Parent fan-in exceeds eight pre-merge results")


def _inputs(sources, snapshots=None):
    from .time_basis import _derived

    _parents(sources)
    prepared = []
    total_bytes = total_cells = 0
    for item in sources:
        if not isinstance(item, dict) or set(item) - FIELDS:
            raise ValueError("Unsupported source declaration fields")
        source_id = item.get("source_id")
        if not isinstance(source_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", source_id):
            raise ValueError("source_id must be a safe unique identifier")
        path = safe_path(snapshots[source_id] if snapshots is not None else item["path"])
        roles = item.get("roles")
        derived = _derived(path, roles, item.get("temporal_result"), item.get("reshape_result"))
        if derived:
            roles = derived["roles"]
        if not roles or not roles.get("feature_cols"):
            raise Clarification(f"Confirm roles for source {source_id}")
        frame, roles, facts, times = _source(path, roles, item.get("timezone"))
        total_bytes += facts["bytes"]
        total_cells += facts["rows"] * len(facts["columns"])
        if total_bytes > LIMITS["input_bytes"] or total_cells > LIMITS["source_cells"]:
            raise ValueError("Combined source byte/cell limits exceeded")
        state = prepare_feature_semantics(
            item.get("feature_units"),
            roles["feature_cols"],
            facts["sha256"],
            derived["feature_semantics"] if derived else None,
        )
        index = {}
        for row in frame.index:
            entity = str(frame.at[row, roles["sample_id_col"]]) if roles["sample_id_col"] else "0"
            key = (entity, times[row])
            if key in index:
                raise ValueError(f"Duplicate physical key in {source_id}: rows {index[key]} and {row}")
            index[key] = int(row)
        prepared.append(
            dict(
                source_id=source_id,
                frame=frame,
                roles=roles,
                facts=facts,
                times=times,
                index=index,
                semantics=state,
                derived=derived,
                path=path,
            )
        )
    return prepared


def _create(
    sources, join=None, primary=None, entity_meaning=None, single_series=False, preserve_split=False, snapshots=None
):
    request = dict(
        sources=copy.deepcopy(sources),
        join=join,
        primary=primary,
        entity_meaning=entity_meaning,
        single_series=single_series,
        preserve_split=preserve_split,
    )
    result = dict(
        schema=PLAN_SCHEMA,
        status="needs_clarification",
        request=request,
        unresolved=[],
        errors=[],
        implementation=implementation_identity(),
        limits=dict(LIMITS),
        budget=dict(run_seconds=60, inspect_seconds=30),
    )
    try:
        if not isinstance(sources, list) or not 2 <= len(sources) <= 4:
            raise ValueError("Declare two to four wide sources")
        if len(json.dumps(request, allow_nan=False).encode()) > LIMITS["declaration_bytes"]:
            raise ValueError("Source declarations exceed the bounded JSON reader")
        if preserve_split is not False:
            raise ValueError("Preserving input evaluation splits is unsupported; merge is unsplit")
        if join is None:
            raise Clarification("Choose left or outer join")
        if join not in ("left", "outer") or type(single_series) is not bool:
            raise ValueError("Unsupported join or single-series declaration")
        if any(not isinstance(s, dict) for s in sources):
            raise ValueError("Each source must be an object")
        for source in request["sources"]:
            source["path"] = str(safe_path(source["path"]))
        sources = request["sources"]
        ids = [s.get("source_id") for s in sources]
        if len(set(ids)) != len(ids):
            raise ValueError("source_id values must be unique")
        primary = primary if primary is not None else ids[0]
        if primary not in ids:
            raise ValueError("Primary source must be declared")
        items = _inputs(sources, snapshots)
        main = next(s for s in items if s["source_id"] == primary)
        identified = [bool(s["roles"]["sample_id_col"]) for s in items]
        if any(identified) != all(identified):
            raise ValueError("Mixed entity/no-entity inputs cannot be broadcast")
        if all(identified) and (not isinstance(entity_meaning, str) or not entity_meaning.strip()):
            raise Clarification("Confirm the common entity identifier meaning")
        if not any(identified) and single_series is not True:
            raise Clarification("Confirm that all inputs describe one series")
        feature_sources, units = {}, {}
        retained = []
        source_bindings = []
        for item, declared in zip(items, sources):
            sid, roles = item["source_id"], item["roles"]
            special = {v for k, v in roles.items() if k != "feature_cols" and v is not None}
            metadata = set(item["facts"]["columns"]) - set(roles["feature_cols"]) - special
            label = roles["label_col"]
            if sid != primary and label:
                raise ValueError("Auxiliary task labels are unsupported; do not silently project them away")
            if label:
                metadata.add(label)
            keep, drop = declared.get("retain_columns"), declared.get("drop_columns")
            if keep is None or drop is None:
                raise Clarification(f"Declare retained/excluded metadata for {sid}, including empty lists")
            if (
                not isinstance(keep, list)
                or not isinstance(drop, list)
                or len(set(keep + drop)) != len(keep + drop)
                or set(keep + drop) != metadata
                or (sid != primary and keep)
            ):
                raise ValueError(f"Explicit metadata projection conflict in {sid}")
            if sid == primary:
                retained = keep
            for feature in roles["feature_cols"]:
                if feature in feature_sources:
                    raise ValueError("Output feature names conflict; explicitly project variables")
                feature_sources[feature] = sid
                units[feature] = item["semantics"]["units"][feature] if item["semantics"] else "unknown"
            binding = dict(
                source_id=sid,
                source={k: item["facts"][k] for k in ("sha256", "bytes", "rows", "columns")},
                roles=roles,
                timezone=declared.get("timezone"),
                feature_semantics=item["semantics"],
            )
            if item["derived"]:
                key = "temporal_result" if declared.get("temporal_result") is not None else "reshape_result"
                request["sources"][len(source_bindings)][key] = item["derived"]["reference"]
                binding[key] = item["derived"]["reference"]
            source_bindings.append(binding)
        structure = [main["roles"][k] for k in ("sample_id_col", "timestamp_col") if main["roles"][k]]
        columns = [*structure, *feature_sources, *retained]
        if len(set(columns)) != len(columns):
            raise ValueError("Output feature/role/retained metadata names conflict")
        keys = set(main["index"])
        if join == "outer":
            for item in items:
                keys.update(item["index"])
        count, width = len(keys), len(feature_sources)
        if (
            count > LIMITS["output_rows"]
            or count * width > LIMITS["feature_cells"]
            or count * len(columns) > LIMITS["output_cells"]
        ):
            raise ValueError("Merge output row/cell limits exceeded before allocation")
        entities = list(
            dict.fromkeys(e for item in [main, *(s for s in items if s is not main)] for e, _ in item["index"])
        )
        order = {e: i for i, e in enumerate(entities)}
        ordered = sorted(keys, key=lambda k: (order[k[0]], k[1]))
        from .merge_artifacts import _missing_reason, _parent_rows, _reference_count

        parents = _parent_rows(items)
        references = 0
        for item in items:
            sid = item["source_id"]
            for key in ordered:
                row = item["index"].get(key)
                if row is None:
                    continue
                for feature in item["roles"]["feature_cols"]:
                    cell = parents[sid][row]["cells"][feature] if sid in parents else {}
                    references += _reference_count(cell)
        if references > LIMITS["contribution_references"]:
            raise ValueError("Cell contribution reference limit exceeded before allocation")
        byte_bound = 2 * sum(len(c.encode()) + 3 for c in columns)
        for entity, point in ordered:
            row = main["index"].get((entity, point))
            meta_bytes = sum(len(str(main["frame"].at[row, c]).encode()) for c in retained) if row is not None else 0
            byte_bound += 2 * (len(entity.encode()) + meta_bytes) + 48 + width * 26 + 3 * len(columns)
        if byte_bound > LIMITS["output_bytes"]:
            raise ValueError("Estimated output CSV byte limit exceeded")
        rows = [
            dict(entity=e, time_ns=t, source_rows={s["source_id"]: s["index"].get((e, t)) for s in items})
            for e, t in ordered
        ]
        features = {}
        by_source = {}
        for item in items:
            sid = item["source_id"]
            selected = [item["index"][k] for k in ordered if k in item["index"]]
            by_source[sid] = dict(
                input_rows=len(item["frame"]),
                matched_keys=len(selected),
                unmatched_output_keys=count - len(selected),
                output_key_denominator=count,
                excluded_rows=sorted(set(item["frame"].index) - set(selected)),
            )
            for feature in item["roles"]["feature_cols"]:
                finite = int(np.isfinite(item["frame"].loc[selected, feature].to_numpy()).sum())
                features[feature] = dict(
                    finite=finite, original_missing=len(selected) - finite, unmatched=count - len(selected)
                )
                if sid in parents:
                    reasons = {}
                    for row in selected:
                        if not np.isfinite(item["frame"].at[row, feature]):
                            reason = _missing_reason(parents[sid][row]["cells"][feature])
                            reasons[reason] = reasons.get(reason, 0) + 1
                    features[feature].update(original_missing=0, parent_missing=reasons)
        source_digest = content_sha256({s["source_id"]: s["facts"]["sha256"] for s in items})
        roles = dict(
            main["roles"],
            feature_cols=list(feature_sources),
            label_col=main["roles"]["label_col"] if main["roles"]["label_col"] in retained else None,
        )
        result.update(
            status="ready",
            primary=primary,
            sources=source_bindings,
            rows=rows,
            columns=columns,
            roles=roles,
            retained=retained,
            feature_sources=feature_sources,
            split=None,
            feature_semantics=prepare_feature_semantics(units, list(feature_sources), source_digest),
            preview=dict(
                output_rows=count,
                feature_cells=count * width,
                sources=by_source,
                features=features,
                split_status="unsplit",
                estimated_csv_bytes_upper_bound=byte_bound,
            ),
        )
        for item in items:
            if file_sha256(item["path"]) != item["facts"]["sha256"]:
                raise ValueError("Source changed while planning")
    except Clarification as error:
        result.update(status="needs_clarification", unresolved=[str(error)])
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        result.update(status="blocked", errors=[str(error)])
    result["sha256"] = content_sha256(result)
    if len(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode()) > LIMITS["declaration_bytes"]:
        result = dict(
            schema=PLAN_SCHEMA,
            status="blocked",
            unresolved=[],
            errors=["Merge plan byte limit exceeded"],
            limits=dict(LIMITS),
        )
        result["sha256"] = content_sha256(result)
    return result


def create_merge_plan(sources, join=None, primary=None, entity_meaning=None, single_series=False, preserve_split=False):
    """Plan two to four declared wide sources without producing data.

    Parameters
    ----------
    sources : list of dict
        Ordered source_id/path/roles/timezone, unit and metadata projections,
        with at most one verified reshape/temporal parent per source.
    join : str or None
        Explicit left or outer key set. Missing intent requires clarification.
    primary : str or None
        Declared source ID, defaulting to the first source.
    entity_meaning : str or None
        Confirmation that entity identifiers denote the same entities.
    single_series : bool
        Explicit consent for all inputs without an entity role to be one series.
    preserve_split : bool
        True is unsupported; merging always creates an unsplit source.

    Returns
    -------
    dict
        Content-bound ready, needs_clarification or blocked contract.
    """
    return _create(sources, join, primary, entity_meaning, single_series, preserve_split)


def validate_merge_plan(plan, *, snapshots=None):
    """Recompute a ready contract; inspectors may read its bundled snapshots.

    Parameters
    ----------
    plan : dict
        Unchanged ready plan.
    snapshots : dict or None
        Internal source_id-to-snapshot paths; original declarations remain bound.

    Returns
    -------
    dict
        The validated unchanged plan.
    """
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "ready"
        or _create(**plan["request"], snapshots=snapshots) != plan
    ):
        raise ValueError("Merge plan/source/parent/implementation changed")
    return plan


def run_merge(plan, output, timeout=60):
    """Produce a ready merge into a new directory within (0, 60] seconds."""
    from .temporal_artifacts import supervise

    return supervise("run", ".", plan, safe_path(output), timeout, kind="merge")


def inspect_merge_result(result, timeout=30):
    """Independently inspect saved data within (0, 30] seconds, without production."""
    from .temporal_artifacts import supervise

    return supervise("inspect", result, None, None, timeout, kind="merge")


def merge_context(result, csv_path, roles=None):
    """Verify the unique unsplit derived source for a downstream consumer."""
    from . import temporal_artifacts
    from .merge_artifacts import _verify

    supplied = declaration(result) if isinstance(result, dict) else None
    root = safe_path(supplied["path"] if supplied else result)
    digest = file_sha256(root / "result.json")
    if supplied and (
        set(supplied) != {"path", "result_sha256", "derived_sha256"} or supplied["result_sha256"] != digest
    ):
        raise ValueError("Merge reference changed")
    verified = _verify(root) if temporal_artifacts._IN_WORKER else inspect_merge_result(root)
    if verified["status"] != "success" or file_sha256(csv_path) != verified["derived_sha256"]:
        raise ValueError("Merge result verification/source binding failed: " + verified.get("error", "unknown"))
    if supplied and supplied["derived_sha256"] != verified["derived_sha256"]:
        raise ValueError("Merge derived identity changed")
    if file_sha256(root / "result.json") != digest:
        raise ValueError("Merge result changed while reading")
    bound = verified["roles"]
    if roles is not None:
        requested = dict(copy.deepcopy(bound), **roles)
        if (
            set(roles) - set(bound)
            or not requested["feature_cols"]
            or len(set(requested["feature_cols"])) != len(requested["feature_cols"])
            or not set(requested["feature_cols"]).issubset(bound["feature_cols"])
            or any(requested[k] != bound[k] for k in bound if k != "feature_cols")
        ):
            raise ValueError("Merge derived roles conflict")
        bound = requested
    semantics = verified["feature_semantics"]
    if bound["feature_cols"] != semantics["feature_columns"]:
        semantics = prepare_feature_semantics(
            {f: semantics["units"][f] for f in bound["feature_cols"]}, bound["feature_cols"], verified["derived_sha256"]
        )
    return dict(
        reference=dict(path=str(root), result_sha256=digest, derived_sha256=verified["derived_sha256"]),
        roles=bound,
        feature_semantics=semantics,
        split=None,
    )
