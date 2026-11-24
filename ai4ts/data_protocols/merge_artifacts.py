"""Supervised merge production and independent snapshot/value verification."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .merge import LIMITS, RESULT_SCHEMA, _inputs, safe_path, validate_merge_plan
from .registry import file_sha256
from .semantics import prepare_feature_semantics
from .temporal_artifacts import LIMITATIONS, _json, _read, _require, write_json
from .temporal_preprocessing import iso

MERGE_LIMITATIONS = [
    "Exact physical-time keys do not establish simultaneous availability, common sampling, IID or model readiness.",
    "No input evaluation split, fitted scaler, model, threshold or window state is inherited.",
    "Missing values remain missing; unmatched, original-empty and derived-parent gaps have distinct lineage.",
    "unknown is an explicit unknown unit, not an inferred unit or a conversion.",
]


def _reference_count(cell):
    return (
        1
        + len(cell.get("source_rows", cell.get("contributors", [])))
        + sum(1 + len(ancestor["evidence"].get("source_rows", [])) for ancestor in cell.get("ancestor_cells", []))
    )


def _missing_reason(cell):
    reason = cell.get("unknown_reason")
    ancestors = {entry["evidence"].get("unknown_reason") for entry in cell.get("ancestor_cells", [])}
    if reason in ("original-missing", "all-missing") and len(ancestors) == 1:
        ancestor = next(iter(ancestors))
        if ancestor in ("absent-measurement", "recorded-missing"):
            return ancestor
    return reason


def _parent_rows(items):
    rows = {}
    references = 0
    for item in items:
        derived = item["derived"]
        if not derived:
            continue
        root = safe_path(derived["reference"]["path"])
        parent_plan = _json(root, "plan.json")
        ancestor = parent_plan.get("reshape_result")
        ancestor_rows = []
        if ancestor:
            with _read(safe_path(ancestor["path"]), "lineage.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    ancestor_rows.append({f: row["cells"][f] for f in item["roles"]["feature_cols"]})
        values = []
        with _read(root, "lineage.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                value = json.loads(line)
                if value["row"] != len(values):
                    raise ValueError("Parent lineage row order differs")
                values.append(value)
                for feature in item["roles"]["feature_cols"]:
                    cell = value["cells"][feature]
                    if ancestor and value["source_rows"]:
                        cell["ancestor_cells"] = [
                            dict(reference=ancestor, row=row, feature=feature, evidence=ancestor_rows[row][feature])
                            for row in value["source_rows"]
                        ]
                    references += _reference_count(cell)
                if references > LIMITS["contribution_references"]:
                    raise ValueError("Parent cell contribution reference limit exceeded")
        if len(values) != len(item["frame"]):
            raise ValueError("Parent lineage shape differs")
        rows[item["source_id"]] = values
    return rows


def _produce(_source, plan, root):
    validate_merge_plan(plan)
    snapshots = {}
    for source in plan["request"]["sources"]:
        folder = root / "sources" / source["source_id"]
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / "source.csv"
        shutil.copyfile(source["path"], path)
        snapshots[source["source_id"]] = path
    validate_merge_plan(plan, snapshots=snapshots)
    write_json(root / "plan.json", plan)
    items = _inputs(plan["request"]["sources"], snapshots)
    by_id = {s["source_id"]: s for s in items}
    parent_rows = _parent_rows(items)
    count = 0
    with (
        (root / "data.csv").open("x", encoding="utf-8", newline="") as output,
        (root / "lineage.jsonl").open("x", encoding="utf-8") as lineage,
    ):
        writer = csv.DictWriter(output, fieldnames=plan["columns"])
        writer.writeheader()
        for number, row in enumerate(plan["rows"]):
            record = {c: "" for c in plan["columns"]}
            if plan["roles"]["sample_id_col"]:
                record[plan["roles"]["sample_id_col"]] = row["entity"]
            record[plan["roles"]["timestamp_col"]] = iso(row["time_ns"])
            primary = by_id[plan["primary"]]
            main_row = row["source_rows"][plan["primary"]]
            if main_row is not None:
                record.update({c: primary["frame"].at[main_row, c] for c in plan["retained"]})
            trace = dict(
                row=number,
                entity=row["entity"],
                time=iso(row["time_ns"]),
                cells={},
                metadata_unknown=list(plan["retained"]) if main_row is None else [],
            )
            for feature, sid in plan["feature_sources"].items():
                item = by_id[sid]
                source_row = row["source_rows"][sid]
                value = item["frame"].at[source_row, feature] if source_row is not None else np.nan
                cell = dict(
                    source_id=sid,
                    source_row=source_row,
                    original_time=(
                        str(item["frame"].at[source_row, item["roles"]["timestamp_col"]])
                        if source_row is not None
                        else None
                    ),
                    status=(
                        "unmatched-source"
                        if source_row is None
                        else "observed" if np.isfinite(value) else "original-missing"
                    ),
                )
                if source_row is not None and sid in parent_rows:
                    evidence = parent_rows[sid][source_row]["cells"][feature]
                    cell["parent"] = dict(
                        reference=item["derived"]["reference"], row=source_row, feature=feature, evidence=evidence
                    )
                    if _missing_reason(evidence):
                        cell["status"] = "parent-" + _missing_reason(evidence)
                    count += _reference_count(evidence)
                else:
                    count += source_row is not None
                if count > LIMITS["contribution_references"]:
                    raise ValueError("Cell contribution reference limit exceeded")
                record[feature] = format(float(value), ".17g") if np.isfinite(value) else ""
                trace["cells"][feature] = cell
            writer.writerow(record)
            lineage.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")
    if (root / "data.csv").stat().st_size > LIMITS["output_bytes"]:
        raise ValueError("Actual output byte limit exceeded")
    digest = file_sha256(root / "data.csv")
    semantics = prepare_feature_semantics(None, plan["roles"]["feature_cols"], digest, plan["feature_semantics"])
    write_json(root / "feature-semantics.json", semantics)
    disposition = {}
    for item in items:
        included = {
            r["source_rows"][item["source_id"]]: n
            for n, r in enumerate(plan["rows"])
            if r["source_rows"][item["source_id"]] is not None
        }
        disposition[item["source_id"]] = [
            dict(
                source_row=i,
                output_row=included.get(i),
                status="included" if i in included else "excluded-auxiliary-key",
            )
            for i in range(len(item["frame"]))
        ]
    write_json(root / "row-disposition.json", disposition)
    write_json(root / "match-summary.json", plan["preview"])
    with (root / "report.md").open("x", encoding="utf-8") as report:
        report.write(
            "# Exact multi-source merge\n\n"
            + json.dumps(plan["preview"], ensure_ascii=False, indent=2)
            + "\n\n"
            + "\n".join(MERGE_LIMITATIONS + LIMITATIONS)
            + "\n"
        )
    write_json(
        root / "execution.json",
        dict(
            status="success", merge_runs=1, model_fit=0, model_predict=0, statistical_estimation=0, replay=0, provider=0
        ),
    )
    write_json(
        root / "pending-result.json",
        dict(
            schema=RESULT_SCHEMA,
            status="success",
            derived_sha256=digest,
            plan_sha256=plan["sha256"],
            sources=plan["sources"],
            roles=plan["roles"],
            split_status="unsplit",
            limitations=MERGE_LIMITATIONS,
        ),
    )
    return dict(status="success", result=str(root), data=str(root / "data.csv"))


def _verify(root):
    root = safe_path(root)
    saved = _json(root, "result.json")
    _require(
        saved.get("schema") == RESULT_SCHEMA and saved.get("status") == "success", "Incomplete/unsupported merge result"
    )
    required = {
        "plan.json",
        "data.csv",
        "lineage.jsonl",
        "row-disposition.json",
        "match-summary.json",
        "feature-semantics.json",
        "execution.json",
        "report.md",
        "supervisor.json",
        "worker.stdout.log",
        "worker.stderr.log",
    }
    _require(required.issubset(saved["files"]), "Incomplete merge inventory")
    for name, digest in saved["files"].items():
        _require(file_sha256(_read(root, name)) == digest, "Artifact hash mismatch: " + name)
    plan = _json(root, "plan.json")
    snapshots = {s["source_id"]: _read(root, "sources/" + s["source_id"] + "/source.csv") for s in plan["sources"]}
    _require(all("sources/" + sid + "/source.csv" in saved["files"] for sid in snapshots), "Unbound snapshot")
    validate_merge_plan(plan, snapshots=snapshots)
    items = _inputs(plan["request"]["sources"], snapshots)
    parents = _parent_rows(items)
    by_id = {s["source_id"]: s for s in items}
    main = by_id[plan["primary"]]
    keys = set(main["index"])
    if plan["request"]["join"] == "outer":
        keys |= {key for s in items for key in s["index"]}
    entities = list(
        dict.fromkeys(entity for s in [main, *(s for s in items if s is not main)] for entity, _ in s["index"])
    )
    order = {entity: i for i, entity in enumerate(entities)}
    keys = sorted(keys, key=lambda key: (order[key[0]], key[1]))
    data_path = _read(root, "data.csv")
    _require(data_path.stat().st_size <= LIMITS["output_bytes"], "Oversized merged CSV")
    data = pd.read_csv(data_path, dtype=str, keep_default_na=False)
    _require(list(data.columns) == plan["columns"] and len(data) == len(keys), "Merged shape/order differs")
    digest = file_sha256(data_path)
    _require(
        digest == saved["derived_sha256"]
        and saved["plan_sha256"] == plan["sha256"]
        and saved["sources"] == plan["sources"]
        and saved["roles"] == plan["roles"]
        and saved["split_status"] == "unsplit",
        "Merge result identity differs",
    )
    disposition = {s["source_id"]: [] for s in items}
    destinations = {key: i for i, key in enumerate(keys)}
    for s in items:
        reverse = {row: destinations.get(key) for key, row in s["index"].items()}
        disposition[s["source_id"]] = [
            dict(
                source_row=i,
                output_row=reverse[i],
                status="included" if reverse[i] is not None else "excluded-auxiliary-key",
            )
            for i in range(len(s["frame"]))
        ]
    _require(_json(root, "row-disposition.json") == disposition, "Source row disposition differs")
    _require(_json(root, "match-summary.json") == plan["preview"], "Match/support summary differs")
    with _read(root, "lineage.jsonl").open(encoding="utf-8") as stream:
        for number, key in enumerate(keys):
            entity, time = key
            trace = json.loads(next(stream))
            actual = data.iloc[number]
            _require(
                trace["row"] == number and trace["entity"] == entity and trace["time"] == iso(time),
                "Lineage key differs",
            )
            _require(actual[plan["roles"]["timestamp_col"]] == iso(time), "Output physical time differs")
            if plan["roles"]["sample_id_col"]:
                _require(actual[plan["roles"]["sample_id_col"]] == entity, "Output entity differs")
            row = main["index"].get(key)
            _require(
                trace["metadata_unknown"] == (list(plan["retained"]) if row is None else []),
                "Metadata unknown facts differ",
            )
            for column in plan["retained"]:
                _require(
                    actual[column] == (str(main["frame"].at[row, column]) if row is not None else ""),
                    "Metadata/label differs",
                )
            _require(set(trace["cells"]) == set(plan["feature_sources"]), "Lineage feature coverage differs")
            for feature, sid in plan["feature_sources"].items():
                source = by_id[sid]
                position = source["index"].get(key)
                value = source["frame"].at[position, feature] if position is not None else np.nan
                finite = np.isfinite(value)
                _require(
                    (
                        actual[feature] == ""
                        if not finite
                        else actual[feature] != "" and float(actual[feature]) == float(value)
                    ),
                    "Merged cell differs from source snapshot",
                )
                expected = dict(
                    source_id=sid,
                    source_row=position,
                    original_time=(
                        str(source["frame"].at[position, source["roles"]["timestamp_col"]])
                        if position is not None
                        else None
                    ),
                    status="unmatched-source" if position is None else "observed" if finite else "original-missing",
                )
                if position is not None and sid in parents:
                    evidence = parents[sid][position]["cells"][feature]
                    expected["parent"] = dict(
                        reference=source["derived"]["reference"], row=position, feature=feature, evidence=evidence
                    )
                    if _missing_reason(evidence):
                        expected["status"] = "parent-" + _missing_reason(evidence)
                _require(trace["cells"][feature] == expected, "Cell provenance differs")
        _require(next(stream, None) is None, "Extra lineage rows")
    semantics = prepare_feature_semantics(None, plan["roles"]["feature_cols"], digest, plan["feature_semantics"])
    _require(_json(root, "feature-semantics.json") == semantics, "Derived units differ")
    execution = _json(root, "execution.json")
    _require(
        execution
        == dict(
            status="success", merge_runs=1, model_fit=0, model_predict=0, statistical_estimation=0, replay=0, provider=0
        ),
        "Unexpected execution facts",
    )
    return dict(
        status="success",
        derived_sha256=digest,
        roles=plan["roles"],
        feature_semantics=semantics,
        split=None,
        split_status="unsplit",
        result=str(root),
        limitations=MERGE_LIMITATIONS,
    )


def _work(request):
    from . import temporal_artifacts

    previous = temporal_artifacts._IN_WORKER
    temporal_artifacts._IN_WORKER = True
    try:
        if request["operation"] == "run":
            return _produce(None, request["plan"], Path(request["output"]))
        return _verify(Path(request["source"]))
    except Exception as error:
        return dict(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        temporal_artifacts._IN_WORKER = previous
