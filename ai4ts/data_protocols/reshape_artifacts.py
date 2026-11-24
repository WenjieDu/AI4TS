"""Long-to-wide production and independent cell/source verification."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil

import pandas as pd

from .ingestion import inspect_csv
from .registry import file_sha256
from .reshape import RESULT_SCHEMA, _read_long, validate_reshape_plan
from .semantics import prepare_feature_semantics
from .temporal_artifacts import _json, _read, _require, write_json
from .temporal_preprocessing import iso

_IN_WORKER = False
_VERIFY_STACK = set()
LIMITATIONS = [
    "Long-to-wide preparation does not align nearby clocks, add a grid, impute or aggregate readings.",
    "Absent measurement records and recorded missing readings both remain NaN with different source reasons.",
    "The result is unsplit; no old evaluation, fitted scaler, model or threshold is inherited.",
    "Reshaping does not establish IID, distribution stability, model applicability or production qualification.",
]


def _produce(source_path, plan, root):
    validate_reshape_plan(plan, source_path)
    shutil.copyfile(source_path, root / "source.csv")
    validate_reshape_plan(plan, root / "source.csv")
    write_json(root / "plan.json", plan)
    source, roles, _, _ = _read_long(root / "source.csv", plan["roles"], plan["request"]["timezone"])
    variables = plan["request"]["variables"]
    retained = []
    with (
        (root / "data.csv").open("x", encoding="utf-8", newline="") as stream,
        (root / "lineage.jsonl").open("x", encoding="utf-8") as lineage,
    ):
        writer = csv.DictWriter(stream, fieldnames=plan["columns"])
        writer.writeheader()
        for number, row in enumerate(plan["rows"]):
            record = dict(row["metadata"])
            if roles["sample_id_col"]:
                record[roles["sample_id_col"]] = row["entity"]
            record[roles["timestamp_col"]] = iso(row["time_ns"])
            item = dict(row=number, entity=row["entity"], time=iso(row["time_ns"]), cells={})
            for variable in variables:
                cell = row["measurements"].get(variable, dict(source_rows=[], selected_row=None))
                chosen = cell["selected_row"]
                raw = source.at[chosen, roles["value_col"]] if chosen is not None else ""
                value = float(raw) if raw != "" else math.nan
                finite = math.isfinite(value)
                record[variable] = format(value, ".17g") if finite else ""
                item["cells"][variable] = dict(
                    source_rows=cell["source_rows"],
                    selected_row=chosen,
                    finite_count=int(finite),
                    unknown_reason=(
                        None if finite else "recorded-missing" if chosen is not None else "absent-measurement"
                    ),
                )
                if chosen is not None:
                    retained.append(chosen)
            writer.writerow(record)
            lineage.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(
        root / "row-disposition.json",
        dict(retained=sorted(retained), discarded=plan["discarded"], excluded=plan["excluded"]),
    )
    derived_sha = file_sha256(root / "data.csv")
    semantics = prepare_feature_semantics(None, variables, derived_sha, unit_state=plan["feature_semantics"])
    if semantics:
        write_json(root / "feature-semantics.json", semantics)
    summary = dict(
        plan["preview"], output_missing=plan["preview"]["absent_measurements"] + plan["preview"]["recorded_missing"]
    )
    with (root / "report.md").open("x", encoding="utf-8") as stream:
        stream.write(
            "# Long-to-wide preparation\n\n"
            + json.dumps(summary, ensure_ascii=False, indent=2)
            + "\n\n"
            + "\n".join(LIMITATIONS)
            + "\n"
        )
    write_json(
        root / "execution.json",
        dict(
            status="success",
            reshape_runs=1,
            model_fit=0,
            model_predict=0,
            statistical_estimation=0,
            scaler_fit=0,
            replay=0,
            provider=0,
        ),
    )
    result = dict(
        schema=RESULT_SCHEMA,
        status="success",
        source_sha256=plan["source"]["sha256"],
        derived_sha256=derived_sha,
        plan_sha256=plan["sha256"],
        roles=plan["wide_roles"],
        summary=summary,
        split_status="unsplit",
        limitations=LIMITATIONS,
        files={p.name: file_sha256(p) for p in root.iterdir() if p.is_file()},
    )
    write_json(root / "pending-result.json", result)
    return dict(
        status="success",
        result=str(root),
        data=str(root / "data.csv"),
        report=str(root / "report.md"),
        summary=summary,
        split_status="unsplit",
    )


def _verify_data(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Invalid reshape result directory")
    saved = _json(root, "result.json")
    _require(saved.get("schema") == RESULT_SCHEMA and saved.get("status") == "success", "Incomplete/unsupported result")
    required = {
        "source.csv",
        "plan.json",
        "data.csv",
        "lineage.jsonl",
        "row-disposition.json",
        "execution.json",
        "report.md",
        "supervisor.json",
        "worker.stdout.log",
        "worker.stderr.log",
    }
    _require(required.issubset(saved["files"]), "Incomplete artifact inventory")
    for name, digest in saved["files"].items():
        _require(file_sha256(_read(root, name)) == digest, "Artifact hash mismatch: " + name)
    plan = _json(root, "plan.json")
    validate_reshape_plan(plan, _read(root, "source.csv"))
    source, roles, _, times = _read_long(root / "source.csv", plan["roles"], plan["request"]["timezone"])
    inspect_csv(_read(root, "data.csv"))
    data = pd.read_csv(root / "data.csv", dtype=str, keep_default_na=False)
    _require(
        list(data.columns) == plan["columns"] and len(data) == plan["preview"]["output_rows"], "Derived shape differs"
    )
    _require(
        saved["source_sha256"] == file_sha256(root / "source.csv")
        and saved["derived_sha256"] == file_sha256(root / "data.csv")
        and saved["plan_sha256"] == plan["sha256"]
        and saved["roles"] == plan["wide_roles"],
        "Source/roles/plan identity differs",
    )
    _require(saved["split_status"] == "unsplit" and not (root / "split.json").exists(), "Unexpected evaluation split")
    _require(_json(root, "supervisor.json")["status"] == "success", "Supervisor did not complete")
    _require(
        _json(root, "execution.json")
        == dict(
            status="success",
            reshape_runs=1,
            model_fit=0,
            model_predict=0,
            statistical_estimation=0,
            scaler_fit=0,
            replay=0,
            provider=0,
        ),
        "Execution accounting differs",
    )
    variables, members, entities, excluded = plan["request"]["variables"], {}, {}, []
    chosen_names = set(variables)
    for index in source.index:
        variable = source.at[index, roles["variable_col"]]
        if variable not in chosen_names:
            excluded.append(int(index))
            continue
        entity = source.at[index, roles["sample_id_col"]] if roles["sample_id_col"] else "0"
        entities.setdefault(entity, len(entities))
        members.setdefault((entity, times[index]), {}).setdefault(variable, []).append(int(index))
    keys = sorted(members, key=lambda item: (entities[item[0]], item[1]))
    retained, discarded = [], []
    absent = missing = finite = 0
    with _read(root, "lineage.jsonl").open(encoding="utf-8") as stream:
        number = 0
        for number, line in enumerate(stream, start=1):
            _require(number <= len(keys), "Extra lineage row")
            index = number - 1
            entity, instant = keys[index]
            item = json.loads(line)
            _require(
                item["row"] == index and item["entity"] == entity and item["time"] == iso(instant),
                "Lineage key/order differs",
            )
            _require(data.at[index, roles["timestamp_col"]] == iso(instant), "Derived physical time differs")
            if roles["sample_id_col"]:
                _require(data.at[index, roles["sample_id_col"]] == entity, "Entity string differs")
            _require(set(item["cells"]) == chosen_names, "Measurement columns differ")
            selected = []
            for variable in variables:
                indexes = members[(entity, instant)].get(variable, [])
                chosen = (
                    (indexes[-1] if plan["request"]["duplicate_policy"] == "keep-last" else indexes[0])
                    if indexes
                    else None
                )
                raw = source.at[chosen, roles["value_col"]] if chosen is not None else ""
                expected = float(raw) if raw != "" else math.nan
                actual = float(data.at[index, variable]) if data.at[index, variable] else math.nan
                _require(
                    actual == expected or math.isnan(actual) and math.isnan(expected), "Copied measurement differs"
                )
                known = math.isfinite(expected)
                reason = None if known else "recorded-missing" if indexes else "absent-measurement"
                _require(
                    item["cells"][variable]
                    == dict(source_rows=indexes, selected_row=chosen, finite_count=int(known), unknown_reason=reason),
                    "Measurement source/support differs",
                )
                absent += int(not indexes)
                missing += int(bool(indexes) and not known)
                finite += int(known)
                if chosen is not None:
                    selected.append(chosen)
                    retained.append(chosen)
                    discarded.extend(
                        dict(row=i, retained=chosen, reason="unselected-duplicate") for i in indexes if i != chosen
                    )
            for column in plan["metadata"]:
                labels = {source.at[i, column] for i in selected}
                _require(len(labels) == 1 and data.at[index, column] == next(iter(labels)), "Metadata/label differs")
        _require(number == len(keys), "Incomplete lineage coverage")
    expected_disposition = dict(retained=sorted(retained), discarded=discarded, excluded=excluded)
    # Producer follows source variable appearance; compare duplicate dispositions by original row.
    disposition = _json(root, "row-disposition.json")
    _require(
        disposition["retained"] == expected_disposition["retained"]
        and disposition["excluded"] == excluded
        and sorted(disposition["discarded"], key=lambda x: x["row"]) == sorted(discarded, key=lambda x: x["row"]),
        "Original row dispositions differ",
    )
    summary = dict(
        plan["preview"],
        absent_measurements=absent,
        recorded_missing=missing,
        finite_measurements=finite,
        unselected_duplicates=len(discarded),
        excluded_variable_records=len(excluded),
        output_missing=absent + missing,
    )
    _require(saved["summary"] == summary, "Missingness/support summary differs")
    semantics = prepare_feature_semantics(
        None, variables, saved["derived_sha256"], unit_state=plan["feature_semantics"]
    )
    if semantics:
        _require(_json(root, "feature-semantics.json") == semantics, "Derived unit lineage differs")
    else:
        _require(not (root / "feature-semantics.json").exists(), "Unexpected unit claim")
    return dict(
        status="success",
        result_status=saved["status"],
        result=str(root),
        data=str(root / "data.csv"),
        report=str(root / "report.md"),
        roles=plan["wide_roles"],
        feature_semantics=semantics,
        split=None,
        split_status="unsplit",
        summary=summary,
        derived_sha256=saved["derived_sha256"],
        source_sha256=saved["source_sha256"],
        plan_sha256=plan["sha256"],
        verification="source_keys_and_independent_cell_copy",
        limitations=LIMITATIONS,
    )


def _verify(root):
    key = str(root.resolve())
    if key in _VERIFY_STACK:
        raise ValueError("Cyclic reshape references")
    _VERIFY_STACK.add(key)
    try:
        return _verify_data(root)
    finally:
        _VERIFY_STACK.remove(key)


def _work(request):
    global _IN_WORKER
    before = _IN_WORKER
    _IN_WORKER = True
    try:
        if request["operation"] == "run":
            return _produce(Path(request["source"]), request["plan"], Path(request["output"]))
        return _verify(Path(request["source"]))
    except Exception as error:
        return dict(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        _IN_WORKER = before
