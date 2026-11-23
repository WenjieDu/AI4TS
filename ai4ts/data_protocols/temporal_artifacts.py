"""Bounded temporal workers and independent saved-data verification."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .ingestion import content_sha256
from .registry import file_sha256
from .semantics import prepare_feature_semantics
from .temporal_preprocessing import RESULT_SCHEMA, _role, _source, iso, validate_temporal_plan

LIMITATIONS = [
    "Time preparation does not impute measurements, certify IID or establish model applicability.",
    "Aggregates are irreversible derived values, not original observations or anti-aliased signal reconstruction.",
    "Bucket end labels do not certify real-time data availability or complete sampling.",
    "No original fitted scaler, model, threshold or training state is inherited.",
]
_IN_WORKER = False
_VERIFY_STACK = set()


def write_json(path, value):
    """Create one finite JSON file without overwriting existing evidence."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _read(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe artifact path")
    path = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Artifact symlinks are unsupported")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Missing or unsafe artifact")
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("Artifact exceeds bounded reader size")
    return path


def _json(root, name):
    return json.loads(_read(root, name).read_text(encoding="utf-8"))


def _rows(plan):
    mode = plan["request"]["mode"]
    step, anchor, times = plan["frequency_ns"], plan["origin_ns"], plan["times_ns"]
    number = 0
    for group in plan["groups"]:
        members = {}
        for row in group["retained"]:
            point = times[row] if mode != "aggregate" else anchor + (times[row] - anchor) // step * step
            members.setdefault(point, []).append(row)
        points = (
            [times[i] for i in group["retained"]]
            if mode == "normalize"
            else range(group["first"], group["last"] + 1, step)
        )
        for point in points:
            label = point + step if mode == "aggregate" else point
            yield number, group, point, label, members.get(point, [])
            number += 1


def _produce(source_path, plan, root):
    validate_temporal_plan(plan, source_path)
    shutil.copyfile(source_path, root / "source.csv")
    validate_temporal_plan(plan, root / "source.csv")
    write_json(root / "plan.json", plan)
    source, roles, _, _ = _source(root / "source.csv", plan["roles"], plan["request"]["timezone"])
    features, mode = roles["feature_cols"], plan["request"]["mode"]
    summary = dict(
        output_missing=0,
        empty_buckets=0,
        all_missing_cells=0,
        original_missing_cells=0,
        last_missing_cells=0,
        grid_added_cells=0,
    )
    with (
        (root / "data.csv").open("x", encoding="utf-8", newline="") as output,
        (root / "lineage.jsonl").open("x", encoding="utf-8") as lineage,
    ):
        writer = csv.DictWriter(output, fieldnames=plan["columns"])
        writer.writeheader()
        for number, group, start, label, rows in _rows(plan):
            record = {column: "" for column in plan["columns"]}
            if rows and mode != "aggregate":
                record.update({column: source.at[rows[0], column] for column in plan["metadata"]})
            else:
                record.update(group["constants"])
            if roles["sample_id_col"]:
                record[roles["sample_id_col"]] = group["entity"]
            record[roles["timestamp_col"]] = iso(label)
            item = dict(
                row=number,
                entity=group["entity"],
                time=iso(label),
                source_rows=rows,
                cells={},
                split=_role(start, group["boundary"]) if group["boundary"] else None,
                metadata_unknown=[c for c in plan["metadata"] if not rows and c not in group["constants"]],
            )
            if mode == "aggregate":
                item.update(
                    interval_start=iso(start),
                    interval_end=iso(label),
                    extends_before_observations=start < group["original_first"],
                    extends_after_observations=label > group["original_last"],
                )
                summary["empty_buckets"] += not bool(rows)
            for feature in features:
                finite = [row for row in rows if np.isfinite(source.at[row, feature])]
                method = plan["request"]["reducers"][feature] if mode == "aggregate" else "copy"
                selected = rows[-1] if rows and method == "last" else rows[0] if rows and method == "copy" else None
                if selected is not None:
                    value = float(source.at[selected, feature])
                    contributors = [selected] if np.isfinite(value) else []
                elif finite:
                    values = source.loc[finite, feature].to_numpy()
                    total = math.fsum(float(value) for value in values)
                    value = total / len(values) if method == "mean" else total
                    contributors = finite
                else:
                    value, contributors = np.nan, []
                if np.isinf(value):
                    raise ValueError("Aggregation overflowed finite numerical output")
                reason = None
                if np.isnan(value):
                    reason = (
                        "grid-added"
                        if mode == "regularize" and not rows
                        else (
                            "empty-bucket"
                            if not rows
                            else (
                                "original-missing"
                                if mode != "aggregate"
                                else "last-missing" if method == "last" and finite else "all-missing"
                            )
                        )
                    )
                    summary["output_missing"] += 1
                    counter = {
                        "grid-added": "grid_added_cells",
                        "original-missing": "original_missing_cells",
                        "all-missing": "all_missing_cells",
                        "last-missing": "last_missing_cells",
                    }.get(reason)
                    if counter:
                        summary[counter] += 1
                record[feature] = "" if np.isnan(value) else format(value, ".17g")
                item["cells"][feature] = dict(
                    method=method,
                    record_count=len(rows),
                    finite_count=len(finite),
                    contributors=contributors,
                    selected_row=selected,
                    unknown_reason=reason,
                )
            writer.writerow(record)
            lineage.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    derived_sha = file_sha256(root / "data.csv")
    split_record = None
    if plan["split"]:
        split_record = dict(
            schema="timeseries-temporal-split-v1",
            original_sha256=plan["source"]["sha256"],
            derived_sha256=derived_sha,
            plan_sha256=plan["sha256"],
            original_plan_sha256=plan["request"]["split_plan"]["sha256"],
            split=plan["split"],
        )
        write_json(root / "split.json", split_record)
    semantics = prepare_feature_semantics(None, features, derived_sha, unit_state=plan["feature_semantics"])
    if semantics:
        write_json(root / "feature-semantics.json", semantics)
    report = [
        "# Temporal preprocessing",
        "",
        f"Mode: {mode}",
        f"Source rows: {plan['source']['rows']}",
        f"Output rows: {plan['preview']['output_rows']}",
        f"Preview: {json.dumps(plan['preview'])}",
        f"Missingness: {json.dumps(summary)}",
        f"Timezone: {plan['request']['timezone'] or 'source offsets'}",
        f"Frequency: {plan['request']['frequency']}; origin: {iso(plan['origin_ns'])}",
        f"Reducers: {json.dumps(plan['request']['reducers'])}",
        "",
        *LIMITATIONS,
    ]
    with (root / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(report) + "\n")
    write_json(
        root / "execution.json",
        dict(
            status="success",
            temporal_runs=1,
            model_fit=0,
            model_predict=0,
            statistical_estimation=0,
            replay=0,
            provider=0,
        ),
    )
    files = {p.name: file_sha256(p) for p in root.iterdir() if p.is_file()}
    result = dict(
        schema=RESULT_SCHEMA,
        status="success",
        source_sha256=plan["source"]["sha256"],
        derived_sha256=derived_sha,
        plan_sha256=plan["sha256"],
        files=files,
        summary=summary,
        roles=roles,
        split_status="inherited" if split_record else "unsplit",
        limitations=LIMITATIONS,
    )
    if plan.get("reshape_result"):
        result["reshape_result"] = plan["reshape_result"]
    if plan.get("merge_result"):
        result["merge_result"] = plan["merge_result"]
    write_json(root / "pending-result.json", result)
    return dict(
        status="success",
        result=str(root),
        data=str(root / "data.csv"),
        report=str(root / "report.md"),
        summary=summary,
        split_status=result["split_status"],
    )


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _verify_data(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Invalid temporal result directory")
    saved = _json(root, "result.json")
    _require(saved.get("schema") == RESULT_SCHEMA and saved.get("status") == "success", "Incomplete/unsupported result")
    required = {"source.csv", "plan.json", "data.csv", "lineage.jsonl", "report.md", "execution.json"}
    _require(required.issubset(saved["files"]), "Incomplete artifact inventory")
    for name, digest in saved["files"].items():
        _require(file_sha256(_read(root, name)) == digest, "Artifact hash mismatch: " + name)
    plan = _json(root, "plan.json")
    validate_temporal_plan(plan, _read(root, "source.csv"))
    _require(saved.get("reshape_result") == plan.get("reshape_result"), "Reshape parent reference differs")
    _require(saved.get("merge_result") == plan.get("merge_result"), "Merge parent reference differs")
    source, roles, _, times = _source(root / "source.csv", plan["roles"], plan["request"]["timezone"])
    data = pd.read_csv(_read(root, "data.csv"), dtype=str, keep_default_na=False)
    _require(
        list(data.columns) == plan["columns"] and len(data) == plan["preview"]["output_rows"], "Derived shape differs"
    )
    _require(file_sha256(root / "data.csv") == saved["derived_sha256"], "Derived source identity differs")
    _require(
        saved["source_sha256"] == plan["source"]["sha256"]
        and saved["plan_sha256"] == plan["sha256"]
        and saved["roles"] == roles,
        "Result identity/roles differ",
    )
    mode, step, origin = plan["request"]["mode"], plan["frequency_ns"], plan["origin_ns"]
    group_by_entity = {g["entity"]: g for g in plan["groups"]}
    expected_members = {}
    for group in plan["groups"]:
        for row in group["retained"]:
            tick = times[row] if mode != "aggregate" else origin + (times[row] - origin) // step * step
            expected_members.setdefault((group["entity"], tick), []).append(row)
    summary = dict(
        output_missing=0,
        empty_buckets=0,
        all_missing_cells=0,
        original_missing_cells=0,
        last_missing_cells=0,
        grid_added_cells=0,
    )
    seen = []
    inherited = {name: [] for name in ("train", "val", "test")}
    with _read(root, "lineage.jsonl").open(encoding="utf-8") as stream:
        for number, line in enumerate(stream):
            _require(number < len(data), "Too many lineage rows")
            item = json.loads(line)
            entity = item["entity"]
            _require(entity in group_by_entity, "Unknown lineage entity")
            group = group_by_entity[entity]
            label = pd.Timestamp(data.at[number, roles["timestamp_col"]])
            _require(label.tzinfo is not None, "Derived timestamps need UTC")
            label = int(label.as_unit("ns").value)
            tick = label - step if mode == "aggregate" else label
            _require(item["row"] == number and item["time"] == iso(label), "Lineage position/time differs")
            rows = expected_members.get((entity, tick), [])
            _require(item["source_rows"] == rows, "Source membership differs")
            if roles["sample_id_col"]:
                _require(data.at[number, roles["sample_id_col"]] == entity, "Entity string differs")
            seen.append((entity, tick))
            if mode == "aggregate":
                _require(
                    tick >= group["first"] and tick <= group["last"] and (tick - origin) % step == 0,
                    "Bucket is outside its declared grid",
                )
                _require(item["interval_start"] == iso(tick) and item["interval_end"] == iso(label), "Interval differs")
                _require(
                    item["extends_before_observations"] == (tick < group["original_first"])
                    and item["extends_after_observations"] == (label > group["original_last"]),
                    "Edge scope differs",
                )
                summary["empty_buckets"] += not bool(rows)
            elif mode == "regularize":
                _require(group["first"] <= tick <= group["last"] and (tick - origin) % step == 0, "Grid differs")
            else:
                _require(bool(rows) and tick == times[rows[0]], "Normalized row has no original observation")
            unknown = [c for c in plan["metadata"] if not rows and c not in group["constants"]]
            _require(item["metadata_unknown"] == unknown, "Metadata unknown flags differ")
            for column in plan["metadata"]:
                value = (
                    source.at[rows[0], column] if rows and mode != "aggregate" else group["constants"].get(column, "")
                )
                _require(data.at[number, column] == value, "Metadata differs")
            _require(set(item["cells"]) == set(roles["feature_cols"]), "Feature support schema differs")
            for feature in roles["feature_cols"]:
                finite = [row for row in rows if math.isfinite(float(source.at[row, feature]))]
                method = plan["request"]["reducers"][feature] if mode == "aggregate" else "copy"
                chosen = rows[-1] if rows and method == "last" else rows[0] if rows and method == "copy" else None
                if chosen is not None:
                    expected = float(source.at[chosen, feature])
                    contributors = [chosen] if math.isfinite(expected) else []
                else:
                    values = [float(source.at[row, feature]) for row in finite]
                    expected = (
                        (math.fsum(values) / len(values) if method == "mean" else math.fsum(values))
                        if values
                        else math.nan
                    )
                    contributors = finite
                actual = float(data.at[number, feature]) if data.at[number, feature] else math.nan
                if method in ("copy", "last"):
                    _require(actual == expected or math.isnan(actual) and math.isnan(expected), "Copied value differs")
                else:
                    _require(
                        math.isnan(actual)
                        and math.isnan(expected)
                        or math.isfinite(actual)
                        and math.isfinite(expected)
                        and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                        "Aggregate value differs",
                    )
                reason = None
                if math.isnan(expected):
                    reason = (
                        "grid-added"
                        if mode == "regularize" and not rows
                        else (
                            "empty-bucket"
                            if not rows
                            else (
                                "original-missing"
                                if mode != "aggregate"
                                else "last-missing" if method == "last" and finite else "all-missing"
                            )
                        )
                    )
                    summary["output_missing"] += 1
                    key = {
                        "grid-added": "grid_added_cells",
                        "original-missing": "original_missing_cells",
                        "all-missing": "all_missing_cells",
                        "last-missing": "last_missing_cells",
                    }.get(reason)
                    if key:
                        summary[key] += 1
                expected_cell = dict(
                    method=method,
                    record_count=len(rows),
                    finite_count=len(finite),
                    contributors=contributors,
                    selected_row=chosen,
                    unknown_reason=reason,
                )
                _require(item["cells"][feature] == expected_cell, "Support/denominator/contributor differs")
            role = _role(tick, group["boundary"]) if group["boundary"] else None
            _require(item["split"] == role, "Inherited role differs")
            if role:
                inherited[role].append(number)
    expected_order = []
    for group in plan["groups"]:
        points = (
            [times[i] for i in group["retained"]]
            if mode == "normalize"
            else range(group["first"], group["last"] + 1, step)
        )
        expected_order.extend((group["entity"], point) for point in points)
    _require(seen == expected_order and summary == saved["summary"], "Output coverage/summary differs")
    if plan["split"]:
        split = _json(root, "split.json")
        expected_split = dict(
            schema="timeseries-temporal-split-v1",
            original_sha256=plan["source"]["sha256"],
            derived_sha256=saved["derived_sha256"],
            plan_sha256=plan["sha256"],
            original_plan_sha256=plan["request"]["split_plan"]["sha256"],
            split=plan["split"],
        )
        _require(split == expected_split and inherited == split["split"]["rows"], "Derived split differs")
    else:
        _require(not (root / "split.json").exists(), "Unexpected split on unsplit data")
    semantics = prepare_feature_semantics(
        None, roles["feature_cols"], saved["derived_sha256"], unit_state=plan["feature_semantics"]
    )
    if semantics:
        _require(_json(root, "feature-semantics.json") == semantics, "Derived unit lineage differs")
    _require(saved["split_status"] == ("inherited" if plan["split"] else "unsplit"), "Split status differs")
    return dict(
        status="success",
        result_status=saved["status"],
        result=str(root),
        data=str(root / "data.csv"),
        report=str(root / "report.md"),
        verification="source_semantics_and_direct_arithmetic",
        summary=summary,
        roles=roles,
        split=plan["split"],
        feature_semantics=semantics,
        derived_sha256=saved["derived_sha256"],
        plan_sha256=plan["sha256"],
        split_status=saved["split_status"],
        limitations=LIMITATIONS,
    )


def _verify(root):
    key = str(root.resolve())
    if key in _VERIFY_STACK:
        raise ValueError("Cyclic temporal source references")
    _VERIFY_STACK.add(key)
    try:
        return _verify_data(root)
    finally:
        _VERIFY_STACK.remove(key)


def _work(request):
    global _IN_WORKER
    previous = _IN_WORKER
    _IN_WORKER = True
    try:
        if request["operation"] == "run":
            return _produce(Path(request["source"]), request["plan"], Path(request["output"]))
        return _verify(Path(request["source"]))
    except Exception as error:
        return dict(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        _IN_WORKER = previous


def supervise(operation, source, plan, output, timeout, *, kind="temporal"):
    """Run one owned worker; read-only verification creates no result files."""
    maximum = 60 if operation == "run" else 30
    if kind not in ("temporal", "reshape", "merge"):
        raise ValueError("Unsupported preparation worker")
    if not math.isfinite(timeout) or not 0 < timeout <= maximum:
        raise ValueError(f"Budget must be in (0,{maximum}] seconds")
    root = None
    if operation == "run":
        if kind == "temporal" and not plan.get("reshape_result") and not plan.get("merge_result"):
            validate_temporal_plan(plan, source)
        elif plan.get("schema") != f"timeseries-{kind}-plan-v1" or plan.get("status") != "ready":
            raise ValueError("A ready supported preparation plan is required")
        root = Path(output).absolute()
        if root.exists() or root.is_symlink():
            raise FileExistsError("Output exists; choose a new directory")
        root.mkdir(parents=True, exist_ok=False)
    request = dict(
        operation=operation, source=str(Path(source).absolute()), plan=plan, output=str(root) if root else None
    )
    code = (
        "import contextlib,json,sys\nwith contextlib.redirect_stdout(sys.stderr):\n"
        f" from ai4ts.data_protocols.{kind}_artifacts import _work\n"
        " result=_work(json.load(sys.stdin))\nprint(json.dumps(result,ensure_ascii=False,allow_nan=False))\n"
    )
    start = time.monotonic()
    stdout = stderr = ""
    try:
        process = subprocess.run(
            [sys.executable, "-c", code],
            input=json.dumps(request, allow_nan=False),
            text=True,
            capture_output=True,
            timeout=timeout,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )
        stdout, stderr = process.stdout, process.stderr
        result = json.loads(stdout)
        if process.returncode:
            raise ValueError(f"{kind.capitalize()} worker failed")
    except subprocess.TimeoutExpired as error:
        stdout = (
            (error.stdout or b"").decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or ""
        )
        stderr = (
            (error.stderr or b"").decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or ""
        )
        result = dict(
            status="failed", error=f"{kind.capitalize()} worker timed out; no automatic retry", timed_out=True
        )
    except Exception as error:
        result = dict(status="failed", error=f"{type(error).__name__}: {error}")
    result["worker_seconds"] = time.monotonic() - start
    if root:
        try:
            with (root / "worker.stdout.log").open("x", encoding="utf-8") as stream:
                stream.write(stdout)
            with (root / "worker.stderr.log").open("x", encoding="utf-8") as stream:
                stream.write(stderr)
            write_json(
                root / "supervisor.json",
                dict(
                    status=result["status"],
                    wall_seconds=result["worker_seconds"],
                    timeout_seconds=timeout,
                    timed_out=result.get("timed_out", False),
                ),
            )
            if result["status"] == "success":
                completed = _json(root, "pending-result.json")
                inventory = root.rglob("*") if kind == "merge" else root.iterdir()
                completed["files"] = {
                    str(p.relative_to(root)): file_sha256(_read(root, str(p.relative_to(root))))
                    for p in inventory
                    if p.is_file() and p.name != "pending-result.json"
                }
                write_json(root / "result.json", completed)
                (root / "pending-result.json").unlink()
        except Exception as error:
            result.update(status="failed", error=f"Result publication failed: {type(error).__name__}: {error}")
        if result["status"] != "success":
            if (root / "result.json").exists():
                (root / "result.json").rename(root / "incomplete-result.json")
            try:
                write_json(root / "failure.json", result)
            except OSError:
                pass
            result["partial_evidence"] = str(root)
    return result
