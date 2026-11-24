"""JSON-first general analysis commands sharing the Python implementation."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import click
import json
from pathlib import Path

from .json_output import JsonCommandGroup, emit_json_result


@click.group(cls=JsonCommandGroup, help="Source-bound diagnostics, fixed-origin baselines, and saved evidence views")
def analyze():
    """Analyze time series without changing the existing model workflows."""


@analyze.command("capabilities")
@click.option("--json", "json_output", is_flag=True, help="Return machine-readable method metadata (the default)")
def capabilities(json_output):
    from ai4ts.analysis import available_methods

    emit_json_result(available_methods)


@analyze.command("plan")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path())
@click.option("--kind", type=click.Choice(["diagnostics", "forecast-baselines"]), default="diagnostics")
@click.option("--sample-id")
@click.option("--timestamp")
@click.option("--features", help="Comma-separated confirmed measurement columns")
@click.option("--methods", help="Comma-separated diagnostic methods; default is profile only")
@click.option("--period", type=int)
@click.option("--period-range", help="Two physical durations, for example 30min,120min")
@click.option("--expected-frequency", help="Declared fixed schedule within the observed span")
@click.option("--time-basis", type=click.Choice(["observation_index"]))
@click.option("--scope", type=click.Choice(["descriptive_all", "train", "val", "test"]), default="descriptive_all")
@click.option("--scope-plan", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--confirmed-plan", type=click.Path(exists=True, dir_okay=False), help="Reuse source-matching confirmed fields"
)
@click.option("--rows", help="Explicit original row range START:STOP, stop exclusive")
@click.option("--targets")
@click.option("--split-policy", type=click.Choice(["chronological", "entity_chronological"]))
@click.option("--seasonal-period", type=int)
@click.option("--candidates", help="last-value,seasonal-naive; a single declared reference is allowed")
@click.option("--units", type=click.Path(exists=True, dir_okay=False), help="Optional explicitly confirmed unit JSON")
@click.option("--temporal-result", type=click.Path(exists=True, file_okay=False))
@click.option("--reshape-result", type=click.Path(exists=True, file_okay=False))
@click.option("--merge-result", type=click.Path(exists=True, file_okay=False))
def plan(**args):
    def action():
        import pandas as pd
        from ai4ts.data_protocols.ingestion import content_sha256
        from ai4ts.data_protocols.registry import file_sha256
        from ai4ts.analysis.artifacts import create_plan, read_json, write_json

        prior = read_json(args["confirmed_plan"]) if args["confirmed_plan"] else None
        if prior:
            declared = dict(prior)
            digest = declared.pop("sha256", None)
            if content_sha256(declared) != digest or prior.get("source", {}).get("sha256") != file_sha256(
                args["input_path"]
            ):
                raise ValueError("Confirmed fields do not match this source/plan")
        previous = (prior or {}).get("request", {})
        roles = (prior or {}).get("roles")
        if args["features"]:
            roles = dict(roles or {}, feature_cols=args["features"].split(","))
            for flag, key in (("sample_id", "sample_id_col"), ("timestamp", "timestamp_col")):
                if args[flag] is not None or key not in roles:
                    roles[key] = args[flag]
            roles.setdefault("label_col", None)
        elif roles:
            roles = dict(roles)
            for flag, key in (("sample_id", "sample_id_col"), ("timestamp", "timestamp_col")):
                if args[flag] is not None:
                    roles[key] = args[flag]
        if args["reshape_result"] is not None or args["merge_result"] is not None:
            roles = dict((prior or {}).get("roles") or {})
            if args["features"] is not None:
                roles["feature_cols"] = args["features"].split(",")
            for flag, key in (("sample_id", "sample_id_col"), ("timestamp", "timestamp_col")):
                if args[flag] is not None:
                    roles[key] = args[flag]
            roles = roles or None
        selected = None
        if args["rows"]:
            start, stop = map(int, args["rows"].split(":"))
            if not 0 <= start < stop or stop - start > 32768:
                raise ValueError("Explicit row range exceeds its bounded limit")
            selected = list(range(start, stop))
        period_range = args["period_range"]
        if period_range:
            period_range = [float(pd.Timedelta(p).total_seconds()) for p in period_range.split(",")]
        units = read_json(args["units"]) if args["units"] else previous.get("feature_units")
        if units is None and prior and prior.get("feature_semantics"):
            state = prior["feature_semantics"]
            units = {"units": state["units"], "origin": "user-confirmed", "source_sha256": state["source_sha256"]}
        request = dict(
            kind=args["kind"],
            roles=roles,
            methods=args["methods"].split(",") if args["methods"] else None,
            period=args["period"] if args["period"] is not None else previous.get("period"),
            period_range=period_range if period_range is not None else previous.get("period_range"),
            expected_frequency=args["expected_frequency"] or previous.get("expected_frequency"),
            time_basis=args["time_basis"] or previous.get("time_basis"),
            scope=args["scope"],
            scope_plan=read_json(args["scope_plan"]) if args["scope_plan"] else None,
            rows=selected,
            targets=args["targets"].split(",") if args["targets"] else None,
            split_policy=args["split_policy"],
            seasonal_period=(
                args["seasonal_period"] if args["seasonal_period"] is not None else previous.get("seasonal_period")
            ),
            candidates=args["candidates"].split(",") if args["candidates"] else None,
            feature_units=units,
            confirmation={"roles": "source-matching confirmed plan" if prior else "explicit declaration"},
        )
        if args["temporal_result"] is not None:
            request["temporal_result"] = args["temporal_result"]
        if args["reshape_result"] is not None:
            request["reshape_result"] = args["reshape_result"]
        if args["merge_result"] is not None:
            request["merge_result"] = args["merge_result"]
        result = create_plan(args["input_path"], **request)
        output = Path(args["output"]).absolute()
        if output.exists() or output.is_symlink():
            raise FileExistsError("Choose a new plan directory")
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", result)
        text = [
            "# Analysis preview",
            "",
            "Status: " + result["status"],
            "",
            "No statistical/forecasting backend was called.",
            "Ready means executable, not scientifically validated or production supported.",
            "",
            "```json",
            json.dumps(
                {
                    k: result.get(k)
                    for k in ("roles", "scope", "groups", "applicability", "budget", "unresolved", "errors")
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
        (output / "preview.md").write_text("\n".join(text), encoding="utf-8")
        return dict(
            status=result["status"],
            plan=str(output / "plan.json"),
            preview=str(output / "preview.md"),
            unresolved=result["unresolved"],
            errors=result["errors"],
            budget=result.get("budget"),
        )

    emit_json_result(action)


@analyze.command("run")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--plan", "plan_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path())
def run(input_path, plan_path, output):
    def action():
        from ai4ts.analysis import compare_baselines, diagnose
        from ai4ts.analysis.artifacts import read_json

        value = read_json(plan_path)
        operation = diagnose if value["request"]["kind"] == "diagnostics" else compare_baselines
        return operation(input_path, value, output)

    emit_json_result(action)


@analyze.command("inspect")
@click.option("--analysis", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--view", type=click.Choice(["summary", "feature", "entity", "method"]), default="summary")
@click.option("--feature")
@click.option("--entity")
@click.option("--method")
@click.option("--output", type=click.Path())
def inspect(analysis, view, feature, entity, method, output):
    def action():
        from ai4ts.analysis import inspect_analysis

        return inspect_analysis(analysis, view, feature, entity, method, output)

    emit_json_result(action)
