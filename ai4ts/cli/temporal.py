"""Thin JSON CLI for AI4TS temporal preprocessing."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import click

from .json_output import JsonCommandGroup, emit_json_result


@click.group(name="temporal", cls=JsonCommandGroup, help="Explicit physical-time preparation; no imputation or models")
def temporal():
    """Plan, execute or inspect source-bound temporal preparation."""


@temporal.command("plan")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path())
@click.option("--mode", required=True, type=click.Choice(["normalize", "regularize", "aggregate"]))
@click.option("--sample-id")
@click.option("--timestamp")
@click.option("--features", help="Confirmed ordered measurement columns, comma separated")
@click.option("--label")
@click.option("--frequency", help="Positive integer seconds/minutes/hours, e.g. 5min")
@click.option("--timezone", help="Explicit interpretation of naive timestamps")
@click.option("--origin", default="epoch", show_default=True)
@click.option("--duplicate-policy", default="error", type=click.Choice(["error", "keep-first", "keep-last"]))
@click.option("--reducers", help="Explicit feature=mean|sum|last pairs, comma separated")
@click.option("--drop-columns", help="Explicitly exclude non-role metadata columns")
@click.option("--units", type=click.Path(exists=True, dir_okay=False))
@click.option("--confirmed-plan", type=click.Path(exists=True, dir_okay=False))
@click.option("--split-plan", type=click.Path(exists=True, dir_okay=False))
@click.option("--reshape-result", type=click.Path(exists=True, file_okay=False))
@click.option("--merge-result", type=click.Path(exists=True, file_okay=False))
def plan(**args):
    def action():
        from ai4ts.data_protocols.temporal_preprocessing import create_temporal_plan, declaration
        from ai4ts.data_protocols.temporal_artifacts import write_json

        prior = declaration(args["confirmed_plan"]) if args["confirmed_plan"] else None
        roles = dict(prior["roles"]) if prior and prior.get("roles") else None
        if args["features"] is not None:
            roles = dict(roles or {}, feature_cols=args["features"].split(","))
        for flag, key in (("sample_id", "sample_id_col"), ("timestamp", "timestamp_col"), ("label", "label_col")):
            if args[flag] is not None:
                roles = dict(roles or {}, **{key: args[flag]})
        reducers = None
        if args["reducers"]:
            pairs = [pair.split("=") for pair in args["reducers"].split(",")]
            if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
                raise ValueError("reducers must be unique feature=method pairs")
            reducers = dict(pairs)
        value = create_temporal_plan(
            args["input_path"],
            mode=args["mode"],
            roles=roles,
            frequency=args["frequency"],
            timezone=args["timezone"],
            origin=args["origin"],
            duplicate_policy=args["duplicate_policy"],
            reducers=reducers,
            drop_columns=args["drop_columns"].split(",") if args["drop_columns"] else None,
            feature_units=declaration(args["units"]) if args["units"] else None,
            confirmed_plan=prior,
            split_plan=args["split_plan"],
            **({"reshape_result": args["reshape_result"]} if args["reshape_result"] is not None else {}),
            **({"merge_result": args["merge_result"]} if args["merge_result"] is not None else {}),
        )
        output = Path(args["output"]).absolute()
        if output.exists() or output.is_symlink():
            raise FileExistsError("Choose a new temporal plan directory")
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", value)
        preview = {
            key: value.get(key)
            for key in ("status", "request", "roles", "preview", "limits", "budget", "unresolved", "errors")
        }
        with (output / "preview.md").open("x", encoding="utf-8") as stream:
            stream.write(
                "# Temporal preparation preview\n\nNo numerical output, imputation or model was produced.\n\n```json\n"
                + json.dumps(preview, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n```\n"
            )
        return dict(
            status=value["status"],
            plan=str(output / "plan.json"),
            preview=str(output / "preview.md"),
            unresolved=value["unresolved"],
            errors=value["errors"],
            impact=value.get("preview"),
        )

    emit_json_result(action)


@temporal.command("run")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--plan", "plan_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path())
@click.option("--seconds", default=60.0, type=float, show_default=True)
def run(input_path, plan_path, output, seconds):
    def action():
        from ai4ts.data_protocols.temporal_preprocessing import declaration, run_temporal_preprocessing

        return run_temporal_preprocessing(input_path, declaration(plan_path), output, timeout=seconds)

    emit_json_result(action)


@temporal.command("inspect")
@click.option("--result", "result_path", required=True, type=click.Path())
@click.option("--seconds", default=30.0, type=float, show_default=True)
def inspect(result_path, seconds):
    def action():
        from ai4ts.data_protocols.temporal_preprocessing import inspect_temporal_result

        return inspect_temporal_result(result_path, timeout=seconds)

    emit_json_result(action)
