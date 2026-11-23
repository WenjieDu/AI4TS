"""Thin JSON entry for AI4TS long-to-wide measurement preparation."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import click

from .json_output import JsonCommandGroup, emit_json_result


class ReshapeGroup(JsonCommandGroup):
    """Include errors parsing the new group itself in its JSON contract."""

    def parse_args(self, ctx, args):
        try:
            return super().parse_args(ctx, args)
        except click.ClickException as error:
            emit_json_result(lambda: dict(status="failed", error=error.format_message()))


@click.group(name="reshape", cls=ReshapeGroup, help="Explicit long-to-wide CSV preparation; no models")
def reshape():
    """Plan, execute or inspect a source-bound conversion."""


@reshape.command("plan")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path())
@click.option("--sample-id")
@click.option("--timestamp")
@click.option("--variable", help="Confirmed column containing measurement names")
@click.option("--value", help="Confirmed numeric reading column")
@click.option("--unit-column")
@click.option("--label")
@click.option(
    "--feature", "features", multiple=True, help="Selected variable, repeated in output order; commas are literal"
)
@click.option("--timezone", help="Explicit interpretation of naive clocks")
@click.option("--duplicate-policy", type=click.Choice(["error", "keep-first", "keep-last"]), help="Defaults to error")
@click.option("--drop-column", "drops", multiple=True, help="Explicit non-role metadata projection")
@click.option("--units", type=click.Path(exists=True, dir_okay=False))
@click.option("--confirmed-plan", type=click.Path(exists=True, dir_okay=False))
def plan(**args):
    def action():
        from ai4ts.data_protocols import create_reshape_plan
        from ai4ts.data_protocols.temporal_preprocessing import declaration
        from ai4ts.data_protocols.temporal_artifacts import write_json

        roles = {}
        for flag, key in (
            ("sample_id", "sample_id_col"),
            ("timestamp", "timestamp_col"),
            ("variable", "variable_col"),
            ("value", "value_col"),
            ("unit_column", "unit_col"),
            ("label", "label_col"),
        ):
            if args[flag] is not None:
                roles[key] = args[flag]
        prior = declaration(args["confirmed_plan"]) if args["confirmed_plan"] else None
        if prior and roles:
            roles = dict(prior["request"]["roles"], **roles)
        value = create_reshape_plan(
            args["input_path"],
            roles=roles or None,
            variables=list(args["features"]) or None,
            timezone=args["timezone"],
            duplicate_policy=args["duplicate_policy"],
            drop_columns=list(args["drops"]) or None,
            feature_units=declaration(args["units"]) if args["units"] else None,
            confirmed_plan=prior,
        )
        output = Path(args["output"]).absolute()
        if output.exists() or output.is_symlink():
            raise FileExistsError("Choose a new reshape plan directory")
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", value)
        preview = {
            k: value.get(k)
            for k in (
                "status",
                "request",
                "wide_roles",
                "preview",
                "candidates",
                "limits",
                "budget",
                "unresolved",
                "errors",
            )
        }
        with (output / "preview.md").open("x", encoding="utf-8") as stream:
            stream.write(
                "# Long-to-wide preview\n\nNo wide CSV, tensor or fitted state was produced.\n\n```json\n"
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


@reshape.command("run")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--plan", "plan_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path())
@click.option("--seconds", default=60.0, type=float, show_default=True)
def run(input_path, plan_path, output, seconds):
    def action():
        from ai4ts.data_protocols import run_reshape
        from ai4ts.data_protocols.temporal_preprocessing import declaration

        return run_reshape(input_path, declaration(plan_path), output, timeout=seconds)

    emit_json_result(action)


@reshape.command("inspect")
@click.option("--result", "result_path", required=True, type=click.Path())
@click.option("--seconds", default=30.0, type=float, show_default=True)
def inspect(result_path, seconds):
    def action():
        from ai4ts.data_protocols import inspect_reshape_result

        return inspect_reshape_result(result_path, timeout=seconds)

    emit_json_result(action)
