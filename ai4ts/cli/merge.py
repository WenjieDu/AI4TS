"""Thin entries for shared AI4TS exact-time multi-source contracts."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import click

from .json_output import emit_json_result
from .reshape import ReshapeGroup


@click.group(name="merge", cls=ReshapeGroup, help="Explicit exact-time joins of two to four wide sources; no models")
def merge():
    """Plan, execute or independently inspect an exact merge."""


@merge.command("plan")
@click.option(
    "--sources",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Bounded JSON object with ordered "sources" declarations',
)
@click.option("--join", type=click.Choice(["left", "outer"]))
@click.option("--primary")
@click.option("--entity-meaning", help="Confirmed common entity identifier meaning")
@click.option("--single-series", is_flag=True, help="Confirm that all sources without entity roles describe one series")
@click.option("--preserve-split", is_flag=True, help="Unsupported intent, rejected before production")
@click.option("--output", required=True, type=click.Path())
def plan(sources, join, primary, entity_meaning, single_series, preserve_split, output):
    def action():
        from ai4ts.data_protocols.merge import create_merge_plan, safe_path
        from ai4ts.data_protocols.temporal_preprocessing import declaration
        from ai4ts.data_protocols.temporal_artifacts import write_json

        declared = declaration(sources)
        if set(declared) != {"sources"}:
            raise ValueError('Sources JSON must contain only "sources"')
        value = create_merge_plan(declared["sources"], join, primary, entity_meaning, single_series, preserve_split)
        folder = safe_path(output)
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / "plan.json", value)
        with (folder / "preview.md").open("x", encoding="utf-8") as stream:
            stream.write(
                "# Exact merge preview\n\nNo merged CSV, tensors or fitted state produced.\n\n"
                + json.dumps(value.get("preview"), ensure_ascii=False, indent=2)
                + "\n"
            )
        return dict(
            status=value["status"],
            plan=str(folder / "plan.json"),
            preview=str(folder / "preview.md"),
            unresolved=value["unresolved"],
            errors=value["errors"],
            impact=value.get("preview"),
        )

    emit_json_result(action)


@merge.command("run")
@click.option("--plan", "plan_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path())
@click.option("--seconds", default=60.0, type=float, show_default=True)
def run(plan_path, output, seconds):
    def action():
        from ai4ts.data_protocols.merge import run_merge
        from ai4ts.data_protocols.temporal_preprocessing import declaration

        return run_merge(declaration(plan_path), output, timeout=seconds)

    emit_json_result(action)


@merge.command("inspect")
@click.option("--result", "result_path", required=True, type=click.Path())
@click.option("--seconds", default=30.0, type=float, show_default=True)
def inspect(result_path, seconds):
    def action():
        from ai4ts.data_protocols.merge import inspect_merge_result

        return inspect_merge_result(result_path, timeout=seconds)

    emit_json_result(action)
