"""Write source-bound ingestion plans without preparing model data."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import click

from .json_output import emit_json_result
from .time_basis import reducers_from_option, selection_from_options


class TimePlanCommand(click.Command):
    """Use JSON parsing failures only for the new opt-in time options."""

    def parse_args(self, ctx, args):
        flags = {
            "--time-basis",
            "--time-selection",
            "--time-confirmation",
            "--timezone",
            "--frequency",
            "--origin",
            "--grid-mode",
            "--reducers",
            "--drop-columns",
            "--merge-result",
        }
        opted_in = any(item.split("=", 1)[0] in flags for item in args)
        try:
            return super().parse_args(ctx, args)
        except click.UsageError as error:
            if not opted_in:
                raise
            message = error.format_message()
            emit_json_result(lambda: dict(status="failed", error=message))


@click.command(
    name="plan",
    cls=TimePlanCommand,
    help="Plan task data roles, splits, and windows without producing tensors",
)
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option(
    "--task",
    required=True,
    type=click.Choice(
        ["imputation", "classification", "forecasting", "clustering", "anomaly_detection", "representation"]
    ),
)
@click.option("--sample_id", default=None)
@click.option("--timestamp", default=None)
@click.option("--features", default=None, help="Confirmed comma-separated ordered numeric features")
@click.option("--label", default=None)
@click.option("--split_policy", type=click.Choice(["entity", "chronological", "row_order", "entity_chronological"]))
@click.option("--window_size", default=None, help="Positive observation count or exact duration, e.g. 24h")
@click.option("--n_pred_steps", default=None)
@click.option("--target_features", default=None)
@click.option("--n_clusters", type=int, default=None)
@click.option("--clustering_model", type=click.Choice(["VaDER", "CRLI"]), default="VaDER")
@click.option("--anomaly_rate", type=float, default=None)
@click.option("--allow_irregular", is_flag=True)
@click.option("--temporal-result", type=click.Path(exists=True, file_okay=False))
@click.option("--reshape-result", type=click.Path(exists=True, file_okay=False))
@click.option("--merge-result", type=click.Path(exists=True, file_okay=False))
@click.option("--time-basis", type=click.Choice(["observation_index", "regular_grid"]))
@click.option("--time-selection", type=click.Path(exists=True, dir_okay=False))
@click.option("--time-confirmation", help="Brief confirmed choice and understood consequences")
@click.option("--timezone")
@click.option("--frequency")
@click.option("--origin", default="epoch")
@click.option("--grid-mode", type=click.Choice(["regularize", "aggregate"]), default="regularize")
@click.option("--reducers")
@click.option("--drop-columns")
def data_plan(
    input_path,
    output_path,
    task,
    sample_id,
    timestamp,
    features,
    label,
    split_policy,
    window_size,
    n_pred_steps,
    target_features,
    n_clusters,
    clustering_model,
    anomaly_rate,
    allow_irregular,
    temporal_result=None,
    reshape_result=None,
    time_basis=None,
    time_selection=None,
    time_confirmation=None,
    timezone=None,
    frequency=None,
    origin="epoch",
    grid_mode="regularize",
    reducers=None,
    drop_columns=None,
    merge_result=None,
):
    """Write a new plan directory; aliases remain suggestions until roles are declared."""
    roles = (
        dict(
            sample_id_col=sample_id,
            timestamp_col=timestamp,
            label_col=label,
            feature_cols=[c.strip() for c in features.split(",")],
        )
        if features is not None
        else None
    )
    if reshape_result is not None or merge_result is not None:
        roles = {
            key: value
            for key, value in (("sample_id_col", sample_id), ("timestamp_col", timestamp), ("label_col", label))
            if value is not None
        }
        if features is not None:
            roles["feature_cols"] = [c.strip() for c in features.split(",")]
        roles = roles or None
    time_requested = (
        any(
            item is not None
            for item in (
                time_basis,
                time_selection,
                time_confirmation,
                timezone,
                frequency,
                reducers,
                drop_columns,
            )
        )
        or origin != "epoch"
        or grid_mode != "regularize"
    )
    if time_requested or merge_result is not None:
        def action():
            from ai4ts.data_protocols.ingestion import create_ingestion_plan

            selection = (
                selection_from_options(
                    kind=time_basis,
                    saved=time_selection,
                    reason=time_confirmation,
                    timezone=timezone,
                    frequency=frequency,
                    origin=origin,
                    mode=grid_mode,
                    reducers=reducers_from_option(reducers),
                    drop_columns=drop_columns.split(",") if drop_columns else None,
                    task=task,
                )
                if time_requested
                else None
            )
            plan = create_ingestion_plan(
                input_path,
                task,
                roles=roles,
                split_policy=split_policy,
                window_size=window_size,
                horizon=n_pred_steps,
                targets=[c.strip() for c in target_features.split(",")] if target_features else None,
                n_clusters=n_clusters,
                clustering_model=clustering_model,
                anomaly_rate=anomaly_rate,
                allow_irregular=allow_irregular,
                temporal_result=temporal_result,
                reshape_result=reshape_result,
                time_selection=selection,
                **({"merge_result": merge_result} if merge_result is not None else {}),
            )
            return save_plan(plan, output_path)

        return emit_json_result(action)
    try:
        from ai4ts.data_protocols.ingestion import create_ingestion_plan, render_ingestion_preview
    except ImportError as error:
        raise click.ClickException("Installed AI4TS lacks ingestion plans; install matching packages") from error
    plan = create_ingestion_plan(
        input_path,
        task,
        roles=roles,
        split_policy=split_policy,
        window_size=window_size,
        horizon=n_pred_steps,
        targets=[c.strip() for c in target_features.split(",")] if target_features else None,
        n_clusters=n_clusters,
        clustering_model=clustering_model,
        anomaly_rate=anomaly_rate,
        allow_irregular=allow_irregular,
        **({"temporal_result": temporal_result} if temporal_result is not None else {}),
        **({"reshape_result": reshape_result} if reshape_result is not None else {}),
        **({"merge_result": merge_result} if merge_result is not None else {}),
    )
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "preview.md").write_text(render_ingestion_preview(plan))
    click.echo(
        json.dumps(
            {
                "status": plan["status"],
                "plan": str(output / "plan.json"),
                "unresolved": plan["unresolved"],
                "errors": plan["errors"],
            },
            ensure_ascii=False,
        )
    )


def save_plan(plan, output_path):
    """Create a new plan directory and return its bounded JSON summary."""
    from ai4ts.data_protocols.ingestion import render_ingestion_preview
    from ai4ts.data_protocols.temporal_artifacts import write_json

    output = Path(output_path).absolute()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    with (output / "preview.md").open("x", encoding="utf-8") as stream:
        stream.write(render_ingestion_preview(plan))
    return dict(
        status=plan["status"],
        plan=str(output / "plan.json"),
        preview=str(output / "preview.md"),
        unresolved=plan["unresolved"],
        errors=plan["errors"],
        time_selection=plan.get("time_selection"),
        next_step=plan.get("next_step"),
    )
