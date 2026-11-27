"""Read-only time preview and explicit time-axis declarations."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import click

from .json_output import JsonCommand, emit_json_result


def selection_from_options(
    *,
    kind=None,
    saved=None,
    reason=None,
    timezone=None,
    frequency=None,
    origin="epoch",
    mode="regularize",
    reducers=None,
    drop_columns=None,
    task=None,
):
    """Build a finite declaration from explicit options, or reuse it unchanged."""
    from ai4ts.data_protocols.temporal_preprocessing import declaration

    if saved is not None:
        if (
            any(
                item is not None
                for item in (
                    kind,
                    reason,
                    timezone,
                    frequency,
                    reducers,
                    drop_columns,
                )
            )
            or origin != "epoch"
            or mode != "regularize"
        ):
            raise ValueError("Reuse --time-selection without overriding its confirmed semantics")
        return declaration(saved)
    grid = None
    if kind == "regular_grid":
        grid = dict(
            mode=mode,
            frequency=frequency,
            origin=origin,
            reducers=reducers,
            drop_columns=drop_columns,
        )
    elif (
        frequency is not None
        or reducers is not None
        or drop_columns is not None
        or origin != "epoch"
        or mode != "regularize"
    ):
        raise ValueError("Grid options require an explicit regular_grid choice")
    return dict(
        kind=kind,
        reason=reason,
        impact_acknowledged=bool(reason),
        timezone=timezone,
        grid=grid,
        context=dict(task=task),
    )


def reducers_from_option(value):
    """Parse the same explicit unique feature=method pairs as temporal plan."""
    if value is None:
        return None
    pairs = [pair.split("=") for pair in value.split(",")]
    if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
        raise ValueError("reducers must be unique feature=method pairs")
    return dict(pairs)


@click.command(
    name="time-basis",
    cls=JsonCommand,
    help="Preview one source's time facts and candidate grid; no data production",
)
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--sample-id")
@click.option("--timestamp")
@click.option("--features")
@click.option("--label")
@click.option("--timezone")
@click.option("--frequency")
@click.option("--origin", default="epoch")
@click.option(
    "--grid-mode",
    "mode",
    type=click.Choice(["regularize", "aggregate"]),
    default="regularize",
)
@click.option("--reducers")
@click.option("--drop-columns")
@click.option("--temporal-result", type=click.Path(exists=True, file_okay=False))
@click.option("--reshape-result", type=click.Path(exists=True, file_okay=False))
@click.option("--merge-result", type=click.Path(exists=True, file_okay=False))
def time_basis(**args):
    """Show time structure and coverage for a candidate grid."""

    def action():
        from ai4ts.data_protocols.time_basis import preview_time_basis

        roles = dict(feature_cols=args["features"].split(",")) if args["features"] is not None else None
        for flag, key in (
            ("sample_id", "sample_id_col"),
            ("timestamp", "timestamp_col"),
            ("label", "label_col"),
        ):
            if args[flag] is not None:
                roles = dict(roles or {}, **{key: args[flag]})
        return preview_time_basis(
            args["input_path"],
            roles,
            timezone=args["timezone"],
            frequency=args["frequency"],
            origin=args["origin"],
            mode=args["mode"],
            reducers=reducers_from_option(args["reducers"]),
            drop_columns=(args["drop_columns"].split(",") if args["drop_columns"] else None),
            temporal_result=args["temporal_result"],
            reshape_result=args["reshape_result"],
            merge_result=args["merge_result"],
        )

    emit_json_result(action)
