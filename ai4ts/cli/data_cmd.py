"""
Data protocol commands for the AI4TS CLI.

Provides validation, inspection, and conversion of time-series data files.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import click

from .json_output import JsonCommand, emit_json_result
from .merge import merge
from .plan import data_plan
from .profile import data_profile
from .reshape import reshape
from .temporal import temporal
from .time_basis import time_basis


@click.group()
def data():
    """Inspect, plan, and prepare time-series data."""


@data.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def validate(file):
    """Check CSV protocol rules and report pass or fail."""
    from ..data_protocols.validation import ValidationError, validate_csv_file, validate_dataframe

    try:
        validate_dataframe(validate_csv_file(file))
    except ValidationError as error:
        raise click.ClickException(f"Validation failed: {error}") from error

    click.echo("Validation passed.")


@data.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def inspect(file):
    """Report columns, inferred types, and missing cells in the source CSV."""
    from ..data_protocols.constants import CLAF_TARGET_SYMBOL, COL_NAME_SAMPLE_ID, COL_NAME_TIMESTAMP
    from ..data_protocols.validation import ValidationError, validate_csv_file

    try:
        frame = validate_csv_file(file)
    except ValidationError as error:
        raise click.ClickException(f"Inspection failed: {error}") from error

    rows, columns = frame.shape
    counts = frame.isna().sum()
    total = int(counts.sum())
    targets = [column for column in frame.columns if CLAF_TARGET_SYMBOL in column]

    click.echo(f"Dataset: {file}")
    click.echo(f"  Rows          : {rows}")
    click.echo(f"  Columns       : {columns}")
    click.echo(f"  Has SAMPLE_ID : {COL_NAME_SAMPLE_ID in frame.columns}")
    click.echo(f"  Has TIMESTAMP : {COL_NAME_TIMESTAMP in frame.columns}")
    click.echo(f"  Has targets   : {bool(targets)}")
    if targets:
        click.echo(f"  Target cols   : {', '.join(targets)}")
    click.echo(f"  Missing cells : {total}/{frame.size} ({total / frame.size:.2%}) across all columns")

    click.echo("\nColumn Details")
    click.echo(f"  {'Column':<30} {'Inferred dtype':<15} {'Non-null':<10} Missing")
    for name, dtype, missing in zip(frame.columns, frame.dtypes, counts):
        click.echo(f"  {name:<30} {str(dtype):<15} {rows - missing:<10} {missing}/{rows} ({missing / rows:.2%})")


@data.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path.")
@click.option(
    "--format", "fmt", type=click.Choice(["csv", "npy"], case_sensitive=False),
    default=None, help="Target format (auto-detected from input if omitted)."
)
@click.option(
    "--mode",
    type=click.Choice(["table", "samples"]),
    help="2D table or 3D samples. CSV defaults to samples. NPY uses its stored dimensions by default.",
)
@click.option("--key", help="Array key to extract for NPZ input.")
@click.option("--context-len", type=int, default=None, help="Context length for CSV in samples mode.")
@click.option(
    "--scaler", type=click.Choice(["standard", "minmax", "none"], case_sensitive=False),
    default="none", help="Scaler for CSV in samples mode."
)
@click.option(
    "--missing-value-strategy",
    type=click.Choice(["nan_keep", "forward_fill"]),
    help="Missing-value handling for CSV in samples mode (default: nan_keep).",
)
def convert(file, output, fmt, mode, key, context_len, scaler, missing_value_strategy):
    """Convert CSV and NPY data or extract a selected NPZ array to NPY."""
    from pathlib import Path

    source = Path(file)
    ext = source.suffix.lower()
    fmt = fmt or ("npy" if ext in (".csv", ".npz") else "csv")
    if (ext, fmt) not in ((".csv", "npy"), (".npy", "csv"), (".npz", "npy")):
        raise click.UsageError("Supported conversions: CSV to NPY, NPY to CSV, and NPZ to NPY.")
    if ext == ".npz" and key is None:
        raise click.UsageError("NPZ extraction requires --key.")
    if ext != ".npz" and key is not None:
        raise click.BadParameter("Only supported for NPZ extraction.", param_hint="--key")
    sample_export = ext == ".csv" and mode != "table"
    if not sample_export:
        for option, supplied in (
            ("--context-len", context_len is not None),
            ("--scaler", scaler != "none"),
            ("--missing-value-strategy", missing_value_strategy is not None),
        ):
            if supplied:
                raise click.BadParameter("Only supported for CSV to NPY in samples mode.", param_hint=option)

    out_path = output or str(source.with_suffix(f".{fmt}"))
    try:
        import numpy as np

        if sample_export:
            from ..data_protocols.conversion import TimeSeriesConverter
            from ..data_protocols.schemas import ProcessingConfig

            options = {"scaler_type": scaler}
            if context_len is not None:
                options["ts_context_len"] = context_len
            if missing_value_strategy is not None:
                options["missing_value_strategy"] = missing_value_strategy
            converter = TimeSeriesConverter(ProcessingConfig(**options))
            converter.from_csv(source)
            array = converter.to_tensor_samples(training_mode=False)["X"]
        elif ext == ".csv":
            import pandas as pd

            from ..data_protocols.constants import CLAF_TARGET_SYMBOL, RESERVED_COLUMNS

            table = pd.read_csv(source)
            if any(name in RESERVED_COLUMNS or CLAF_TARGET_SYMBOL in name for name in table.columns):
                raise ValueError("Table mode requires feature columns only. Remove role columns or use samples mode.")
            array = table.to_numpy()
            if array.dtype.kind not in "biuf":
                raise ValueError("Table mode requires numeric CSV columns.")
        elif ext == ".npz":
            with np.load(source, allow_pickle=False) as archive:
                if key not in archive.files:
                    raise ValueError(f"Array key {key!r} not found in NPZ. Available keys: {', '.join(archive.files)}")
                array = archive[key]
        else:
            array = np.load(source, allow_pickle=False)

        expected_rank = {"table": 2, "samples": 3}.get(mode)
        if expected_rank is not None and array.ndim != expected_rank:
            raise ValueError(f"{mode} mode requires {expected_rank} dimensions, got {array.ndim}.")

        if fmt == "npy":
            np.save(out_path, array, allow_pickle=False)
        else:
            if array.ndim == 2:
                import pandas as pd

                frame = pd.DataFrame(array, columns=[f"feat_{i}" for i in range(array.shape[1])])
            elif array.ndim == 3:
                from ..data_protocols.conversion import TimeSeriesConverter

                frame = TimeSeriesConverter().tensor_to_dataframe(
                    array, strip_anchors=False, strip_padding=False, inverse_scale=False
                )
            else:
                raise ValueError(f"Unsupported array shape {array.shape}.")
            frame.to_csv(out_path, index=False)
    except ImportError as error:
        raise click.ClickException(f"Conversion dependencies unavailable: {error}") from error
    except Exception as error:
        raise click.ClickException(f"Conversion failed: {error}") from error

    click.echo(f"Converted to {out_path}")
    if fmt == "npy":
        click.echo(f"  Shape: {array.shape}")


@data.command("export-tsfm", cls=JsonCommand)
@click.option("--input", "history", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--future", type=click.Path(exists=True, dir_okay=False))
@click.option("--sample-id", required=True)
@click.option("--timestamp", required=True)
@click.option("--features", required=True, help="Comma-separated ordered context features, including targets.")
@click.option("--targets", required=True, help="Comma-separated ordered target features.")
@click.option("--n-pred-steps", required=True, type=int)
@click.option("--frequency", help="Fixed cadence, inferred from history if omitted.")
@click.option("--covariates", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path(file_okay=False))
def export_tsfm(history, future, sample_id, timestamp, features, targets, n_pred_steps, frequency, covariates, output):
    """Export raw-scale forecast context, future inputs, and a manifest."""

    def action():
        from ..data_protocols import export_tsfm_frames
        from ..data_protocols.temporal_preprocessing import declaration

        return export_tsfm_frames(
            history,
            output,
            sample_id_col=sample_id,
            timestamp_col=timestamp,
            feature_columns=features.split(","),
            target_features=targets.split(","),
            n_pred_steps=n_pred_steps,
            covariates=declaration(covariates) if covariates is not None else None,
            future_csv=future,
            frequency=frequency,
        )

    emit_json_result(action)


data.add_command(data_profile)
data.add_command(data_plan)
data.add_command(time_basis)
data.add_command(temporal)
data.add_command(reshape)
data.add_command(merge)
