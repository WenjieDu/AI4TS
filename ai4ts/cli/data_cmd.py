"""
Data protocol commands for the AI4TS CLI.

Provides validation, inspection, and conversion of time-series data files.
"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: Apache-2.0

import os
import sys

import click


def _check_data_protocols():
    """Try to import data_protocols and return availability status."""
    try:
        from ..data_protocols.validation import validate_csv_file, validate_dataframe  # noqa: F401
        return True
    except ImportError:
        return False


@click.group()
def data():
    """Validate, inspect, and convert time-series data files."""
    pass


@data.command()
@click.argument("file", type=click.Path(exists=True))
def validate(file):
    """Validate a CSV file against the AI4TS data protocol.

    FILE is the path to a CSV file to validate.
    """
    try:
        from ..data_protocols.validation import validate_csv_file, ValidationError
    except ImportError:
        click.echo(
            click.style("Error: ", fg="red")
            + "The data_protocols module is not installed. "
            "Install it with: pip install ai4ts[data]"
        )
        sys.exit(1)

    click.echo(f"Validating {click.style(file, bold=True)} ...")

    try:
        from ..data_protocols.validation import validate_dataframe

        df = validate_csv_file(file)
        metadata = validate_dataframe(df)
        click.echo(click.style("✓ Validation passed", fg="green", bold=True))
        click.echo(f"  Samples  : {metadata.n_samples}")
        click.echo(f"  Features : {metadata.n_features} numeric")
        click.echo(f"  Missing  : {metadata.overall_missing_rate:.2%}")
    except ValidationError as e:
        click.echo(click.style("✗ Validation failed", fg="red", bold=True))
        click.echo(f"  {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(click.style("✗ Error during validation", fg="red", bold=True))
        click.echo(f"  {e}")
        sys.exit(1)


@data.command()
@click.argument("file", type=click.Path(exists=True))
def inspect(file):
    """Inspect data structure and display metadata.

    FILE is the path to a CSV file to inspect.
    """
    try:
        import pandas as pd
    except ImportError:
        click.echo(
            click.style("Error: ", fg="red")
            + "pandas is required for data inspection. "
            "Install it with: pip install pandas"
        )
        sys.exit(1)

    click.echo(f"Inspecting {click.style(file, bold=True)} ...\n")

    try:
        df = pd.read_csv(file)
    except Exception as e:
        click.echo(click.style("✗ Failed to read file", fg="red", bold=True))
        click.echo(f"  {e}")
        sys.exit(1)

    n_rows, n_cols = df.shape
    missing_total = df.isnull().sum().sum()
    missing_rate = missing_total / (n_rows * n_cols) if (n_rows * n_cols) > 0 else 0.0

    # try to extract richer metadata from data_protocols
    try:
        from ..data_protocols.validation import validate_dataframe
        from ..data_protocols.utils import ensure_sample_id, ensure_timestamp

        df_proto = ensure_sample_id(df.copy())
        df_proto = ensure_timestamp(df_proto)
        metadata = validate_dataframe(df_proto)

        click.echo(click.style("Dataset Overview", fg="cyan", bold=True))
        click.echo(f"  File          : {file}")
        click.echo(f"  Rows          : {n_rows}")
        click.echo(f"  Columns       : {n_cols}")
        click.echo(f"  Samples       : {metadata.n_samples}")
        click.echo(f"  Numeric feats : {metadata.n_features}")
        click.echo(f"  Has SAMPLE_ID : {metadata.has_sample_id}")
        click.echo(f"  Has TIMESTAMP : {metadata.has_timestamp}")
        click.echo(f"  Has targets   : {metadata.has_classification_target}")
        click.echo(f"  Missing cells : {missing_total} ({metadata.overall_missing_rate:.2%})")
        if metadata.classification_target_names:
            click.echo(f"  Target cols   : {', '.join(metadata.classification_target_names)}")
        click.echo()

        click.echo(click.style("Column Details", fg="cyan", bold=True))
        header = f"  {'Column':<30} {'Role':<20} {'Dtype':<12} {'Missing%':<10}"
        click.echo(header)
        click.echo("  " + "-" * (len(header) - 2))
        for col_desc in metadata.columns:
            col_missing = 0.0
            if col_desc.role.value == "feature" and col_desc.has_missing:
                col_missing = col_desc.missing_rate
            click.echo(
                f"  {col_desc.name:<30} {col_desc.role.value:<20} "
                f"{(col_desc.dtype.value if col_desc.dtype else '-'):<12} {col_missing:<10.2%}"
            )
    except ImportError:
        # Fallback: basic column info without data_protocols
        click.echo(click.style("Dataset Overview", fg="cyan", bold=True))
        click.echo(f"  File          : {file}")
        click.echo(f"  Rows          : {n_rows}")
        click.echo(f"  Columns       : {n_cols}")
        click.echo(f"  Missing cells : {missing_total} ({missing_rate:.2%})")
        click.echo()

        click.echo(click.style("Column Details", fg="cyan", bold=True))
        header = f"  {'Column':<30} {'Dtype':<12} {'Non-null':<10} {'Missing%':<10}"
        click.echo(header)
        click.echo("  " + "-" * (len(header) - 2))
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            col_missing = df[col].isnull().sum() / n_rows if n_rows > 0 else 0.0
            click.echo(f"  {col:<30} {dtype:<12} {non_null:<10} {col_missing:<10.2%}")


@data.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path.")
@click.option(
    "--format", "fmt", type=click.Choice(["csv", "npy"], case_sensitive=False),
    default=None, help="Target format (auto-detected from input if omitted)."
)
@click.option("--context-len", type=int, default=None, help="Context length for conversion.")
@click.option(
    "--scaler", type=click.Choice(["standard", "minmax", "none"], case_sensitive=False),
    default="none", help="Scaler to apply during conversion."
)
def convert(file, output, fmt, context_len, scaler):
    """Convert between CSV and NumPy formats.

    FILE is the path to a .csv or .npy file.
    """
    ext = os.path.splitext(file)[1].lower()

    # auto-detect target format
    if fmt is None:
        fmt = "npy" if ext == ".csv" else "csv"

    try:
        from ..data_protocols.conversion import TimeSeriesConverter
        from ..data_protocols.schemas import ProcessingConfig
        from ..data_protocols.enums import ScalerType
    except ImportError:
        # fallback to basic numpy/pandas conversion
        click.echo(
            click.style("Note: ", fg="yellow")
            + "data_protocols not available, using basic conversion."
        )
        _basic_convert(file, output, fmt, ext)
        return

    # map scaler string to enum
    scaler_map = {
        "standard": ScalerType.STANDARD,
        "minmax": ScalerType.MINMAX,
        "none": ScalerType.NONE,
    }
    scaler_type = scaler_map.get(scaler, ScalerType.NONE)

    config_kwargs = {"scaler_type": scaler_type}
    if context_len is not None:
        config_kwargs["ts_context_len"] = context_len

    try:
        proc_config = ProcessingConfig(**config_kwargs)
        converter = TimeSeriesConverter(config=proc_config)

        if ext == ".csv" and fmt == "npy":
            import numpy as np

            df, metadata = converter.from_csv(file)
            result = converter.to_tensor_samples(training_mode=False)
            out_path = output or os.path.splitext(file)[0] + ".npy"
            np.save(out_path, result["X"])
            click.echo(click.style("✓", fg="green") + f" Converted to {out_path}")
            click.echo(f"  Shape: {result['X'].shape} (samples, timesteps, features)")
        elif ext == ".npy" and fmt == "csv":
            import numpy as np

            arr = np.load(file)
            if arr.ndim == 2:
                import pandas as pd
                out_df = pd.DataFrame(arr, columns=[f"feat_{i}" for i in range(arr.shape[1])])
            elif arr.ndim == 3:
                # 3D tensor: flatten samples back to DataFrame
                converter_new = TimeSeriesConverter(config=proc_config)
                out_df = converter_new.tensor_to_dataframe(arr, strip_anchors=False, strip_padding=False, inverse_scale=False)
            else:
                click.echo(click.style("Error: ", fg="red") + f"Unsupported array shape {arr.shape}.")
                sys.exit(1)
            out_path = output or os.path.splitext(file)[0] + ".csv"
            out_df.to_csv(out_path, index=False)
            click.echo(click.style("✓", fg="green") + f" Converted to {out_path}")
        else:
            click.echo(click.style("Error: ", fg="red") + f"Cannot convert {ext} to {fmt}.")
            sys.exit(1)
    except Exception as e:
        click.echo(click.style("✗ Conversion failed", fg="red", bold=True))
        click.echo(f"  {e}")
        sys.exit(1)


def _basic_convert(file, output, fmt, ext):
    """Fallback conversion using only numpy and pandas."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        click.echo(click.style("Error: ", fg="red") + f"Missing dependency: {e}")
        sys.exit(1)

    try:
        if ext == ".csv" and fmt == "npy":
            df = pd.read_csv(file)
            arr = df.values
            out_path = output or os.path.splitext(file)[0] + ".npy"
            np.save(out_path, arr)
        elif ext == ".npy" and fmt == "csv":
            arr = np.load(file)
            df = pd.DataFrame(arr)
            out_path = output or os.path.splitext(file)[0] + ".csv"
            df.to_csv(out_path, index=False)
        else:
            click.echo(click.style("Error: ", fg="red") + f"Cannot convert {ext} to {fmt}.")
            sys.exit(1)

        click.echo(click.style("✓", fg="green") + f" Converted to {out_path}")
    except Exception as e:
        click.echo(click.style("✗ Conversion failed", fg="red", bold=True))
        click.echo(f"  {e}")
        sys.exit(1)
