"""Summarize local time-series CSV structure for users and agents."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json

import click


@click.command(name="profile", help="Summarize CSV sample lengths, feature missingness, and sampling")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="Input CSV file path")
@click.option(
    "--task",
    default=None,
    type=click.Choice(
        ["imputation", "classification", "forecasting", "clustering", "anomaly_detection", "representation"]
    ),
    help="Optional task hint. Without this option, the profile reports an inferred hint.",
)
@click.option(
    "--json", "json_output", is_flag=True, default=False, help="Output raw JSON (default: human-readable summary)"
)
def data_profile(input_path, task, json_output):
    """Analyze a CSV dataset and generate a DataProfile.

    The DataProfile is a lightweight JSON structure describing the dataset's
    statistics, column schema, timestamp regularity, and sample lengths.
    Agents use this to make informed decisions about preprocessing strategy.
    """
    from ai4ts.data_protocols.profile import analyze_dataset

    profile = analyze_dataset(input_path, task_type=task)
    d = profile.to_dict()
    d["task_type_source"] = "provided" if task is not None else "inferred"

    if json_output:
        click.echo(json.dumps(d, indent=2))
    else:
        stats = d["dataset_stats"]
        schema = d["schema_mapping"]
        ts_info = d["timestamp_info"]

        print(f"\n{'=' * 65}")
        print(f"Data Profile: {input_path}")
        print(f"{'=' * 65}")
        print(f"  Samples:          {stats['n_samples']}")
        print(f"  Features:         {stats['n_features']}")
        print(f"  Total rows:       {stats['total_rows']}")
        print(f"  Missing rate:     {stats['missing_rate']:.2%}")
        print(
            f"  Sample lengths:   min={stats['min_sample_length']}, "
            f"max={stats['max_sample_length']}, "
            f"avg={stats['avg_sample_length']:.1f}"
        )
        print(f"  Variable length:  {stats['has_variable_length']}")
        print("\n  Per-feature missing rates:")
        for feature, rate in stats["per_feature_missing_rate"].items():
            print(f"    {feature}: {rate:.2%}")

        print("\n  Schema:")
        print(f"    SAMPLE_ID:      {schema['sample_id_col'] or '(not present)'}")
        print(f"    TIMESTAMP:      {schema['timestamp_col'] or '(not present)'}")
        print(f"    Features:       {schema['feature_cols']}")
        print(f"    Label column:   {schema['label_col'] or '(none)'}")
        if schema["n_classes"]:
            print(f"    Classes:        {schema['n_classes']}")
        print(f"\n  Task hint ({d['task_type_source']}): {d['task_type']}")

        if ts_info["median_step_seconds"] is not None:
            regularity = "regular" if ts_info["is_regular"] else "irregular"
            print(f"  Sampling: {regularity}, median interval {ts_info['median_step_seconds']:g}s")
        if ts_info.get("warning"):
            print(f"\n  ⚠ WARNING: {ts_info['warning']}")

        # Suggest appropriate strategy
        if stats["has_variable_length"] or stats["max_sample_length"] > 200:
            if d["task_type"] == "classification":
                print(f"\n  → Suggested strategy: pad_only (pad to max_len={stats['max_sample_length']})")
            else:
                print("\n  → Suggested strategy: sliding_window (non-overlapping, window_size≤200)")
        else:
            print("\n  → Suggested strategy: direct (uniform length ≤ 200)")

        print(f"\n{'=' * 65}\n")
