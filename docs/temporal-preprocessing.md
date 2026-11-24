# Explicit temporal preprocessing

This local API prepares physical time before diagnostics or model input preparation.
It uses pandas/NumPy, never an estimator, model, provider, or cloud converter.
Time alignment does not prove IID, remove distribution shift, or establish that a
previous model/scaler applies to the derived data.

Use the AI4TS CLI to plan, execute, and inspect a five-minute grid:

```sh
ai4ts-cli data temporal plan --input sensors.csv --output clock-plan \
  --sample-id id --timestamp time --features temperature --mode regularize \
  --frequency 5min --timezone Asia/Singapore
ai4ts-cli data temporal run --input sensors.csv --plan clock-plan/plan.json --output clock-result
ai4ts-cli data temporal inspect --result clock-result
```

```python
from ai4ts.data_protocols import (
    create_temporal_plan, validate_temporal_plan,
    run_temporal_preprocessing, inspect_temporal_result,
)

roles = {"sample_id_col": "id", "timestamp_col": "time", "feature_cols": ["temperature"]}
plan = create_temporal_plan(
    "sensors.csv", mode="regularize", roles=roles, frequency="5min",
    # Required only for timestamps lacking offsets:
    timezone="Asia/Singapore",
)
assert plan["status"] == "ready", (plan["unresolved"], plan["errors"])
validate_temporal_plan(plan, "sensors.csv")
result = run_temporal_preprocessing("sensors.csv", plan, "new-grid-result")
assert result["status"] == "success", result
checked = inspect_temporal_result("new-grid-result")
assert checked["status"] == "success", checked
```

The source must be UTF-8/BOM wide CSV with unique nonempty headers, at most32MiB
and10,000,000 source cells. IDs such as `001` and `NA` stay strings. Empty
measurements stay NaN; infinity and invalid numerical measurements are rejected.
Only explicitly selected measurement columns become numbers.

## Operations

| Mode | Rule |
| --- | --- |
| normalize | Stable physical-time ordering per entity; preserve selected complete records and metadata |
| regularize | Insert missing exact grid points between each entity's first/last observation; new measurements remain NaN |
| aggregate | Group retained observations into fixed-duration intervals; declare a reducer for every feature |

The default duplicate policy is `error`. Explicit `keep-first` or `keep-last`
selects an entire row by original position. Complementary missing values in
duplicate records are never spliced into an apparently complete observation.

Grid frequency is a positive integer duration, e.g. `30s`, `5min`, or `1h`.
The public default origin is UTC epoch; a timezone-aware absolute origin can be
specified. Entities share that origin and frequency. Off-grid timestamps are
rejected for regularize: no snapping, rounding, point removal or extrapolation.
The declared grid does not prove the original acquisition schedule.

Timezone-aware input is converted to UTC with original precision retained.
Naive input requires `timezone`; no machine timezone is assumed. Mixed
aware/naive, numeric epoch, invalid timestamps and ambiguous/nonexistent DST
times are rejected. Monthly/business calendars, multi-file alignment, interpolation,
imputation, signal reconstruction and anti-alias filtering are outside this API.

For aggregate, declare e.g. `reducers={"temperature": "mean", "energy": "sum"}`.
Buckets are `[start,end)`, labelled at `end`: a01:00 observation belongs to the
bucket starting at01:00, not the bucket ending at01:00.

- `mean`: finite-observation arithmetic mean, **not** a time-weighted mean.
- `sum`: sum of finite observations; no finite values means NaN, never zero.
- `last`: value of the last retained record, including NaN; never last-nonempty.

Empty buckets, all-missing buckets, true zeros, original missing cells and new grid
cells remain distinguishable through support counts and unknown reasons. Boundary
intervals record their relation to the observed span; end labels do not certify
real-time availability or complete sampling. Sums do not automatically integrate
rates, difference cumulative meters, or convert units.

Normalize preserves non-feature values. Regularize preserves existing metadata;
new rows copy only entity-constant metadata and otherwise record unknown blanks.
Aggregate copies only entity-constant metadata. Varying non-role columns require
explicit `drop_columns=["note"]` or preparation is blocked. Confirmed labels/role
columns cannot silently be dropped. Excluded columns remain in the source snapshot.

## Plans, limits and artifacts

Plan states are `ready`, `needs_clarification` and `blocked`. Planning parses time,
builds indexes/assignments and counts impact; it does not produce aggregate values,
a derived CSV, scaler state, artificial masks, windows or statistical estimates.
Changed source/declarations/code require a new plan.

Output bounds are100,000 rows,1,000,000 selected feature cells and10,000,000
contribution references. The complete derived CSV must also fit the existing
32MiB/10,000,000-cell intake limits; the preview includes a conservative byte
upper bound covering retained metadata. Limits are checked before grid allocation.
Run workers are bounded to60 seconds; inspect to30 seconds. A caller may lower
these limits. There is no automatic retry, truncation, sampling or overwrite.

Each new result directory contains:

- `source.csv`, `plan.json`, and derived `data.csv`;
- `lineage.jsonl`: per-derived-row original membership and per-feature support,
  numerical contributors, selected last row and unknown reason;
- optional `split.json` and `feature-semantics.json`;
- `report.md`, `execution.json`, worker logs and supervisor record;
- `result.json`, written last after the data/lineage/report inventory is complete.

Original row positions are zero-based. The plan also records unselected duplicates.
Results preserve source bytes and original physical-time strings in the snapshot.
Aggregation is irreversible: lineage identifies inputs but cannot recover minute
observations from their hourly mean. Subsequent model exports concern the derived
source, never replacement of the original observations.

Inspect reads the existing bundle without rewriting it or calling the producer.
It checks content, paths, original membership, grid/intervals, metadata, missingness,
units and splits, and independently checks copied/last values exactly and aggregate
arithmetic at `rtol=atol=1e-12`. Rewriting a hash does not waive these checks.
Hashes are integrity identifiers, not authenticated signatures.

## Evaluation partitions

Supply `split_plan` as a source-matching ready ingestion plan to preserve existing
entity/chronological/entity_chronological roles. Row-order-only semantics are not
physical time. Inserted grid points inherit their entity or temporal interval.
Aggregation requires internal split boundaries to coincide with bucket boundaries;
cross-boundary buckets are rejected rather than discarded, clipped or reassigned.
A train bucket `[01:00,02:00)` labelled02:00 remains train when validation begins
at02:00. Emptying a formerly nonempty role is an error.

Without a prior split, the result is explicitly `unsplit`. A later new evaluation
design is possible, but it is not preservation of an old split. Downstream callers
must pass the explicit temporal result reference to inherit exact derived rows,
source lineage and units. Bare CSV use carries no inherited split claim. Original
fitted scalers/models/thresholds are not inherited; any later train-only data
transformation belongs to the derived training rows and a separate user request.

For downstream data preparation, use `clock-result/data.csv` and
`--temporal-result clock-result` in `ai4ts-cli data plan`. Pass the ready plan
and that same CSV to PyPOTS for H5 packaging. See the
[PyPOTS CLI guide](https://docs.pypots.com/en/latest/pypots_cli.html#pypots-cli-preparation).

This source-workspace functionality is not installed-host, independent-user,
cross-platform or production qualification.
