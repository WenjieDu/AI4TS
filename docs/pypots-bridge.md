# Local PyPOTS bridge

AI4TS owns the source roles, split and window plan, and prepared array
semantics. PyPOTS consumes the plan and matching CSV and writes its model H5
input. See the [CLI plan example](cli-guide.md#prepare-a-model-input-plan),
[task data and exports](task-data-and-exports.md), and
[PyPOTS CLI guide](https://docs.pypots.com/en/latest/pypots_cli.html#pypots-cli-preparation).

## Source roles and model arrays

`prepare_for_pypots` accepts a confirmed `SchemaMapping` (or its dict) containing
`sample_id_col`, `timestamp_col`, `feature_cols`, and optional `label_col`.
Identifiers and timestamps retain their source strings; only the selected
features are converted to numbers. Imputation treats labels and notes as
passthrough columns, without classification encoding.
Raw CSV headers must be unique and nonempty; pandas-generated names are not
accepted as source semantics. IDs such as `001`, `NA`, and `NULL` remain literal
strings. Empty IDs are rejected; feature columns retain normal NA parsing.

Rows are stably sorted by timestamp within each sample, with original positions
stored in a version 2 registry. Missing timestamps, duplicate sample timestamps,
infinite features and training features without observations are rejected.
Without a timestamp the declared order is `row_order`; no artificial clock is
created. Irregular sampling requires `allow_irregular=True`, acknowledging
analysis by observation index without resampling. Values remain on their raw
scale; non-overlapping windows limit context across window boundaries.

Legacy registries remain readable for array reconstruction. Semantic CSV export
requires source and prediction fingerprints and exact source-row coverage.

`split_source_rows` assigns entities (at least three) to independent splits or
partitions one series in chronological/row order. Each single-series partition
needs at least two records. Two entities and shorter data are rejected. The
returned row positions are passed to `prepare_for_pypots(row_indices=...)`;
no window can cross a split boundary. Train/validation/test target ratios are
70/10/20, with nonempty small holdouts taking priority over rounded proportions.

Prepare training first, then reuse its `registry.feature_columns` and
`registry.window_size` in all other calls. Explicit windows never shrink;
short samples and tails are padded. Results include `valid_mask` (real rows)
and `indicating_mask` (finite observations artificially hidden). Only val/test
receive repeatable MCAR masking; `full` retains natural missingness alone.

## Opt-in standard-v1 preprocessing

The legacy bridge remains on the raw scale. The `standard-v1` profile uses
`data_protocols.preprocessing.fit_scaler` on unmasked training observations only.
`save_scaler` / `load_scaler` store finite JSON (mean, positive scale, variance,
observation counts, ordered features and a training-observation SHA-256), without
pickle. `transform_values(..., inverse=True)` restores original units, including
standardized zero to the training mean. Constant features have scale 1; a training
feature with no observation is rejected. Loading checks shape, finite values,
feature order and the supplied training fingerprint. This module does not import
PyPOTS, BenchPOTS or torch; StandardScaler is imported only when fitting.

Use `prepare_for_pypots(..., preprocessing="standard-v1", set_type="train",
row_indices=split["rows"]["train"], split_info=split)` first. Its `scaler_state`
is passed unchanged to val/test/full; a missing state is rejected. Each evaluation
split derives an independent seed with `SeedSequence([seed, split_index])` and
uses BenchPOTS point missingness while restoring the caller's NumPy RNG. Train,
val and test keep `X_ori` and static indicating masks; full keeps only natural
missingness. Padding and unknown targets are excluded. SAITS may additionally
apply its existing dynamic training mask. Registries/predictions bind the scaler
SHA; semantic CSV export inverse-scales estimates and preserves observed strings.


## Ordinary CSV ingestion plans

`create_ingestion_plan` inspects UTF-8 (including BOM), comma-separated wide CSV
without fitting a model/scaler, injecting masks or allocating 3-D tensors.
Column aliases and numeric values are evidence, not confirmation. Supply roles
already confirmed in the user's request; missing roles/intent produce
`needs_clarification`, invalid data produces `blocked`. No generated JSON needs
manual editing. The agent asks only unresolved semantic questions.

```python
from ai4ts.data_protocols import create_ingestion_plan, render_ingestion_preview
plan = create_ingestion_plan(
    "source.csv", "imputation",
    roles={"sample_id_col": "device_id", "timestamp_col": "time",
           "feature_cols": ["temperature"], "label_col": None},
    split_policy="entity_chronological", window_size=8,
)
print(render_ingestion_preview(plan))
```

`entity` holds out independent samples/devices (at least three entities).
`chronological`/`row_order` apply to a single stream. Explicit
`entity_chronological` partitions each entity's ordered original rows: validation
=max(2,floor(0.1*n)), test=max(2,floor(0.2*n)), training is the remainder.
It evaluates the same devices' future, not unseen-device generalization or one
global calendar cutoff. No split borrows another split's history. Short entities
fail rather than being dropped. Classification, clustering and representations
require independent samples and cannot use this policy. Unspecified legacy
policies keep their old behavior, including rejecting two independent entities.

The preview reports history length, forecast horizon, and stride. Forecast
windows advance one record. Imputation and anomaly windows are non-overlapping,
and independent samples are padded as whole samples. It also reports counts,
padding, source missingness and a bounded array-memory estimate. The automatic
imputation L rule is an engineering default, not an inferred signal period.
Only a common regular sampling interval supports exact durations such as `24h`.
At five-minute sampling, 24h/2h means 288 history observations and 24 future
steps (history endpoint span is 287 intervals). Gaps/mixed frequency require
explicit observation counts and consent or upstream preprocessing. No resampling,
unit inference, timestamp fabrication, long-table pivot or arbitrary stride is
performed. Actual mask denominators and model quality are unknown at preview.

`validate_ingestion_plan(plan, source)` recomputes source/role/split/window facts,
not just the digest. `verify_prepared_plan(plan, role, prepared)` checks real
shapes, padding and each source/target row mapping before model fit. The plan is
not a scaler/task-state/registry substitute; training alone fits those states.
Content-identical source relocation is allowed; changed bytes/config require a
new plan. The SHA256 is an integrity check, not a signature. AgenticTSA binds
its candidate identity outside the shared data contract.
