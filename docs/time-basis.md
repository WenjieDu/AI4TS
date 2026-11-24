# Explicit time representation

`ai4ts.data_protocols.preview_time_basis(csv_path, roles, ...)` works on one raw
wide CSV or a verified reshape, temporal, or merge result. Supply at most one
of `--reshape-result`, `--temporal-result`, and `--merge-result` with the
result's `data.csv`. The preview reports physical timestamps, original row
indices, per-entity positive interval min/median/max, duplicate and reversed
rows, and finite measurement counts. Without a sampling declaration, missing
time points remain unknown.

```sh
ai4ts-cli data time-basis --input sensors.csv --timestamp time \
  --features temperature --frequency 5min
```

This example uses ISO timestamps with offsets. For a source with naive
timestamps, pass a confirmed `--timezone`.

Naive clocks require an explicit timezone. Numeric epochs, mixed aware/naive
timestamps and ambiguous/nonexistent DST interpretations are rejected. Physical
time comparisons preserve nanoseconds and timezone offsets. Entity strings such
as `001` and `NA` retain their literal identity.

A supplied fixed `frequency`, aware `origin` (or `epoch`) and `mode` preview the
existing temporal grid rules. Exact `regularize` rejects off-grid points;
`aggregate` needs every feature's `mean`, `sum` or `last` reducer and explicit
exclusion of varying metadata when required. Grid support is computed from sparse
keys and finite flags, not aggregate values. `last` considers the final retained
row, including a missing final value. Per-feature finite/missing cell counts and
longest unsupported grid runs are reported; all-empty features remain empty.
Output bounds are checked before grid allocation (100,000 rows/1,000,000
measurement cells, plus the existing byte/source-cell limits). No min-gap/GCD
frequency guess, coarsening search, interpolation or implicit imputation occurs.
Calendar resampling is outside this fixed-duration API.

For modeling, explicitly pass `time_selection` to `create_ingestion_plan`:

```python
selection = {
    "kind": "observation_index",
    "reason": "User accepts record-count windows and ignored physical intervals",
    "impact_acknowledged": True,
    "timezone": None,
    "context": {"task": "imputation"},
}
plan = create_ingestion_plan(path, "imputation", roles=roles,
    split_policy="chronological", window_size=8, time_selection=selection)
```

Only this opt-in path creates ingestion version 2. It keeps roles, parameters,
split and window-preview fields consumed by existing preparation. No selection
keeps version 1 and its original `allow_irregular` meaning/recomputation. Unknown
versions are rejected; neither version is silently migrated.

The saved `time_selection` binds its original decision source, current input,
roles, timezone, task/method context, reason and known consequences with finite
JSON/content identities. It is a declaration, not an authenticated user
signature. Reuse it for unchanged data even after a path rename. Material source,
role, task or grid changes need a new explicit decision. AI4TS treats a model
profile reference as context only; PyPOTS checks its actual package/record bytes.
Saved context also binds window/horizon, ordered targets, evaluation policy and
applicable task parameters. Equivalent integer/string step declarations normalize
to the same record count. Fixed AgenticTSA workflows additionally check that the
profile context describes their actual model, without limiting generic AI4TS use.

`observation_index` maps to existing `allow_irregular=True`; its default False
does not negate the new explicit choice. History and horizon count records, not
physical durations; future wall-clock time is unknown. `regular_grid` records
`grid={mode,frequency,origin,reducers,drop_columns}`. Conflicting old True flags
are rejected. An already matching regular source can continue. Otherwise the
plan returns `needs_preparation` with the existing temporal next step. Naive
observation-index data uses explicit normalize; naive grid data can normalize
and construct the grid in the same existing temporal transform.

Run temporal plan/run/inspect separately. Reuse the saved choice with its verified
derived source and sole `temporal_result`: `origin_source_sha256` remains the
original decision source and `input_sha256` becomes the verified current CSV.
The transformation's mode, frequency, origin, reducers, exclusions and timezone
must match. Never replace the origin digest to hide a transformation. Existing
inherited split rows survive before windows; crossing a partition with an
aggregation bucket remains forbidden. No old scaler/model/threshold is inherited.
Natural missing cells, new grid cells and padding do not acquire evaluation truth.

Physical-time methods may use the raw irregular clock directly; a NN-specific
choice is not a mandatory form for every analysis. Finer grids can be sparse,
coarser aggregation changes resolution/value meaning, and observation indices
discard interval meaning. These are user-selected modeling approximations, not
accuracy, IID, new-install, host or production guarantees.
