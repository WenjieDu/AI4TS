# Forecasting covariate roles

Declare which inputs are available at prediction time. A column's values alone
cannot establish whether it was known at the forecast origin. For example, a
published promotion schedule may be known in advance, while actual promotions
recorded after the fact belong in the context.

## Declaration

Pass a JSON object keyed by selected non-target feature column. Column names
match the source exactly. For example:

```json
{
  "scheduled_promotion": {
    "role": "known",
    "kind": "numeric",
    "evidence": {"origin": "source", "reference": "Published promotion schedule, rev. 4"}
  },
  "region": {
    "role": "static",
    "kind": "categorical",
    "evidence": {"origin": "user-confirmed", "reference": "Store master data confirmation"}
  },
  "observed_weather": {"role": "past", "kind": "numeric"}
}
```

| Field | Values and meaning |
| --- | --- |
| `role` | `past` for context-only, `known` for available at the forecast origin, `static` for entity-level properties |
| `kind` | `numeric` for numbers, `categorical` for category labels |
| `evidence` | Required for `known` and `static`, optional for `past` |

For evidence, supply `origin` as `source` or `user-confirmed` and a nonempty,
trimmed, single-line `reference` of at most 256 characters.

Columns omitted from the declaration are recorded as
`{"role":"past","kind":"numeric","evidence":{"origin":"default"}}`. AI4TS
reserves the `default` origin for this purpose. Declare a categorical column
explicitly. Targets, entity identifiers, timestamps, unselected columns, and
unknown fields are invalid declaration entries.

## Python API

Import `export_tsfm_frames`, `validate_future_covariates`, and
`prepare_for_pypots` from `ai4ts.data_protocols`.

```python
def export_tsfm_frames(
    history_csv: str | Path,
    output: str | Path,
    *,
    sample_id_col: str | None = None,
    timestamp_col: str,
    feature_columns: list[str],
    target_features: list[str],
    n_pred_steps: int,
    covariates: dict | None = None,
    future_csv: str | Path | None = None,
    frequency: str | None = None,
) -> dict:
    ...

def validate_future_covariates(
    history: pd.DataFrame,
    future: pd.DataFrame | None,
    *,
    sample_id_col: str | None = None,
    timestamp_col: str,
    feature_columns: list[str],
    target_features: list[str],
    n_pred_steps: int,
    covariates: dict | None = None,
    frequency: str | None = None,
) -> pd.DataFrame:
    ...
```

`feature_columns` selects the ordered context features, including all targets.
`target_features` is a nonempty ordered subset. The context output starts with
the entity identifier and timestamp, followed by the selected features in
order. The returned future frame uses the same two leading columns, followed by
the declared known columns in feature order. The input frames are unchanged.

Omit `sample_id_col` or pass `None` when the inputs describe a single series
without an entity column. Both outputs use the string entity ID `"0"` in a
generated `item_id` column. AI4TS appends underscores if needed to avoid
colliding with a history column.

For forecasting preparation, pass the same optional `covariates` keyword to
`prepare_for_pypots(..., task="forecasting", covariates=...)`. The resolved
mapping and its SHA-256 are recorded as `covariates` and `covariates_sha256` in
the training task state and window registry. Evaluation and full preparation
reuse the training declaration and reject a conflicting override.

## PyPOTS forecasting inputs

For newly prepared training state, `X` begins with `target_features` in the
requested order. Covariates follow in source feature order. The registry
records this model order in `feature_columns` and preserves the source
order in `column_roles["feature_cols"]`. Previously saved forecasting state
continues to use its original model input layout. Refit to adopt this routing
for an existing model.

When preparing from a Python ingestion plan, supply `covariates` to
`create_ingestion_plan` to preview and bind the generated column order and
tensor width before preparation.

- Past numeric features retain their historical values. A known numeric
  feature becomes `<col>@lead`. At input position `i` it reads position
  `i + n_pred_steps`. Each window needs at least `n_pred_steps` history
  positions to carry the entire known horizon. Training, validation, and
  test use contiguous rows from their own split.
- Each categorical covariate becomes one encoding column per target,
  including the single-target case. Names are `<col>@te:<target>` in
  `target_features` order. Categorical covariates use context values
  for encoding, including when their declaration has role `known`.
- A static covariate must agree across nonmissing observations within
  an entity. Each window broadcasts the value available at its origin.
  If the value has yet to be observed, a numeric static input stays
  missing and a categorical static input uses the missing category.

Each encoding column uses only observed training targets for its
corresponding target. With smoothing equal to one, the encoded value is
`(entity_mean + category_sum) / (1 + category_count)`. Missing labels have
their own category. An unseen label uses the entity's training mean. A
previously unseen entity, or an entity without observations for this target,
uses that target's global training mean. Save `task_state` to reuse these
lookups in validation, test, and full preparation. With `standard-v1`,
numeric leads use their source feature's training scaler and each encoding
uses its corresponding target's training scaler.

For known numeric leads, supply `future_csv` to
`prepare_for_pypots(..., task="forecasting", set_type="full", task_state=...)`.
The table follows the same horizon validation described below. An explicit
`frequency` can be supplied for that validation. The last historical
window uses the supplied known values for its horizon. Earlier windows use
the known values in history. The external CSV fingerprint and horizon
timestamps are bound into the registry. The final `X_pred` stays missing and
unscored. `export_forecasts` emits the target names in `target_features` order.

## History and future validation

1. Provide nonempty history with unique, nonempty column names and a
   timestamp column. If supplied, the entity column must be distinct and
   contain nonempty strings. CSV preserves identifiers such as `001`.
   Use absolute ISO 8601 timestamps. Naive timestamps are interpreted as
   UTC. Numeric epochs require upstream conversion. Column names and entity
   strings are case-sensitive.
2. History must have unique timestamps within each entity and a common regular
   cadence. Supply a positive pandas offset alias such as `"5min"`, `"MS"`,
   `"QE"`, `"W-MON"` or `"B"`, or let AI4TS infer it with `pd.infer_freq`
   from each entity with at least three observations. Inferred offsets must
   agree and all entities must follow the resulting grid. If no entity has
   three observations, matching two-point intervals can establish a fixed
   cadence. Supply an explicit offset to use a calendar rule in that case
   or when all entities are singletons. Unordered rows are sorted by entity
   and timestamp.
3. A future table is required when any covariate has role `known`. Its columns
   must be exactly the timestamp, any declared identifier, and all known
   covariates. Targets, past and static covariates, and columns absent from
   history are rejected. If there are no known covariates and no future
   table, AI4TS generates the entity and timestamp rows.
4. The future table must cover the same entities as history and contain exactly
   `n_pred_steps` unique timestamps per entity. The horizon must be a positive
   integer. For each entity the future timestamps must equal the next
   consecutive grid points after its final history timestamp, calculated
   as `last_timestamp + k * offset` for each step `k` starting at one.
5. Selected numeric columns and targets must contain numbers or missing values.
   Infinities are rejected. Blank cells are missing, and numeric `NaN` and
   `nan` tokens are also missing. Category text is preserved. Missing cells
   stay missing, including in declared known columns.
6. A static covariate's nonmissing values must agree within each entity.
   Different entities may have different values. Missing cells are preserved.

An invalid declaration or table raises `ValueError` with the failed check.

## Export and manifest

`export_tsfm_frames` writes `context.csv`, `future.csv`, and `manifest.json`
to a new output directory and returns the manifest. Both CSVs have one row
per entity and timestamp. Values retain their source scale. The export uses
no imputation, scaling, aggregation, or categorical encoding.

| Manifest field | Meaning |
| --- | --- |
| `schema`, `status` | `ai4ts-tsfm-frames-v1` and `success` |
| `sample_id_col`, `timestamp_col` | Entity and timestamp column names |
| `feature_columns`, `target_features` | Ordered context features and targets |
| `covariates`, `covariates_sha256` | Resolved roles, kinds, and evidence, plus the canonical declaration hash |
| `n_pred_steps`, `frequency` | Horizon and normalized offset alias, such as `1h` or `MS` |
| `time_basis`, `timezone` | `regular_timestamp` and `UTC` |
| `source_sha256`, `future_source_sha256` | History and future CSV hashes. The second is null for a generated grid |
| `files` | SHA-256 of each output CSV, keyed by filename |
| `consequences` | List of output effects |

The consequences record zero dropped rows, unselected context columns, UTC
ordering, and future grid generation when applicable. For a single series
without an entity column, `generated_entity_id` records the generated column
name and its string value.

## CLI

```sh
ai4ts-cli data export-tsfm --input history.csv --future future.csv \
  --sample-id item_id --timestamp timestamp \
  --features demand,scheduled_promotion,region,observed_weather --targets demand \
  --n-pred-steps 12 --frequency 1h --covariates covariates.json --output tsfm-input
```

`--sample-id` is optional for a single series without an entity column.
`--covariates`, `--future`, and `--frequency` follow the declaration and
cadence rules above. On success the command prints the manifest as JSON.
Validation failures produce a JSON error and a nonzero exit.
