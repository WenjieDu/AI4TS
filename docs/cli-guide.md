# ai4ts-cli Guidance

Install AI4TS and check the available commands.

```sh
python -m pip install ai4ts
ai4ts-cli version
ai4ts-cli --help
```

To use the CLI from a source checkout, run `python -m pip install .` from the
repository root.

To run the full test suite from the repository root, install the analysis extra
for statistical reference tests and BenchPOTS for data preparation tests.

```sh
python -m pip install '.[analysis]' pytest 'benchpots==1'
python -m pytest tests -q
```

Use `data` for source inspection and preparation, `analyze` for local
statistics and forecast baselines, and `service` for TimeSeries.AI requests.
Each command accepts `--help`.

## Inspect a source

The examples below use `source.csv` with columns `time,temperature`, one
regularly sampled series, and ISO timestamps with UTC offsets. Keep missing
measurements empty. Use inspection to confirm the roles before preparing data.

```sh
ai4ts-cli data inspect source.csv
ai4ts-cli data profile --input source.csv --json
ai4ts-cli data time-basis --input source.csv --timestamp time --features temperature
```

`inspect` reports columns, inferred types, and missing cells. `profile` adds
sample lengths, per-feature missing rates, and sampling summaries.
`task_type_source` indicates whether the task was supplied or inferred.
Use `data validate FILE` to check a file against the AI4TS CSV protocol.
For multiple entities, pass their identifier with `--sample-id` to `time-basis`. See
[time representation](time-basis.md) and [feature units](feature-units.md) for
explicit time and unit declarations.

For numeric format conversion, use `data convert measurements.csv --mode table
-o measurements.npy` to preserve 2D row and column order. CSV defaults to
`samples` mode for 3D arrays. Missing values stay NaN unless a strategy such
as `--missing-value-strategy forward_fill` is specified. NPY-to-CSV uses the
stored dimensions by default. Extracting from NPZ requires an explicit
`--key`.

## Transform data when needed

Choose a transformation based on the input and the required output.

| Input or goal | Command | Guide |
| --- | --- | --- |
| One measurement per row | `data reshape` | [Long-to-wide ingestion](long-format-ingestion.md) |
| A normalized clock, fixed grid, or time buckets | `data temporal` | [Temporal preprocessing](temporal-preprocessing.md) |
| Two to four wide sources with matching timestamps | `data merge` | [Exact-time merge](merge.md) |

Use `plan` to check the transformation, `run` to create a result directory,
and `inspect` to verify the saved output. Keep the result directory alongside
its `data.csv` so downstream plans can verify provenance.

## Prepare a model input plan

The standalone `data plan` records roles, splits, and windows for model data
preparation. For this example, use a single series with enough observations
for training, validation, and test windows of eight records.

```sh
ai4ts-cli data plan --input source.csv --task imputation \
  --timestamp time --features temperature --split_policy chronological \
  --window_size 8 --output intake
```

Read `intake/preview.md` and use `intake/plan.json` once the plan is ready.
For multiple devices, declare `--sample_id` and choose a split that matches
the evaluation goal. For transformed data, pass the resulting CSV and the
matching `--reshape-result`, `--temporal-result`, or `--merge-result`.
The [PyPOTS bridge](pypots-bridge.md) explains how to pass the plan to PyPOTS.
See [task data and exports](task-data-and-exports.md) for state, masks, and
task-specific outputs.

## Export inputs for a forecasting foundation model

Use `data export-tsfm` to write raw-scale context and future CSVs and a
source-bound manifest. Declare known-future covariates with evidence and
provide their values for exactly the forecast horizon. Numeric and categorical
static covariates must be constant within each entity wherever observed.
Use absolute ISO 8601 timestamps. The forecast grid accepts fixed and calendar
offsets such as `1h`, `MS`, and `W-MON`. Omit `--sample-id` to generate
an entity ID for a single series. Generated grids use the 10 million cell limit.
See the declaration format, grid rules, and command example in [forecasting covariate roles](covariate-roles.md).

## Run a local analysis

`analyze plan` records the analysis scope, method, and parameters. Use it for
diagnostics or fixed-origin forecast baselines. Install `ai4ts[analysis]` for
the diagnostic and plotting dependencies.

```sh
ai4ts-cli analyze capabilities --json
ai4ts-cli analyze plan --kind diagnostics --input source.csv \
  --timestamp time --features temperature --output diagnosis-plan
ai4ts-cli analyze run --input source.csv --plan diagnosis-plan/plan.json \
  --output diagnosis
ai4ts-cli analyze inspect --analysis diagnosis
```

For multiple devices, add `--sample-id` to the plan command to keep their
time series separate. See [local diagnostics and forecast baselines](general-analysis.md)
for method assumptions and interpreting saved results.

## Call the remote service

`service` uses the TimeSeries.AI API through `ai4ts.client.TimeSeriesAI`.
Files supplied to these commands are sent to the selected service. Set
`TIMESERIESAI_API_KEY` or use `ai4ts-cli config set-key`. To select a server,
set `TIMESERIESAI_BASE_URL` or run `ai4ts-cli config set-url URL`.
Environment variables take precedence over the CLI configuration.

```sh
ai4ts-cli service learn train.csv
# Set SESSION_ID to the ID printed by learn.
ai4ts-cli service --session-id "$SESSION_ID" forecast --steps 24
ai4ts-cli service --session-id "$SESSION_ID" impute incomplete.csv
ai4ts-cli service --session-id "$SESSION_ID" persist --alias experiment
ai4ts-cli service restore "$SESSION_ID"
```

Pass the ID printed by `learn` or `restore` as `--session-id` before the
task name. `impute`, `forecast`, `classify`, `detect`, `cluster`, and `clean`
use the specified session's data when the optional CSV is omitted. Missing
sessions, HTTP failures, and missing task results return a nonzero exit code.
