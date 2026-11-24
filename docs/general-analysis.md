# Local diagnostics and forecast baselines

`ai4ts.data_protocols.analysis.create_analysis_plan` creates a source-bound
preview for `diagnostics` or `forecast-baselines`. It imports no statistical
backends, prepares no model tensors and never repairs the input.

```python
from ai4ts.data_protocols.analysis import create_analysis_plan, validate_analysis_plan

roles = {"sample_id_col": "device_id", "timestamp_col": "time",
         "feature_cols": ["temperature"], "label_col": None}
plan = create_analysis_plan("source.csv", roles=roles, methods=["acf"])
if plan["status"] == "ready":
    validate_analysis_plan(plan, "source.csv")
```

Missing declarations return `needs_clarification`; invalid or oversized inputs
return `blocked`. Candidates in source inspection are suggestions, not confirmed
roles. Unit mappings are explicit declarations; undeclared units remain unknown.
Matching declarations can be reused without asking the same questions again.

Time diagnostics preserve original rows and string identifiers, including `001`
and `NA`. Each entity has its own intervals and regularity. Duplicate/reversed
timestamps are reported, never sorted away. A missing timestamp column needs an
explicit `time_basis="observation_index"`; this does not enable physical-time
methods or irregular-time forecasting.

`expected_frequency="5min"` declares a schedule anchored at the first observed
timestamp and counts absent grid points only inside the observed first-to-last
span when every timestamp aligns. Without that declaration, irregular intervals
do not imply missing observations. No rows are inserted. Empty feature cells and
missing observation times have separate meanings and denominators.

The source intake limit is 32 MiB and 10,000,000 source cells. Analysis is limited
to eight entity/feature pairs and 4096 selected rows per entity. `rows` can name
an explicit ordered source subset; there is no automatic sampling. `scope` can
select train/val/test from an existing source-bound ingestion `scope_plan`.
Whole-source exploration is descriptive and is not isolated model-selection
evidence.

Baseline plans require ordered targets, a confirmed chronological or
entity_chronological purpose, and an explicit seasonal period when requested.
They reuse the 70/10/20 source split. At 240 rows that is 168/24/48; both val and
test forecast distances start at the training cutoff, so test starts at step25.
This is a fixed-origin protocol, distinct from model L/H rolling windows.

The content digest is not a signature. Changing source contents, roles, selected
rows, parameters or implementation identity requires a new plan. Relocating an
unchanged source is supported. The original source and model ingestion contracts
remain unchanged.

## Execute and inspect saved analyses

The optional `ai4ts.analysis` API and `ai4ts-cli analyze` commands combine
AI4TS's data protocols with explicit statistical and forecasting adapters.
They do not require a neural model or turn all inputs into model windows.

Install `ai4ts[analysis]` for statistics and plots, or `ai4ts[analysis-forecasting]`
for forecast plots. The fixed-origin forecast calculations use core NumPy.
The initial development profile targets macOS arm64/Python 3.13 with statsmodels
0.15.0. Optional packages can have stricter Python requirements than core
AI4TS. Use `capabilities` to inspect dependency availability. These examples
use a wide CSV with `device_id`, `time`, and `temperature` columns.

```sh
ai4ts-cli analyze capabilities --json
ai4ts-cli analyze plan --kind diagnostics --input source.csv \
  --sample-id device_id --timestamp time --features temperature \
  --methods acf,stl --period 12 \
  --output plans/diagnosis
ai4ts-cli analyze run --input source.csv --plan plans/diagnosis/plan.json \
  --output analyses/diagnosis
ai4ts-cli analyze inspect --analysis analyses/diagnosis --view feature --feature temperature
```

The source must be a bounded UTF-8/BOM comma-separated wide CSV. Entity, time,
feature, target and unit declarations are explicit; numeric identifiers are not
automatically features. `--confirmed-plan` reuses source-matching declarations,
including known periods; it never authorizes changed input. Missing declarations
produce concrete questions. The user does not need to hand-write role JSON.
Supply `--sample-id` whenever timestamps belong to multiple devices. With
that option omitted, the CLI treats the data as one series.
Plan/run/inspect never install packages or call a provider.

All plans bind source bytes, column order, original rows, entities, roles, scope,
method parameters, limits and implementation/dependency identities. A relocated
unchanged source works; changing its contents requires a new plan. An explicit
`--rows START:STOP` selects original rows without implicit sampling. An existing
ingestion `--scope-plan` can bind train/val/test diagnosis. Default whole-source
diagnosis is descriptive, not isolated model-selection evidence.

## Supported methods and applicability

- Profile: counts, finite observations, missing rate, missing row runs, robust
  quantiles, constant/all-empty features and separate per-entity time facts.
- ACF: regular timestamps, at least16 finite nonconstant observations,
  `missing='conservative'`, nlags=min(40,n_rows//4). Original grid retained;
  each lag has a pair count, and fewer than8 pairs gives an unsupported value.
- PACF: complete regular series, at least16 points, `method='ywm'`, the same
  bounded lag rule. No unverified confidence interval is added.
- ADF/KPSS: complete regular nonconstant series, at least32 points. ADF has
  a unit-root null and bounded AIC lag selection; KPSS has a stationarity null
  and a fixed bounded lag rule. Actual settings and backend warnings are saved.
  Non-rejection is not proof; contradictory evidence remains uncertain.
- STL: complete regular series, an explicitly declared integer period>=2 and
  at least two cycles, robust decomposition. Trend/season/residual arrays are
  saved and reconstruct the input; this is not causal attribution.
- Lomb–Scargle: actual strictly increasing times, at least32 finite nonconstant
  measurements and an explicit physical search interval. Use
  `--methods lombscargle --period-range 30min,120min`. A fixed256-point angular
  frequency grid is saved. Missing measurements are filtered with their real
  times retained, never compressed into a regular series. A spectral peak is a
  candidate in this interval; significance is not estimated.

No method silently fills, sorts, deduplicates, resamples or drops time rows.
Constant/short/all-empty data still have a profile; unsupported methods retain
their reason. Numeric timestamps require explicit upstream epoch/unit conversion.
Missing cells, missing expected time points and genuinely irregular sampling
are different: `--expected-frequency 5min` declares a grid anchored at the first
observed timestamp, counted only within that observed span. Without a schedule,
missing time-point counts remain unknown. Mixed entity frequencies stay separate.

Diagnostic limits:32MiB/10,000,000 source cells; selected groups <=4096 rows each,
<=8 entity/feature pairs, <=6 methods,256 spectral frequencies,60s computation and
publication worker. No methods means profile only. Exceeding limits requires an
explicit smaller scope; it never silently samples. Internal statistical fits
(for example STL estimation) are statistical computation, not zero computation.

## Fixed-origin forecast baselines

```sh
ai4ts-cli analyze plan --kind forecast-baselines --input source.csv \
  --sample-id device_id --timestamp time --features temperature --targets temperature \
  --split-policy entity_chronological --seasonal-period 12 --output plans/baselines
ai4ts-cli analyze run --input source.csv --plan plans/baselines/plan.json \
  --output analyses/baselines
```

The example evaluates each device on its own future. For one series, omit
`--sample-id` and choose `--split-policy chronological`. Methods are fixed
NumPy last-value and seasonal-naive. `--candidates last-value` explicitly
requests a single reference.
The seasonal period is declared, never selected from test results.

The split uses 70% training, 10% validation, and 20% test rows. A 240-row
series gives 168, 24, and 48 rows respectively. Both candidates prepare their
reference cycle from training and predict validation steps 1 through 24.
Validation raw MAE selects
per entity/target, exact ties favor last-value. Only the selected method predicts
test steps25..72 from the same training cutoff. There is no updating/refitting
on validation/test observations and no rolling-window leaderboard comparison.

Earlier training gaps can remain. Last-value requires a finite last training
value; seasonal-naive conservatively requires >=two periods of history and a
complete last period. Invalid history is not automatically filled. Missing
future truth is excluded using one common mask; no valid truth means no score.
Scores retain target units and are not averaged across different dimensions.
Per entity/target: at most two preparations and three predictions, <=128 future steps; <=8 pairs and
120s per run. Failures preserve partial evidence and never trigger retries.

## Saved evidence and followups

`timeseries-analysis-v1` bundles contain the immutable source snapshot, bound
plan, numeric arrays, results, warnings, operation ledger, real figures and
report. The completion manifest is written last. New runs and optional view
exports require new directories. Failures leave evidence; they do not repair or
overwrite old results. Existing neural-model bundles still use their own tools.

`inspect` supports summary/feature/entity/method views of saved facts and links.
It checks integrity, source rows, masks, denominators and units; fixed baselines
also have a closed-form numeric verification. It does not rerun statistical
tests, load estimators, predict, fit, replay or create new figures. It states that
scope explicitly rather than claiming fresh verification of every p-value.
Use a new plan for a new method, data interval, model or computation budget.

Python callers can use `ai4ts.analysis.diagnose`, `compare_baselines`,
`inspect_analysis` and `available_methods`; plans come from the same CLI/shared
analysis-plan API. For `run`, success, partial results, and failure return exit
codes 0, 2, and 1 respectively. For `plan`, ready, needs clarification, and
blocked return the same codes. A successful inspection exits 0 and preserves the
recorded analysis outcome.
