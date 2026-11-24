# Explicit long-to-wide ingestion

The source has one measurement per row, with confirmed timestamp, variable-name
and numeric value columns, optional entity/unit/label columns. A variable list
defines the selected measurements and their output order. This is one bounded
UTF-8/BOM CSV, not a multi-file join or approximate time alignment.

Use the AI4TS CLI to plan, execute, and inspect the conversion:

```sh
ai4ts-cli data reshape plan --input sensors-long.csv --output reshape-plan \
  --sample-id device --timestamp time --variable metric --value reading \
  --feature temperature --feature pressure --timezone Asia/Singapore
ai4ts-cli data reshape run --input sensors-long.csv --plan reshape-plan/plan.json --output reshape-result
ai4ts-cli data reshape inspect --result reshape-result
```

```python
from ai4ts.data_protocols import create_reshape_plan, validate_reshape_plan

plan = create_reshape_plan(
    "sensors-long.csv",
    roles={"sample_id_col": "device", "timestamp_col": "time",
           "variable_col": "metric", "value_col": "reading"},
    variables=["temperature", "pressure"],
    timezone="Asia/Singapore",  # needed for naive clocks only
)
assert plan["status"] == "ready", (plan["unresolved"], plan["errors"])
validate_reshape_plan(plan, "sensors-long.csv")
```

Plans can be ready, needs_clarification, or blocked. They calculate keys and
counts, not a dense table, model tensors, aggregate values or fitted state.
Changing source bytes/declarations/code needs a new plan. Source is at most
32 MiB and 10,000,000 cells; output at most 100,000 rows and 1,000,000 selected
measurement cells, while still fitting the existing complete CSV reader limits.
Conservative byte estimates include retained metadata and quoting. There is no
automatic clipping, sampling or retry.
The complete serialized plan must also fit the 32 MiB declaration reader;
an oversized mapping is blocked before returning ready. Clarification plans
retain source identity and confirmed fields so a missing timezone can be added.

Physical time defines each entity/time output key. Offset-equivalent timestamps
coincide; three-second differences remain different rows. Naive clocks require
an explicit timezone; numeric epoch, mixed naive/aware and ambiguous/nonexistent
DST are unsupported. UTC output preserves timestamp precision. Entity IDs such
as `001` and `NA` remain strings; a missing entity column means one stream.

The output row set is the union of selected measurement timestamps per entity.
Unselected variables do not create extra all-empty rows. Actual all-empty
measurements do preserve their real time key. Entities follow their first selected
appearance; physical times increase within entity; variables follow the explicit
list. No grid, snapping, interpolation, scaling or imputation is implicit.

Duplicates default to error on entity/time/variable. Explicit keep-first or
keep-last chooses one complete original measurement, including its NaN, units
and metadata. There is no averaging, summing or last-nonempty fallback. Missing
measurement records and records with empty readings both become NaN with distinct
lineage reasons; zero stays zero. Nonempty invalid/infinite readings are rejected.

Variable names become headers without renaming, trimming or numeric encoding.
Header/role collisions block. Labels and metadata may be copied only when selected
records at a wide key agree exactly; varying non-role columns require explicit
drop_columns. Confirmed labels cannot be silently dropped. All original columns
remain in the source snapshot.

Units must agree per variable across selected records/entities. A mixed known/empty
unit needs explicit confirmation; conflicting known units block. Explicit units
can complete unknowns but cannot pretend to convert inconsistent source units.
Unknown remains unknown, with source-bound evidence retained.

The conversion is unsplit. Preserving an existing long-table evaluation is not
supported in this version; a later wide-table evaluation needs its own confirmed
intent. Reshaping does not prove IID, distribution stability or the applicability
of any old fitted model, scaler or threshold.

## Execute and inspect

```python
from ai4ts.data_protocols import run_reshape, inspect_reshape_result

result = run_reshape("sensors-long.csv", plan, "new-reshape-result")
assert result["status"] == "success", result
checked = inspect_reshape_result("new-reshape-result")
assert checked["status"] == "success", checked
```

Each new bundle retains source.csv, plan.json, wide data.csv, lineage.jsonl with
the selected source row and support for each measurement cell, row-disposition.json,
optional feature-semantics.json, report.md, execution.json and worker/supervisor
logs. result.json is written last. The original long file remains unchanged;
explicitly dropped metadata still exists in its snapshot.

Inspect independently verifies source keys, copying/NaN reasons, metadata, units,
dispositions and support. It never calls the producer or an estimator and does not
rewrite the bundle. Rehashing a changed value does not waive semantic checks.
Run/inspect have owned 60/30-second worker limits, optionally shortened. Existing
directories and symlinks are rejected; timeouts/partial writes retain evidence
without a valid completion marker or automatic retry. These are local integrity
checks, not authenticated signatures or production qualification.

## Explicit downstream references

`create_ingestion_plan`, `create_analysis_plan` and `create_temporal_plan` accept
`reshape_result="new-reshape-result"` with that result's data.csv. Roles and units
are independently verified and reused; an explicit subset/reordering must remain
within confirmed measurements, and entity/time/label roles cannot silently change.
An ingestion plan needs a newly confirmed evaluation intent. Profile-only analysis
does not create partitions, masks or model tensors.

For reshape→temporal→later preparation, use reshape_result on the temporal plan,
then only temporal_result downstream. The temporal result retains a content-bound
local parent path. Parent loss/corruption blocks continuation; preserve it. Giving
both top-level references is rejected. Only this bounded two-stage chain is
supported; no whole parent bundle is copied and no automatic path discovery occurs.
If a new ready wide-source split_plan is explicitly supplied to time preparation,
the existing exact partition and aggregation-boundary rules remain unchanged.

The independent source/measurement lineage does not alter WindowRegistry's
one-time coverage of final wide-source rows. Numeric preparation and artificial
holdouts belong to the final training source, never the original long row count.

For a requested descriptive profile, create the existing analysis plan with the
verified wide source. For a separately confirmed new evaluation, use ingestion
and raw-scale preparation (no model or scaler fitting):

```python
from ai4ts.data_protocols.analysis import create_analysis_plan
from ai4ts.data_protocols.ingestion import create_ingestion_plan
from ai4ts.data_protocols.pipeline import prepare_for_pypots

wide = "new-reshape-result/data.csv"
profile = create_analysis_plan(wide, reshape_result="new-reshape-result")
assert profile["status"] == "ready", profile
intake = create_ingestion_plan(
    wide, "imputation", reshape_result="new-reshape-result",
    split_policy="entity_chronological", window_size=12, allow_irregular=True,
)
assert intake["status"] == "ready", intake
train = prepare_for_pypots(
    wide, ingestion_plan=intake, window_size=12, set_type="train", preprocessing=None,
)
```

Here `allow_irregular=True` is an explicit choice to use observation order. If a
regular grid is requested instead, preserve the new wide-source partitions through
one temporal stage, then reference only its final result:

```python
from ai4ts.data_protocols import create_temporal_plan, run_temporal_preprocessing

grid_plan = create_temporal_plan(
    wide, "regularize", reshape_result="new-reshape-result",
    split_plan=intake, frequency="5min",
)
assert grid_plan["status"] == "ready", grid_plan
grid = run_temporal_preprocessing(wide, grid_plan, "new-grid-result")
assert grid["status"] == "success", grid
final_intake = create_ingestion_plan(
    "new-grid-result/data.csv", "imputation", temporal_result="new-grid-result", window_size=12,
)
assert final_intake["status"] == "ready", final_intake
train_profile = create_analysis_plan(
    "new-grid-result/data.csv", temporal_result="new-grid-result", scope="train",
)
assert train_profile["status"] == "ready", train_profile
```

The declared `5min` schedule must actually contain the observed instants; planning
blocks an off-grid source. These examples assume enough observations for the new
partition/window declarations. Run local diagnostics with `ai4ts-cli analyze`.
For model inputs, create a ready `ai4ts-cli data plan` with `--reshape-result`
or the final `--temporal-result` and give PyPOTS the matching CSV and plan.
See [temporal preprocessing](temporal-preprocessing.md) for the time operations
and the [PyPOTS CLI guide](https://docs.pypots.com/en/latest/pypots_cli.html#pypots-cli-preparation)
for H5 output.
