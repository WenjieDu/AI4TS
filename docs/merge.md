# Exact multi-source time-series ingestion

Join wide CSV sources by exact timestamps and, when present, common entity IDs.
The example below assumes `temperature.csv` has columns `time,temperature` and
`pressure.csv` has columns `time,pressure`. Both files use timestamps with
UTC offsets and describe the same single series.

## Plan, run, and inspect

Save the following as `sources.json` alongside the CSVs and run the commands
from that directory.

```json
{
  "sources": [
    {
      "source_id": "temperature",
      "path": "temperature.csv",
      "roles": {
        "sample_id_col": null,
        "timestamp_col": "time",
        "feature_cols": ["temperature"],
        "label_col": null
      },
      "retain_columns": [],
      "drop_columns": []
    },
    {
      "source_id": "pressure",
      "path": "pressure.csv",
      "roles": {
        "sample_id_col": null,
        "timestamp_col": "time",
        "feature_cols": ["pressure"],
        "label_col": null
      },
      "retain_columns": [],
      "drop_columns": []
    }
  ]
}
```

```sh
ai4ts-cli data merge plan --sources sources.json --join left \
  --primary temperature --single-series --output merge-plan
ai4ts-cli data merge run --plan merge-plan/plan.json --output merged
ai4ts-cli data merge inspect --result merged
```

Read `merge-plan/preview.md` and proceed once the plan is ready. This left join
keeps timestamps from the temperature source. The completed result includes
`merged/data.csv` and match and lineage records.

## API and join rules

`create_merge_plan(sources, join, primary=None, entity_meaning=None, single_series=False)` plans two to four wide CSV sources. `validate_merge_plan(plan)` rechecks all inputs/parents; `run_merge(plan, new_directory, timeout=60)` produces once, and `inspect_merge_result(directory, timeout=30)` independently checks saved snapshots and actual values. No model, estimator or scaler is constructed/fitted.

Each source declares `source_id`, `path`, `roles`, optional `timezone`/`feature_units`, and explicit ordered `retain_columns`/`drop_columns` lists (empty lists are meaningful). Roles contain ordered `feature_cols` plus entity, physical timestamp and optional label roles. IDs retain literal strings. All identified inputs need a confirmed common `entity_meaning`; all no-entity inputs require `single_series=True`. Mixed inputs cannot be broadcast. Source IDs are an ASCII letter followed by at most 63 ASCII letters/digits/underscores/hyphens. Source/result paths reject symlinks.

`join="left"` retains primary keys; `join="outer"` retains the union. Primary defaults to the first source. Entity order follows primary first appearance then auxiliary new appearances; each entity is ordered by physical time. Feature order follows source declaration and selected variable order. Structural columns use primary names. Exact matching uses entity/UTC nanoseconds: equivalent offsets match, subsecond differences stay distinct. No nearest/as-of matching, rounding, resampling, interpolation, ID normalization or clock correction. Duplicate keys and feature/role/projection conflicts block before production. Naive clocks need timezone; mixed/invalid/numeric/ambiguous clocks fail.

Primary metadata/label must explicitly be retained or dropped; auxiliary metadata must explicitly be dropped. Confirmed auxiliary labels are unsupported. Outer-only keys have unknown primary metadata/labels. Units retain per-source evidence and the existing `unknown` sentinel, without conversion or inference.

## Inputs and limits

Inputs are raw wide sources or have one verified `reshape_result`/`temporal_result` (the latter may have a reshape parent). No merge in an input chain; duplicate/cyclic directories and unsupported parent shapes fail. At most four branches/eight pre-merge results plus this merge and one downstream temporal result are supported. Explicit derived parents stay available and are inspected under the same owned worker deadline, without grandchildren. Raw originals may disappear after snapshots are saved.

Bounds: each direct CSV 32 MiB, combined 64 MiB/10 million cells; output 100,000 rows/1 million measurement cells/10 million total cells/32 MiB, at most 1 million contribution references. Keys, projections, conservative output bytes and parent contributions are checked before allocation. Oversized declarations or sparse unions fail rather than being sampled/truncated.

## Outputs and downstream use

New bundles contain `sources/<source_id>/source.csv`, `plan.json`, `data.csv`, `feature-semantics.json`, `lineage.jsonl`, `row-disposition.json`, `match-summary.json`, report/execution/worker/supervisor facts, then `result.json` last. Cell lineage preserves source/row/original time and verified parent-cell evidence. Unmatched-source, original-missing, parent grid-added/absent-measurement and zero remain distinct. Key matches and finite feature support have separate denominators. Independent inspection rebuilds keys, values, missingness, metadata, units and dispositions from snapshots/parents without a producer; rehashing a changed output does not prove correctness.

Outputs are unsplit; `preserve_split=True` rejects unsupported evaluation migration. Ingestion, analysis, time preview and temporal planning accept a unique `merge_result`, mutually exclusive with top-level reshape/temporal references. After merge→temporal, pass only the final `temporal_result`; it retains/verifies its merge parent. Model preparation needs a separately confirmed final-table evaluation design and generic time-representation choice. Irregular unions or aligned clocks/features do not certify IID, old-model compatibility, real-time availability or no future leakage.
