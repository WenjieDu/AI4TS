# Source-bound feature units

`prepare_for_pypots` accepts optional `feature_units`, independently of the
column-role mapping and `standard-v1` numerical preprocessing. A plain mapping
is an explicit caller/user confirmation; omitted features remain `unknown`.
No unit is inferred from a name or value, and no values are converted.

```python
from ai4ts.data_protocols.pipeline import prepare_for_pypots

train = prepare_for_pypots(
    "train.csv", window_size=8, schema_mapping={"feature_cols": ["temperature", "pressure"]},
    preprocessing="standard-v1", feature_units={"temperature": "degC"},
)
test = prepare_for_pypots(
    "test.csv", window_size=8, schema_mapping={"feature_cols": ["temperature", "pressure"]},
    set_type="test", preprocessing="standard-v1", scaler_state=train["scaler_state"],
    unit_state=train["feature_semantics"],
)
```

For classification, forecasting, clustering, anomaly detection and
representation, the returned training `task_state` also carries units; supplying
that state inherits them. For imputation, supply the separate `unit_state`.
Different files get their own source identity while retaining the training
declaration. An explicit conflicting evaluation unit is rejected.

To declare source evidence rather than user confirmation, use:

```python
from ai4ts.data_protocols.registry import file_sha256

feature_units = {
    "units": {"temperature": "degC"}, "origin": "source",
    "source_sha256": file_sha256("train.csv"), "reference": "author-dataset-version-1",
}
```

The reference is a short public identifier/URL, not document contents. Labels
must be nonempty single lines up to 64 characters; references are limited to
256. Unknown feature keys, wrong source identities and malformed declarations
fail before preparation. Unit metadata binds to ordered features and source via
`feature_semantics`/`feature_semantics_sha256` in the registry and task state;
its digest is independent of the scaler's numerical digest. Missing or changed
declared bindings fail validation. Both fields absent is the legacy path.

Physical units apply to original-scale features and errors (MAE/RMSE unit, MSE
squared unit). Standardized errors are dimensionless. Aggregates of features
with different units have no common physical unit. Identifiers, probabilities,
latent vectors and standardized anomaly scores do not inherit sensor units.

The same inputs/seed without `feature_units` keep the prior arrays, masks,
transformations and numerical results. Adding units changes metadata/file
identities, not data or CSV field names.
