# Task data and exports

This reference describes AI4TS model inputs, training-derived data state, and
export fields. See the [PyPOTS bridge](pypots-bridge.md) for source roles,
imputation arrays, and ingestion plans. Model selection, fitting, and held-out
evaluation are covered by the PyPOTS and AgenticTSA workflow guides.

## Training-derived data state

Prepare the training split first and save `result['task_state']`. Pass that state and the same
preprocessing mode to val/test/full. The state contains training roles, fixed
window, classes or forecast settings, training content fingerprints, and numeric
scaler state. State is never refitted on those splits. The registry binds its
SHA-256 along with the current source, feature order and window positions.
Changing a state after binding fails export/replay validation. Separate CSV
splits may reuse the training state; for a shared source, supply `split_info`
and original `row_indices` so training fingerprints and split isolation can
also be checked against the source.

## Classification

Confirm sample, timestamp, label, and ordered feature roles before splitting.
Classification requires independent sample IDs and one consistent nonempty
label per sample. Labels use the sorted training mapping, including labels
that look numeric. Validation and test reuse the mapping and reject unknown
classes and samples exceeding the training window. Padding is NaN with a
separate validity mask.

Given a confirmed source, role mapping, and entity split:

```python
from ai4ts.data_protocols.pipeline import prepare_for_pypots

train = prepare_for_pypots(source, task='classification',
    schema_mapping=roles, row_indices=split['rows']['train'], split_info=split,
    preprocessing='standard-v1')
val = prepare_for_pypots(source, task='classification', set_type='val',
    task_state=train['task_state'], row_indices=split['rows']['val'],
    split_info=split, preprocessing='standard-v1')
```

`export_classifications` checks the source, state, and registry bindings
before returning sample IDs, decoded predictions, and named probability columns.

## Forecasting

Forecasting requires historical `window_size`, positive `n_pred_steps`, and a
nonempty ordered `target_features` subset. Inputs and future targets both stay
inside their split. A stride of one produces historical `X` and future `X_pred`;
the future validity mask is `X_pred_valid_mask`, independent of input masking.
Artificial missingness only changes input observations. A training-only scaler
fits all observed training-split rows before forming history/future windows,
as in BenchPOTS' forecast conversion. `transform_targets` selects corresponding
training parameters in target order. Validation/test targets never fit scaling.

New forecasting states order the targets first in `X`. The
[forecasting covariate contract](covariate-roles.md#pypots-forecasting-inputs)
defines the generated model columns, fitted encoding, and supplied future
horizon.

Single streams use chronological/declared observation order. Multiple streams
use disjoint entities and chronological windows inside each entity (new-entity
forecasting). Split row overlaps, partial/interleaved single-stream splits,
unknown targets, label-role targets, wrong horizons and too-short splits fail.
`set_type='full'` additionally produces the last historical window with an
entirely unknown future, explicitly unscored.

`export_forecasts` checks the source, state, and registry bindings and emits
raw-scale long rows retaining origin,
horizon, target order, source positions, predictions and known truth. Only a
verified regular timestamp frequency is extrapolated; otherwise future positions
use observation indices. Neither operation writes over the source CSV.

CSV scaling is per feature across observed training rows. BenchPOTS UCR/UEA
preprocessing scales flattened timestep columns; a CSV conversion of UCR data
with matching original splits is a source/split reference, not an exact replica
of that benchmark preprocessing. This distinction must appear in reports.

## Clustering and anomaly detection

Clustering uses isolated independent samples, the training window/scaler and
optional UTF-8 sample references. References are not training class mappings and
unseen validation reference labels are valid; only classification rejects unknown
classes. Label conflicts within a clustering sample still fail. No label role
means no external truth. `export_clusters` retains string IDs and cluster numbers.

Anomaly preparation splits chronologically before non-overlapping windows,
retains every source row and pads tails with invalid NaNs. Optional anomaly_y is
point-level 0/1 or unknown; input missingness does not change it. Training scaler
and source/state bindings are reused unchanged in validation/test. `export_anomalies`
retains unscorable real points, excludes synthetic padding rows and records score,
fixed threshold, decision, truth and evaluated status. Numerical task targets
remain separate from input missingness.

## Representation data and bound sample vectors

Explicit `task='representation'` confirms independent sample IDs and keeps one
padded window per sample with exact source-row coverage. Training fixes ordered
features, window and scaler. Optional sample labels use the sorted training class
mapping, with the same conflict/unknown-label rejection as classification. Without
a label role the result contains no y or classes; label columns never enter X.
Later splits cannot refit state or truncate a sample exceeding the training window.

`export_representations` checks the representation registry/source/state binding
and finite [n_samples, n_dimensions] vectors before emitting string sample IDs,
valid lengths, coordinates, and optional decoded fixed-probe probabilities.
Per-time pooling and downstream probe fitting belong to the model workflow.
