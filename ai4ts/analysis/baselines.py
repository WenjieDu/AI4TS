"""Fixed-origin NumPy baselines; no updating, tuning or neural models."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from .capabilities import CATALOG
from .diagnostics import record_call


def _predict(cycle, horizon, ledger_path, **facts):
    record_call(ledger_path, "baseline_predict", forecast_horizon=horizon, **facts)
    result = np.take(cycle, np.asarray(horizon) - 1, mode="wrap")
    if result.shape != (len(horizon),) or not np.isfinite(result).all():
        raise ValueError("Baseline forecast shape or finite values differ from the plan")
    return result


def compute_baselines(source, plan, ledger_path=None):
    """Compare fixed candidates on validation and test only the selected one.

    Parameters
    ----------
    source : pandas.DataFrame
        Validated original records.
    plan : dict
        Ready fixed-origin analysis plan.
    ledger_path : str or Path or None
        Append-only operation ledger.

    Returns
    -------
    dict
        Evidence records and numeric arrays; no serialized estimators.
    """
    records, arrays = [], {}
    applicability = {(r["entity"], r["feature"], r["method"]): r for r in plan["applicability"]}
    units = (plan["feature_semantics"] or {}).get("units", {})
    names = plan["request"]["candidates"] or ["last-value", "seasonal-naive"]
    for group in plan["groups"]:
        partitions = group["partitions"]
        for feature in plan["request"]["targets"]:
            train, val, test = [
                source.loc[partitions[s], feature].to_numpy(dtype=float) for s in ("train", "val", "test")
            ]
            mask, test_mask = np.isfinite(val), np.isfinite(test)
            pair, cycles = [], {}
            for name in names:
                condition = applicability[group["entity"], feature, name]
                item = dict(
                    entity=group["entity"],
                    feature=feature,
                    method=name,
                    backend="numpy",
                    version=np.__version__,
                    reference=CATALOG[name][-1],
                    unit=units.get(feature, "unknown"),
                    status=condition["status"],
                    reason=condition["reason"],
                    origin_row=group["origin_row"],
                    row_indices=partitions["train"],
                    finite_observations=int(np.isfinite(train).sum()),
                    parameters=dict(
                        strategy="last", sp=1 if name == "last-value" else plan["request"]["seasonal_period"]
                    ),
                    validation=None,
                    test=None,
                    selected=False,
                    arrays={},
                    warnings=[],
                )
                index = len(records) + len(pair)
                for key, values in dict(
                    train=train, val_truth=val, val_mask=mask, test_truth=test, test_mask=test_mask
                ).items():
                    array_key = f"r{index}_{key}"
                    arrays[array_key] = values
                    item["arrays"][key] = array_key
                if not mask.any():
                    item.update(status="unsupported", reason="No finite validation truth; selection unavailable")
                elif condition["status"] == "ready":
                    try:
                        record_call(
                            ledger_path,
                            "baseline_prepare",
                            entity=group["entity"],
                            feature=feature,
                            method=name,
                            training_rows=len(train),
                        )
                        period = item["parameters"]["sp"]
                        cycle = train[-period:].copy()
                        predicted = _predict(
                            cycle,
                            group["validation_fh"],
                            ledger_path,
                            entity=group["entity"],
                            feature=feature,
                            method=name,
                            split="val",
                        )
                        key = f"r{index}_val_prediction"
                        arrays[key] = predicted
                        item["arrays"]["val_prediction"] = key
                        item.update(
                            status="succeeded",
                            validation=dict(
                                mae=float(np.abs(predicted[mask] - val[mask]).mean()),
                                scored_positions=int(mask.sum()),
                                excluded_positions=int((~mask).sum()),
                            ),
                        )
                        cycles[name] = cycle
                    except Exception as error:
                        item.update(status="failed", reason=f"{type(error).__name__}: {error}")
                pair.append(item)
            if all(r["status"] == "succeeded" for r in pair):
                selected = min(pair, key=lambda r: (r["validation"]["mae"], r["method"] != "last-value"))
                selected["selected"] = True
                selected["selection_basis"] = "validation raw MAE; exact tie favors last-value"
                record_call(
                    ledger_path,
                    "selection",
                    entity=group["entity"],
                    feature=feature,
                    method=selected["method"],
                    validation={r["method"]: r["validation"] for r in pair},
                )
                try:
                    predicted = _predict(
                        cycles[selected["method"]],
                        group["test_fh"],
                        ledger_path,
                        entity=group["entity"],
                        feature=feature,
                        method=selected["method"],
                        split="test",
                    )
                    key = f"r{len(records) + pair.index(selected)}_test_prediction"
                    arrays[key] = predicted
                    selected["arrays"]["test_prediction"] = key
                    selected["test"] = dict(
                        mae=float(np.abs(predicted[test_mask] - test[test_mask]).mean()) if test_mask.any() else None,
                        scored_positions=int(test_mask.sum()),
                        excluded_positions=int((~test_mask).sum()),
                        evaluated=bool(test_mask.any()),
                    )
                except Exception as error:
                    selected.update(status="failed", reason=f"{type(error).__name__}: {error}")
            else:
                for item in pair:
                    item["selection_basis"] = "incomplete comparison; no final test or winner"
            records.extend(pair)
    return dict(records=records, arrays=arrays)
