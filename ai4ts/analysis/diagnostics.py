"""Fixed statistical adapters operating on original two-dimensional series."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import json
import time
import warnings

import numpy as np

from .capabilities import CATALOG


def record_call(path, operation, **facts):
    """Persist a call before execution so interrupted work remains countable."""
    if path is not None:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(dict(operation=operation, monotonic=time.monotonic(), **facts), allow_nan=False) + "\n"
            )


def missing_runs(values, rows):
    missing = np.isnan(values)
    edges = np.diff(np.r_[False, missing, False].astype(int))
    return [
        dict(start_row=int(rows[start]), end_row=int(rows[end - 1]), observations=int(end - start))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    ]


def profile_series(values, rows):
    """Describe finite observations and empty cells without imputation."""
    finite = values[np.isfinite(values)]
    return dict(
        observations=len(values),
        finite_observations=len(finite),
        missing_cells=int(np.isnan(values).sum()),
        missing_rate=float(np.isnan(values).mean()),
        all_missing=not len(finite),
        constant=bool(len(finite) and np.ptp(finite) == 0),
        quantiles=(
            dict(zip(("min", "q25", "median", "q75", "max"), np.quantile(finite, [0, 0.25, 0.5, 0.75, 1]).tolist()))
            if len(finite)
            else None
        ),
        missing_runs=missing_runs(values, rows),
    )


def calculate(method, values, temporal, period=None, period_range=None):
    """Calculate one applicable fixed method, returning facts and raw arrays."""
    n = len(values)
    nlags = min(40, n // 4)
    if method in ("acf", "pacf", "adf", "kpss"):
        from statsmodels.tsa import stattools
    if method == "acf":
        params = dict(nlags=nlags, missing="conservative", fft=True, adjusted=False)
        numbers = np.asarray(stattools.acf(values, **params))
        mask = np.isfinite(values)
        support = np.array([int((mask[: n - lag] & mask[lag:]).sum()) for lag in range(nlags + 1)])
        return dict(
            parameters=params,
            values=[float(v) if count >= 8 else None for v, count in zip(numbers, support)],
            pair_counts=support.tolist(),
            confidence_intervals=None,
        ), dict(acf=numbers, pair_counts=support)
    if method == "pacf":
        params = dict(nlags=nlags, method="ywm")
        numbers = np.asarray(stattools.pacf(values, **params))
        return dict(parameters=params, values=numbers.tolist(), confidence_intervals=None), dict(pacf=numbers)
    if method == "adf":
        params = dict(regression="c", maxlag=min(12, n // 4 - 1), autolag="AIC")
        stat, pvalue, lag, nobs, critical, icbest = stattools.adfuller(values, **params)
        return (
            dict(
                parameters=params,
                statistic=float(stat),
                pvalue=float(pvalue),
                used_lag=int(lag),
                effective_observations=int(nobs),
                critical_values=critical,
                information_criterion=float(icbest),
                null_hypothesis="unit root",
                interpretation="Evidence only; non-rejection does not prove the null",
            ),
            {},
        )
    if method == "kpss":
        params = dict(regression="c", nlags=min(12, max(1, n // 8)))
        stat, pvalue, lag, critical = stattools.kpss(values, **params)
        return (
            dict(
                parameters=params,
                statistic=float(stat),
                pvalue=float(pvalue),
                used_lag=int(lag),
                effective_observations=n,
                critical_values=critical,
                null_hypothesis="level stationarity",
                interpretation="Retain boundary warnings; conflicting evidence remains uncertain",
            ),
            {},
        )
    if method == "stl":
        from statsmodels.tsa.seasonal import STL

        estimator = STL(values, period=period, robust=True)
        result = estimator.fit()
        arrays = dict(
            trend=np.asarray(result.trend), seasonal=np.asarray(result.seasonal), residual=np.asarray(result.resid)
        )
        error = np.max(np.abs(values - sum(arrays.values())))
        return (
            dict(
                parameters=estimator.config,
                reconstruction_max_error=float(error),
                interpretation="Descriptive decomposition, not causal attribution",
            ),
            arrays,
        )
    if method == "lombscargle":
        from scipy.signal import lombscargle

        mask = np.isfinite(values)
        seconds = np.asarray(temporal["seconds"])[mask]
        omega = np.linspace(2 * np.pi / period_range[1], 2 * np.pi / period_range[0], 256)
        params = dict(floating_mean=True, normalize=True)
        power = lombscargle(seconds, values[mask], omega, **params)
        peak = int(np.argmax(power))
        return dict(
            parameters={**params, "period_range_seconds": period_range, "frequency_points": 256},
            finite_positions=np.flatnonzero(mask).tolist(),
            peak_index=peak,
            candidate_period_seconds=float(2 * np.pi / omega[peak]),
            significance=None,
            interpretation="Candidate within the declared search range; significance not estimated",
        ), dict(angular_frequency=omega, power=power)
    raise ValueError(f"Unsupported method: {method}")


def compute_diagnostics(source, plan, ledger_path=None):
    """Execute only the methods and original rows declared by a ready plan."""
    records, arrays = [], {}
    applicability = {(r["entity"], r["feature"], r["method"]): r for r in plan["applicability"]}
    units = (plan["feature_semantics"] or {}).get("units", {})
    for group in plan["groups"]:
        rows = group["time"]["rows"]
        for feature in plan["roles"]["feature_cols"]:
            values = source.loc[rows, feature].to_numpy(dtype=float)
            common = dict(
                entity=group["entity"],
                feature=feature,
                unit=units.get(feature, "unknown"),
                row_indices=rows,
                finite_observations=int(np.isfinite(values).sum()),
                scope=plan["scope"],
            )
            records.append(
                dict(
                    **common,
                    method="profile",
                    backend="numpy",
                    version=importlib.metadata.version("numpy"),
                    status="succeeded",
                    result=profile_series(values, rows),
                    arrays={},
                )
            )
            for method in plan["request"]["methods"] or []:
                status = applicability[group["entity"], feature, method]
                backend = CATALOG[method][0]
                item = dict(
                    **common,
                    method=method,
                    backend=backend,
                    reference=CATALOG[method][-1],
                    status=status["status"],
                    reason=status["reason"],
                    arrays={},
                    result=None,
                    warnings=[],
                )
                try:
                    item["version"] = importlib.metadata.version(backend)
                    if item["status"] == "ready":
                        record_call(
                            ledger_path,
                            "statistical_estimation",
                            entity=group["entity"],
                            feature=feature,
                            method=method,
                        )
                        with warnings.catch_warnings(record=True) as captured:
                            warnings.simplefilter("always")
                            result, numeric = calculate(
                                method,
                                values,
                                group["time"],
                                plan["request"]["period"],
                                plan["request"]["period_range"],
                            )
                        item["warnings"] = [str(w.message)[:2000] for w in captured]
                        item["result"] = result
                        for name, array in numeric.items():
                            if not np.isfinite(array).all():
                                raise ValueError("Backend produced nonfinite statistical output")
                            key = f"r{len(records)}_{name}"
                            arrays[key] = array
                            item["arrays"][name] = key
                        # Strict JSON below also rejects NaN p values/statistics.
                        json.dumps(result, allow_nan=False)
                        item["status"] = "succeeded"
                except importlib.metadata.PackageNotFoundError:
                    item.update(status="unavailable", reason=f"Install the optional {backend} backend")
                except Exception as error:
                    item.update(status="failed", reason=f"{type(error).__name__}: {error}", result=None)
                records.append(item)
    return dict(records=records, arrays=arrays)
