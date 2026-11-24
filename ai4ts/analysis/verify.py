"""Independent saved-array checks; no statistical or forecasting backend calls."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json
import zipfile
from pathlib import Path

import numpy as np

from .artifacts import SCHEMA, _validate, read_json, sha, write_json
from .capabilities import CATALOG


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(actual, expected, message):
    actual, expected = np.asarray(actual), np.asarray(expected)
    _require(
        actual.shape == expected.shape and np.allclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True),
        message,
    )


def _safe_file(root, relative, limit=32 * 1024 * 1024):
    path = Path(relative)
    _require(
        isinstance(relative, str) and not path.is_absolute() and ".." not in path.parts and "\\" not in relative,
        "Analysis path escapes its bundle",
    )
    result = root / path
    _require(
        all(not p.is_symlink() for p in [result, *result.parents] if p == root or root in p.parents),
        "Linked analysis files are not supported",
    )
    _require(
        result.is_file() and result.stat().st_size <= limit and root.resolve() in result.resolve().parents,
        "Missing or oversized analysis file",
    )
    return result


def _numeric_archive(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        _require(
            len(entries) <= 512 and sum(e.file_size for e in entries) <= 32 * 1024 * 1024,
            "Numeric archive exceeds its uncompressed budget",
        )
        _require(len({e.filename for e in entries}) == len(entries), "Duplicate numeric archive members")
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    _require(
        all(a.dtype.kind in "bifu" and a.ndim <= 2 and a.size <= 65536 for a in result.values()),
        "Unsupported numeric array type or shape",
    )
    return result


def _profile(record, values, rows):
    facts = record["result"]
    finite, missing = np.isfinite(values), np.isnan(values)
    _require(
        facts["observations"] == len(values)
        and facts["finite_observations"] == int(finite.sum())
        and facts["missing_cells"] == int(missing.sum())
        and facts["all_missing"] == (not finite.any()),
        "Profile observation or missing denominator differs",
    )
    _same(facts["missing_rate"], missing.mean(), "Profile missing rate differs")
    _require(facts["constant"] == bool(finite.any() and np.ptp(values[finite]) == 0), "Constant marker differs")
    if finite.any():
        _same(
            list(facts["quantiles"].values()), np.quantile(values[finite], [0, 0.25, 0.5, 0.75, 1]), "Quantiles differ"
        )
    else:
        _require(facts["quantiles"] is None, "All-missing quantiles must be unknown")
    reconstructed = np.zeros(len(values), dtype=bool)
    for run in facts["missing_runs"]:
        start, end = rows.index(run["start_row"]), rows.index(run["end_row"])
        _require(
            end >= start and run["observations"] == end - start + 1 and not reconstructed[start : end + 1].any(),
            "Invalid missing-run mapping",
        )
        reconstructed[start : end + 1] = True
    _require(np.array_equal(reconstructed, missing), "Missing-run coverage differs")


def _diagnostic(record, values, temporal, plan, arrays):
    method, data = record["method"], record["result"]
    numeric = {k: arrays[v] for k, v in record["arrays"].items()}
    if method == "profile":
        _profile(record, values, record["row_indices"])
    elif method in ("acf", "pacf"):
        nlags = min(40, len(values) // 4)
        expected = (
            dict(nlags=nlags, missing="conservative", fft=True, adjusted=False)
            if method == "acf"
            else dict(nlags=nlags, method="ywm")
        )
        _require(data["parameters"] == expected and data["confidence_intervals"] is None, "Lag parameters differ")
        _require(
            data["parameters"]["nlags"] == nlags and numeric[method].shape == (nlags + 1,), "Lag dimensions differ"
        )
        _require(np.isfinite(numeric[method]).all(), "Nonfinite lag values")
        _same(numeric[method][0], 1, "Lag zero differs")
        if method == "acf":
            finite = np.isfinite(values)
            pairs = np.array([int((finite[: len(values) - lag] & finite[lag:]).sum()) for lag in range(nlags + 1)])
            _require(data["parameters"]["missing"] == "conservative", "Missing policy changed")
            _same(data["pair_counts"], pairs, "Lag support differs")
            _same(numeric["pair_counts"], pairs, "Saved lag support differs")
            for value, expected, count in zip(data["values"], numeric[method], pairs):
                _require(
                    value is None if count < 8 else value is not None and np.isclose(value, expected),
                    "Low-support lag semantics differ",
                )
        else:
            _same(data["values"], numeric[method], "PACF record differs from saved array")
    elif method in ("adf", "kpss"):
        expected = (
            dict(regression="c", maxlag=min(12, len(values) // 4 - 1), autolag="AIC")
            if method == "adf"
            else dict(regression="c", nlags=min(12, max(1, len(values) // 8)))
        )
        _require(data["parameters"] == expected, "Statistical test parameters differ")
        _require(np.isfinite(data["statistic"]) and 0 <= data["pvalue"] <= 1, "Invalid recorded test statistic")
        expected_null = "unit root" if method == "adf" else "level stationarity"
        _require(data["null_hypothesis"] == expected_null, "Statistical null hypothesis differs")
        if method == "adf":
            _require(
                data["effective_observations"] == len(values) - data["used_lag"] - 1,
                "ADF effective observations differ",
            )
        else:
            _require(data["effective_observations"] == len(values), "KPSS support differs")
    elif method == "stl":
        _require(data["parameters"]["robust"] is True, "STL robust setting differs")
        _require(
            all(v.shape == values.shape and np.isfinite(v).all() for v in numeric.values()),
            "STL component shapes differ",
        )
        _same(numeric["trend"] + numeric["seasonal"] + numeric["residual"], values, "STL reconstruction differs")
        _require(data["parameters"]["period"] == plan["request"]["period"], "STL period differs")
        error = np.max(np.abs(values - numeric["trend"] - numeric["seasonal"] - numeric["residual"]))
        _same(data["reconstruction_max_error"], error, "STL reconstruction error differs")
    elif method == "lombscargle":
        low, high = plan["request"]["period_range"]
        _require(
            data["parameters"]
            == dict(floating_mean=True, normalize=True, period_range_seconds=[low, high], frequency_points=256),
            "Spectral parameters differ",
        )
        expected = np.linspace(2 * np.pi / high, 2 * np.pi / low, 256)
        _same(numeric["angular_frequency"], expected, "Physical frequency conversion differs")
        _require(numeric["power"].shape == (256,) and np.isfinite(numeric["power"]).all(), "Invalid spectrum")
        _same(data["finite_positions"], np.flatnonzero(np.isfinite(values)), "Spectral source mask differs")
        peak = int(np.argmax(numeric["power"]))
        _require(data["peak_index"] == peak and data["significance"] is None, "Spectral peak or significance differs")
        _same(data["candidate_period_seconds"], 2 * np.pi / expected[peak], "Candidate period differs")


def _baselines(records, source, plan, arrays):
    by_pair = {}
    for item in records:
        by_pair.setdefault((item["entity"], item["feature"]), []).append(item)
    for group in plan["groups"]:
        for feature in plan["request"]["targets"]:
            pair = by_pair[group["entity"], feature]
            split = group["partitions"]
            truth = {name: source.loc[rows, feature].to_numpy(dtype=float) for name, rows in split.items()}
            scores = {}
            for item in pair:
                numeric = {k: arrays[v] for k, v in item["arrays"].items()}
                if item["status"] == "succeeded":
                    _require(
                        item["validation"] is not None and "val_prediction" in numeric,
                        "Successful baseline lacks validation evidence",
                    )
                _require(item["origin_row"] == group["origin_row"], "Forecast origin differs")
                _same(numeric["train"], truth["train"], "Training context differs")
                for role in ("val", "test"):
                    mask = np.isfinite(truth[role])
                    _same(numeric[role + "_truth"], truth[role], "Forecast truth differs")
                    _require(np.array_equal(numeric[role + "_mask"], mask), "Target mask differs")
                    key = role + "_prediction"
                    if key not in numeric:
                        continue
                    horizon = group["validation_fh" if role == "val" else "test_fh"]
                    period = 1 if item["method"] == "last-value" else plan["request"]["seasonal_period"]
                    _require(item["parameters"] == dict(strategy="last", sp=period), "Baseline parameters differ")
                    expected = truth["train"][-period:][(np.asarray(horizon) - 1) % period]
                    _same(numeric[key], expected, "Closed-form baseline forecast differs")
                    facts = item["validation" if role == "val" else "test"]
                    _require(
                        facts["scored_positions"] == int(mask.sum())
                        and facts["excluded_positions"] == int((~mask).sum()),
                        "Forecast metric denominator differs",
                    )
                    if mask.any():
                        score = float(np.abs(expected[mask] - truth[role][mask]).mean())
                        _same(facts["mae"], score, "Forecast MAE differs")
                        if role == "val":
                            scores[item["method"]] = score
                    else:
                        _require(facts["mae"] is None, "No truth cannot have a score")
            if len(scores) == len(pair):
                chosen = min(scores, key=lambda name: (scores[name], name != "last-value"))
                _require([i["method"] for i in pair if i["selected"]] == [chosen], "Validation selection differs")
                _require(
                    all("test_prediction" not in i["arrays"] for i in pair if i["method"] != chosen),
                    "Unselected candidate was tested",
                )
                winner = next(i for i in pair if i["method"] == chosen)
                _require(
                    winner["status"] != "succeeded"
                    or (winner.get("test") is not None and "test_prediction" in winner["arrays"]),
                    "Successful selected baseline lacks final test evidence",
                )
            else:
                _require(
                    not any(i["selected"] or i.get("test") for i in pair), "Incomplete comparison cannot select or test"
                )


def _ledger(ledger, records, plan):
    """Check per-method call evidence and ordering, including failed attempts."""
    diagnostic = plan["request"]["kind"] == "diagnostics"
    conditions = {(r["entity"], r["feature"], r["method"]): r for r in plan["applicability"]}
    by_key = {(r["entity"], r["feature"], r["method"]): r for r in records}
    seen = {}
    for index, call in enumerate(ledger):
        key = (call["entity"], call["feature"], call["method"])
        _require(
            key in by_key and key in conditions and conditions[key]["status"] == "ready", "Call outside applicable plan"
        )
        operation = call["operation"]
        signature = (key, operation, call.get("split"))
        _require(signature not in seen, "Repeated analysis call; automatic retries are unsupported")
        seen[signature] = index
        if diagnostic:
            _require(operation == "statistical_estimation", "Diagnostic call kind differs")
            continue
        item = by_key[key]
        group = next(g for g in plan["groups"] if g["entity"] == item["entity"])
        if operation == "baseline_prepare":
            _require(call["training_rows"] == len(group["partitions"]["train"]), "Baseline training cutoff differs")
        elif operation == "baseline_predict":
            role = call.get("split")
            _require(role in ("val", "test") and (key, "baseline_prepare", None) in seen, "Predict without prepared training cycle")
            _require(
                call["forecast_horizon"] == group["validation_fh" if role == "val" else "test_fh"],
                "Recorded forecast horizon differs",
            )
            if role == "test":
                _require(item["selected"] and (key, "selection", None) in seen, "Test before validation selection")
        elif operation == "selection":
            pair = [r for r in records if (r["entity"], r["feature"]) == key[:2]]
            _require(
                item["selected"] and all((key[:2] + (r["method"],), "baseline_predict", "val") in seen for r in pair),
                "Selection before candidate validation",
            )
            _require(
                call["validation"] == {r["method"]: r["validation"] for r in pair}, "Selection ledger evidence differs"
            )
        else:
            raise ValueError("Forecasting call kind differs")
    for key, item in by_key.items():
        if diagnostic:
            expected = item["method"] != "profile" and item["status"] == "succeeded"
            _require(not expected or (key, "statistical_estimation", None) in seen, "Missing statistical call ledger")
        else:
            for role, field in (("val", "validation"), ("test", "test")):
                _require(not item.get(field) or (key, "baseline_predict", role) in seen, "Missing forecast call ledger")


def verify_analysis(bundle):
    """Check saved evidence without invoking statistical/forecasting estimators."""
    from ai4ts.data_protocols.analysis import read_analysis_source

    root = Path(bundle).absolute()
    _require(root.is_dir() and not root.is_symlink(), "Analysis bundle must be a real directory")
    complete = read_json(_safe_file(root, "complete.json"))
    _require(
        complete.get("schema") == SCHEMA and complete.get("status") in ("success", "partial"),
        "Unsupported or incomplete analysis",
    )
    files = complete["files"]
    _require(isinstance(files, dict) and len(files) <= 256, "File manifest exceeds bounds")
    _require(
        {"analysis.json", "plan.json", "results.json", "source.csv", "arrays.npz", "calls.jsonl", "report.md"}.issubset(
            files
        ),
        "Required analysis files are not bound by the manifest",
    )
    for relative, facts in files.items():
        path = _safe_file(root, relative)
        _require(
            path.stat().st_size == facts["bytes"] and sha(path) == facts["sha256"],
            "Analysis file integrity differs: " + relative,
        )
    meta, plan, result = [read_json(_safe_file(root, name)) for name in ("analysis.json", "plan.json", "results.json")]
    _require(meta["schema"] == SCHEMA and meta["status"] == complete["status"], "Analysis outcome differs")
    _validate(plan, _safe_file(root, "source.csv"))
    _require(meta["kind"] == plan["request"]["kind"], "Analysis kind differs")
    _require(
        meta["source_sha256"] == plan["source"]["sha256"]
        and meta["plan_sha256"] == plan["sha256"]
        and meta["execution_identity"] == plan["request"]["execution_identity"],
        "Analysis binding differs",
    )
    source, _, _ = read_analysis_source(root / "source.csv", plan["roles"])
    arrays = _numeric_archive(_safe_file(root, "arrays.npz"))
    records = result["records"]
    _require(isinstance(records, list) and len(records) <= 64, "Result count exceeds bounds")
    units = (plan["feature_semantics"] or {}).get("units", {})
    diagnostic = plan["request"]["kind"] == "diagnostics"
    methods = (
        ["profile", *(plan["request"]["methods"] or [])]
        if diagnostic
        else (plan["request"]["candidates"] or ["last-value", "seasonal-naive"])
    )
    features = plan["roles"]["feature_cols"] if diagnostic else plan["request"]["targets"]
    expected_keys = {(g["entity"], f, m) for g in plan["groups"] for f in features for m in methods}
    _require(
        len(records) == len(expected_keys)
        and {(r["entity"], r["feature"], r["method"]) for r in records} == expected_keys,
        "Method, entity or feature coverage differs",
    )
    for item in records:
        group = next(g for g in plan["groups"] if g["entity"] == item["entity"])
        rows = group["time"]["rows"] if diagnostic else group["partitions"]["train"]
        values = source.loc[rows, item["feature"]].to_numpy(dtype=float)
        _require(
            item["row_indices"] == rows
            and item["unit"] == units.get(item["feature"], "unknown")
            and item["finite_observations"] == int(np.isfinite(values).sum()),
            "Record source, unit or denominator differs",
        )
        _require(
            item["method"] in CATALOG and item["backend"] == CATALOG[item["method"]][0], "Backend attribution differs"
        )
        package = plan["request"]["execution_identity"]["packages"].get(item["backend"])
        if package is not None:
            _require(item.get("version") == package["version"], "Backend version attribution differs")
        _require(
            item["status"] in ("succeeded", "skipped", "unsupported", "unavailable", "failed"), "Unknown method state"
        )
        if diagnostic:
            condition = next(
                (
                    r
                    for r in plan["applicability"]
                    if (r["entity"], r["feature"], r["method"]) == (item["entity"], item["feature"], item["method"])
                ),
                None,
            )
            if condition:
                allowed = (
                    ("succeeded", "failed", "unavailable") if condition["status"] == "ready" else (condition["status"],)
                )
                _require(item["status"] in allowed, "Diagnostic applicability differs")
        _require(all(key in arrays for key in item["arrays"].values()), "Missing numeric evidence")
        expected_figures = 1 if item["status"] == "succeeded" and item["method"] not in ("adf", "kpss") else 0
        _require(len(item.get("figures", [])) == expected_figures, "Required figure evidence is missing")
        if diagnostic and item["status"] == "succeeded":
            _diagnostic(item, values, group["time"], plan, arrays)
    if not diagnostic:
        _baselines(records, source, plan, arrays)
    calls = _safe_file(root, "calls.jsonl", 1024 * 1024).read_text().splitlines()
    ledger = [json.loads(line) for line in calls]
    _require(
        len(ledger) <= 128
        and all(
            i["operation"] in {"statistical_estimation", "baseline_prepare", "baseline_predict", "selection"} for i in ledger
        ),
        "Unexpected analysis computation ledger",
    )
    counts = {
        name: sum(i["operation"] == name for i in ledger)
        for name in ("statistical_estimation", "baseline_prepare", "baseline_predict")
    }
    _require(
        counts["baseline_prepare"] <= plan["budget"]["baseline_prepare"]
        and counts["baseline_predict"] <= plan["budget"]["baseline_predict"],
        "Baseline call budget exceeded",
    )
    if diagnostic:
        _require(counts["baseline_prepare"] == counts["baseline_predict"] == 0, "Diagnostics ran forecasting")
    _ledger(ledger, records, plan)
    for figure in result["figures"]:
        _require(figure["path"] in files, "Figure is not bound by the manifest")
        path = _safe_file(root, figure["path"], 8 * 1024 * 1024)
        _require(
            path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") and all(k in arrays for k in figure["arrays"]),
            "Invalid figure evidence binding",
        )
        matching = [
            r
            for r in records
            if (r["entity"], r["feature"], r["method"]) == (figure["entity"], figure["feature"], figure["method"])
        ]
        _require(
            len(matching) == 1
            and figure["source_rows"] == matching[0]["row_indices"]
            and figure["path"] in matching[0].get("figures", []),
            "Figure source mapping differs",
        )
    paths = [figure["path"] for figure in result["figures"]]
    _require(
        len(paths) == len(set(paths)) and set(paths) == {p for r in records for p in r.get("figures", [])},
        "Figure coverage differs",
    )
    expected_status = "success" if all(i["status"] == "succeeded" for i in records) else "partial"
    _require(meta["status"] == expected_status, "Overall method outcome differs")
    return (
        meta,
        plan,
        result,
        dict(
            status="passed",
            verification_scope="integrity_and_source_semantics",
            closed_form_baseline_verified=not diagnostic,
            statistical_recomputation=False,
            recorded_calls=counts,
        ),
    )


def inspect_analysis(bundle, view="summary", feature=None, entity=None, method=None, output_dir=None):
    """Read verified saved facts without rerunning any statistical backend.

    Parameters
    ----------
    bundle : str or Path
        Complete compatible analysis directory.
    view : str
        summary, feature, entity, or method.
    feature, entity, method : str or None
        Exact selector for the chosen view.
    output_dir : str or Path or None
        Optional new directory for a local view export.

    Returns
    -------
    dict
        Recorded facts, links and the explicitly limited verification scope.
    """
    _require(view in ("summary", "feature", "entity", "method"), "Unsupported analysis view")
    selector = dict(feature=feature, entity=entity, method=method)
    if view != "summary":
        _require(isinstance(selector[view], str) and bool(selector[view]), "An exact view selector is required")
    root = Path(bundle).absolute()
    meta, plan, result, verified = verify_analysis(root)
    records = [i for i in result["records"] if view == "summary" or i[view] == selector[view]]
    facts = dict(
        status="success",
        result_status=meta["status"],
        analysis=str(root),
        kind=meta["kind"],
        view=view,
        matched=bool(records),
        verification=verified,
        report=str(root / "report.md"),
        records=[{k: v for k, v in i.items() if k not in ("row_indices", "arrays")} for i in records],
        figures=[str(root / p) for i in records for p in i.get("figures", [])],
        production_supported=False,
    )
    if not records:
        facts["missing"] = "No saved evidence matches this selector; a new calculation needs a new plan"
    if output_dir is not None:
        output = Path(output_dir).absolute()
        _require(
            root.resolve() not in output.resolve().parents and not output.exists() and not output.is_symlink(),
            "Choose a new view directory outside the original analysis",
        )
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "view.json", facts)
        (output / "view.md").write_text(
            f"# Saved analysis view\n\n[Original report]({root / 'report.md'})\n\n"
            "Verification: integrity/source semantics; no statistical recomputation.\n",
            encoding="utf-8",
        )
        write_json(output / "ready.json", dict(status="ready", source_analysis=str(root), view=view))
        facts["export"] = str(output)
    return facts
