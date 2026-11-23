"""Numeric references, fixed-origin seams and saved-evidence trust boundaries."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai4ts.data_protocols.analysis import read_analysis_source
from ai4ts.analysis import artifacts as a
from ai4ts.analysis.baselines import compute_baselines
from ai4ts.analysis.diagnostics import calculate, compute_diagnostics, profile_series
from ai4ts.analysis.verify import inspect_analysis, verify_analysis

ROLES = dict(feature_cols=["x"], sample_id_col="id", timestamp_col="time", label_col=None)


@pytest.fixture(autouse=True)
def numeric_reference_ledger(monkeypatch):
    """Count development statistics separately from the formal D/F matrix."""
    path = os.environ.get("ANALYSIS_TEST_STAT_LEDGER")
    if not path:
        return
    from scipy import signal
    from statsmodels.tsa import seasonal, stattools

    def record(method):
        with open(path, "a") as stream:
            stream.write(json.dumps(dict(phase="development_numeric_reference", method=method)) + "\n")

    for module, name in [(stattools, n) for n in ("acf", "pacf", "adfuller", "kpss")] + [(signal, "lombscargle")]:
        original = getattr(module, name)

        def wrapped(*args, _original=original, _name=name, **kwargs):
            record(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapped)
    original_stl = seasonal.STL

    def stl(*args, **kwargs):
        instance = original_stl(*args, **kwargs)

        class Decomposition:
            config = instance.config

            def fit(self):
                record("STL.fit")
                return instance.fit()

        return Decomposition()

    monkeypatch.setattr(seasonal, "STL", stl)


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "source.csv"
    pd.DataFrame(
        dict(
            id="001",
            time=pd.date_range("2024-01-01", periods=240, freq="5min"),
            x=np.tile(np.arange(12, dtype=float), 20),
            note="unchanged",
        )
    ).to_csv(path, index=False)
    return path


def baseline(csv, **kwargs):
    return a.create_plan(
        csv,
        kind="forecast-baselines",
        roles=ROLES,
        targets=["x"],
        split_policy="chronological",
        seasonal_period=12,
        **kwargs,
    )


def compute(csv, plan, ledger=None):
    frame, _, _ = read_analysis_source(csv, ROLES)
    return compute_baselines(frame, plan, ledger)


def test_profiles_keep_denominators_and_runs():
    result = profile_series(np.array([1.0, np.nan, np.nan, 4.0]), [10, 12, 13, 15])
    assert result["missing_rate"] == 0.5 and result["finite_observations"] == 2
    assert result["missing_runs"] == [dict(start_row=12, end_row=13, observations=2)]
    assert profile_series(np.full(3, np.nan), [0, 1, 2])["quantiles"] is None
    assert profile_series(np.ones(2), [0, 1])["constant"]


def test_acf_support_keeps_grid_and_matches_official():
    from statsmodels.tsa.stattools import acf

    values = np.sin(np.arange(40))
    values[[0, 1, 5, 7, 11, 14, 15, 20, 24, 25, 31]] = np.nan
    facts, arrays = calculate("acf", values, {})
    np.testing.assert_allclose(arrays["acf"], acf(values, nlags=10, missing="conservative", fft=True, adjusted=False))
    pairs = [sum(np.isfinite(values[i]) and np.isfinite(values[i + lag]) for i in range(40 - lag)) for lag in range(11)]
    assert facts["pair_counts"] == pairs and facts["confidence_intervals"] is None
    assert not np.allclose(arrays["acf"][:6], acf(values[np.isfinite(values)], nlags=5))
    sparse = np.full(40, np.nan)
    sparse[::2] = np.arange(20)
    facts, _ = calculate("acf", sparse, {})
    assert facts["pair_counts"][1] == 0 and facts["values"][1] is None


def test_pacf_and_tests_match_fixed_official_parameters():
    from statsmodels.tsa import stattools

    values = np.random.default_rng(2024).normal(size=96) + np.arange(96) * 0.01
    _, arrays = calculate("pacf", values, {})
    np.testing.assert_allclose(arrays["pacf"], stattools.pacf(values, nlags=24, method="ywm"))
    adf, _ = calculate("adf", values, {})
    expected = stattools.adfuller(values, regression="c", maxlag=12, autolag="AIC")
    np.testing.assert_allclose([adf["statistic"], adf["pvalue"]], expected[:2])
    assert adf["effective_observations"] == expected[3] and adf["null_hypothesis"] == "unit root"
    kpss, _ = calculate("kpss", values, {})
    expected = stattools.kpss(values, regression="c", nlags=12)
    np.testing.assert_allclose([kpss["statistic"], kpss["pvalue"]], expected[:2])
    assert kpss["null_hypothesis"] == "level stationarity" and kpss["used_lag"] == 12


def test_stl_reference_and_two_cycle_minimum(csv):
    from statsmodels.tsa.seasonal import STL

    frame = pd.read_csv(csv, dtype={"id": str}).iloc[:24]
    frame.to_csv(csv, index=False)
    plan = a.create_plan(csv, roles=ROLES, methods=["stl"], period=12)
    assert plan["applicability"][0]["status"] == "ready"
    values = frame.x.to_numpy()
    facts, arrays = calculate("stl", values, {}, period=12)
    expected = STL(values, period=12, robust=True).fit()
    np.testing.assert_allclose(arrays["trend"], expected.trend)
    np.testing.assert_allclose(sum(arrays.values()), values, atol=1e-12)
    assert facts["parameters"]["robust"] is True


def test_spectral_physical_time_and_reference():
    from scipy.signal import lombscargle

    seconds = np.cumsum(np.random.default_rng(2024).uniform(200, 400, 192))
    seconds -= seconds[0]
    values = np.sin(2 * np.pi * seconds / 3600)
    values[5] = np.nan
    facts, arrays = calculate("lombscargle", values, dict(seconds=seconds), period_range=[1800.0, 7200.0])
    assert abs(facts["candidate_period_seconds"] - 3600) < 60 and facts["significance"] is None
    omega = np.linspace(2 * np.pi / 7200, 2 * np.pi / 1800, 256)
    valid = np.isfinite(values)
    np.testing.assert_allclose(
        arrays["power"], lombscargle(seconds[valid], values[valid], omega, floating_mean=True, normalize=True)
    )
    converted, _ = calculate("lombscargle", values, dict(seconds=seconds / 60 * 60), period_range=[30 * 60, 120 * 60])
    assert converted["candidate_period_seconds"] == facts["candidate_period_seconds"]


@pytest.mark.parametrize("values", [np.ones(40), np.full(40, np.nan), np.arange(8.0)])
def test_inapplicable_methods_make_no_statistical_calls(csv, monkeypatch, values):
    frame = pd.read_csv(csv, dtype={"id": str}).iloc[: len(values)]
    frame["x"] = values
    frame.to_csv(csv, index=False)
    plan = a.create_plan(csv, roles=ROLES, methods=["acf", "pacf", "adf", "kpss", "stl"], period=12)
    import ai4ts.analysis.diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "calculate", lambda *a, **k: pytest.fail("Inapplicable statistical method"))
    frame, _, _ = read_analysis_source(csv, ROLES)
    result = compute_diagnostics(frame, plan)
    assert all(i["status"] == "skipped" for i in result["records"][1:])


def test_fixed_origin_masks_selection_and_test_independence(csv, tmp_path):
    frame = pd.read_csv(csv, dtype={"id": str})
    frame.loc[[4, 33, 172, 200], "x"] = np.nan
    frame.to_csv(csv, index=False)
    plan = baseline(csv, feature_units={"x": "degC"})
    ledger = tmp_path / "calls.jsonl"
    result = compute(csv, plan, ledger)
    records = result["records"]
    assert [i["method"] for i in records if i["selected"]] == ["seasonal-naive"]
    assert records[0]["test"] is None and records[1]["test"]["scored_positions"] == 47
    assert all(i["validation"]["scored_positions"] == 23 and i["unit"] == "degC" for i in records)
    log = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [c["operation"] for c in log] == ["baseline_prepare", "baseline_predict"] * 2 + [
        "selection", "baseline_predict"
    ]
    assert [c["training_rows"] for c in log if c["operation"] == "baseline_prepare"] == [168, 168]
    assert [c["forecast_horizon"] for c in log if c["operation"] == "baseline_predict"] == [
        list(range(1, 25)), list(range(1, 25)), list(range(25, 73))
    ]
    for record in records:
        train = result["arrays"][record["arrays"]["train"]]
        assert len(train) == 168 and np.isnan(train[4])
    frame.loc[192:, "x"] = 1e6
    frame.to_csv(csv, index=False)
    changed = compute(csv, baseline(csv))["records"]
    assert [i["method"] for i in changed if i["selected"]] == ["seasonal-naive"]
    assert changed[1]["validation"] == records[1]["validation"]


def test_tie_favors_last_value_and_unknown_test_has_no_score(csv):
    frame = pd.read_csv(csv, dtype={"id": str})
    frame["x"] = 3.0
    frame.loc[192:, "x"] = np.nan
    frame.to_csv(csv, index=False)
    selected = next(i for i in compute(csv, baseline(csv))["records"] if i["selected"])
    assert selected["method"] == "last-value" and selected["test"]["mae"] is None
    assert selected["test"]["scored_positions"] == 0 and not selected["test"]["evaluated"]


def test_candidate_failure_and_missing_context_do_not_test(csv, monkeypatch, tmp_path):
    from ai4ts.analysis import baselines

    predict = baselines._predict

    def broken(cycle, horizon, ledger_path, **facts):
        if len(cycle) == 12:
            raise ValueError("controlled failure")
        return predict(cycle, horizon, ledger_path, **facts)

    monkeypatch.setattr(baselines, "_predict", broken)
    ledger = tmp_path / "calls.jsonl"
    records = compute(csv, baseline(csv), ledger)["records"]
    assert [r["status"] for r in records] == ["succeeded", "failed"]
    assert all(not r["selected"] and r["test"] is None for r in records)
    calls = ledger.read_text()
    assert sum(json.loads(line)["operation"] == "baseline_predict" for line in calls.splitlines()) == 1
    frame = pd.read_csv(csv, dtype={"id": str})
    frame.loc[167, "x"] = np.nan
    frame.to_csv(csv, index=False)
    assert all(r["status"] == "unsupported" for r in compute(csv, baseline(csv), ledger)["records"])
    assert ledger.read_text() == calls


def test_seasonal_test_horizon_keeps_phase_after_partial_validation_cycle(csv):
    frame = pd.read_csv(csv, dtype={"id": str}).iloc[:90]
    frame.to_csv(csv, index=False)
    result = compute(csv, baseline(csv))
    winner = next(r for r in result["records"] if r["selected"])
    assert winner["method"] == "seasonal-naive"
    # 63 training and 9 validation rows. Test starts at phase 9 of the
    # final training cycle, not at phase 0.
    np.testing.assert_array_equal(
        result["arrays"][winner["arrays"]["test_prediction"]],
        np.r_[np.arange(12), np.arange(6)],
    )


def reseal(root):
    meta = a.read_json(root / "analysis.json")
    a.write_json(
        root / "complete.json",
        dict(
            schema=a.SCHEMA,
            status=meta["status"],
            files={
                p.relative_to(root).as_posix(): dict(sha256=a.sha(p), bytes=p.stat().st_size)
                for p in root.rglob("*")
                if p.is_file() and p.name != "complete.json"
            },
        ),
    )


@pytest.fixture
def saved_bundle(csv, tmp_path):
    plan = baseline(csv)
    ledger = tmp_path / "ledger.jsonl"
    computed = compute(csv, plan, ledger)
    return publish_computed(tmp_path / "baseline-bundle", csv, plan, ledger, computed)


def publish_computed(root, csv, plan, ledger, computed):
    arrays = computed.pop("arrays")
    root.mkdir()
    meta = dict(
        schema=a.SCHEMA,
        status="success" if all(r["status"] == "succeeded" for r in computed["records"]) else "partial",
        kind=plan["request"]["kind"],
        source_sha256=plan["source"]["sha256"],
        plan_sha256=plan["sha256"],
        execution_identity=plan["request"]["execution_identity"],
        production_supported=False,
    )
    (root / "source.csv").write_bytes(csv.read_bytes())
    a.write_json(root / "plan.json", plan)
    a.write_json(root / "analysis.json", meta)
    np.savez_compressed(root / "arrays.npz", **arrays)
    computed["figures"] = a._plots(root, read_analysis_source(csv, ROLES)[0], plan, computed, arrays)
    a.write_json(root / "results.json", computed)
    a._report(root, plan, computed, meta)
    (root / "calls.jsonl").write_bytes(ledger.read_bytes())
    reseal(root)
    return root


def test_complete_diagnostic_evidence_and_import_failure(csv, tmp_path, monkeypatch):
    frame = pd.read_csv(csv, dtype={"id": str}).iloc[:96]
    frame["x"] = np.sin(np.arange(96) * 2 * np.pi / 12) + np.random.default_rng(2024).normal(0, 0.02, 96)
    frame.to_csv(csv, index=False)
    plan = a.create_plan(
        csv,
        roles=ROLES,
        methods=["acf", "pacf", "adf", "kpss", "stl", "lombscargle"],
        period=12,
        period_range=[1800.0, 7200.0],
    )
    source = read_analysis_source(csv, ROLES)[0]
    ledger = tmp_path / "statistics.jsonl"
    computed = compute_diagnostics(source, plan, ledger)
    assert all(r["status"] == "succeeded" for r in computed["records"]), computed
    root = publish_computed(tmp_path / "diagnostics", csv, plan, ledger, computed)
    verified = verify_analysis(root)[-1]
    assert not verified["statistical_recomputation"] and verified["recorded_calls"]["statistical_estimation"] == 6
    import ai4ts.analysis.diagnostics as diagnostics

    def unavailable(*args, **kwargs):
        raise ImportError("controlled backend import failure")

    monkeypatch.setattr(diagnostics, "calculate", unavailable)
    plan = a.create_plan(csv, roles=ROLES, methods=["acf"])
    failed = compute_diagnostics(source, plan, tmp_path / "failed-call.jsonl")
    assert failed["records"][1]["status"] == "failed" and failed["records"][1]["result"] is None
    assert "controlled backend import failure" in failed["records"][1]["reason"]


def test_saved_baseline_closed_form_and_views_are_compute_free(saved_bundle, monkeypatch, tmp_path):
    import socket
    import ai4ts.analysis.diagnostics as diagnostics
    import ai4ts.analysis.baselines as baselines

    monkeypatch.setattr(diagnostics, "calculate", lambda *a, **k: pytest.fail("Statistical computation"))
    monkeypatch.setattr(baselines, "compute_baselines", lambda *a, **k: pytest.fail("Forecasting computation"))
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("Network call"))
    before = {p: a.sha(p) for p in saved_bundle.rglob("*") if p.is_file()}
    assert verify_analysis(saved_bundle)[-1]["closed_form_baseline_verified"]
    for view, selector in [
        ("summary", {}),
        ("feature", dict(feature="x")),
        ("entity", dict(entity="001")),
        ("method", dict(method="seasonal-naive")),
    ]:
        assert inspect_analysis(saved_bundle, view, **selector)["matched"]
    assert not inspect_analysis(saved_bundle, "feature", feature="unknown")["matched"]
    exported = inspect_analysis(saved_bundle, output_dir=tmp_path / "view")
    assert Path(exported["export"], "ready.json").is_file()
    assert before == {p: a.sha(p) for p in saved_bundle.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="outside"):
        inspect_analysis(saved_bundle, output_dir=saved_bundle / "view")


@pytest.mark.parametrize(
    "fault",
    ["origin", "phase", "prediction", "mae", "selection", "denominator", "ledger", "removed-test", "removed-figure"],
)
def test_independent_reader_rejects_rehashed_semantic_tampering(saved_bundle, fault):
    result = a.read_json(saved_bundle / "results.json")
    selected = next(r for r in result["records"] if r["selected"])
    if fault == "origin":
        selected["origin_row"] -= 1
    elif fault == "phase":
        selected["parameters"]["sp"] = 6
    elif fault == "prediction":
        with np.load(saved_bundle / "arrays.npz", allow_pickle=False) as saved:
            arrays = dict(saved)
        arrays[selected["arrays"]["test_prediction"]] += 1
        np.savez_compressed(saved_bundle / "arrays.npz", **arrays)
    elif fault == "mae":
        selected["test"]["mae"] += 1
    elif fault == "selection":
        selected["selected"] = False
    elif fault == "denominator":
        selected["validation"]["scored_positions"] -= 1
    elif fault == "ledger":
        (saved_bundle / "calls.jsonl").write_text("")
    elif fault == "removed-test":
        selected["test"] = None
        selected["arrays"].pop("test_prediction")
    elif fault == "removed-figure":
        selected["figures"] = []
        result["figures"] = [f for f in result["figures"] if f["method"] != selected["method"]]
    a.write_json(saved_bundle / "results.json", result)
    reseal(saved_bundle)
    with pytest.raises(ValueError):
        verify_analysis(saved_bundle)


@pytest.mark.parametrize("fault", ["missing", "escape", "symlink", "schema", "identity", "object-array"])
def test_reader_trust_boundaries(saved_bundle, fault, tmp_path):
    if fault == "missing":
        (saved_bundle / "figures/000.png").unlink()
    elif fault == "escape":
        complete = a.read_json(saved_bundle / "complete.json")
        complete["files"]["../outside"] = dict(bytes=0, sha256="0" * 64)
        a.write_json(saved_bundle / "complete.json", complete)
    elif fault == "symlink":
        (saved_bundle / "report.md").unlink()
        (saved_bundle / "report.md").symlink_to(tmp_path / "absent")
    elif fault in ("schema", "identity"):
        meta = a.read_json(saved_bundle / "analysis.json")
        if fault == "schema":
            meta["schema"] = "unsupported"
        else:
            meta["execution_identity"] = {}
        a.write_json(saved_bundle / "analysis.json", meta)
        reseal(saved_bundle)
    else:
        np.savez_compressed(saved_bundle / "arrays.npz", unsafe=np.array([{}], dtype=object))
        reseal(saved_bundle)
    with pytest.raises((ValueError, OSError)):
        verify_analysis(saved_bundle)


def test_worker_profile_public_api_does_not_overwrite(csv, tmp_path):
    plan = a.create_plan(csv, roles=ROLES)
    before = csv.read_bytes()
    output = tmp_path / "analysis"
    result = a.diagnose(csv, plan, output)
    assert result["status"] == "success", result
    assert result["verification"]["recorded_calls"] == dict(statistical_estimation=0, baseline_prepare=0, baseline_predict=0)
    assert "0 empty cells among 240" in (output / "report.md").read_text()
    with pytest.raises(FileExistsError):
        a.diagnose(csv, plan, output)
    assert csv.read_bytes() == before


def test_bounds_require_explicit_reduction_and_never_clip(csv):
    frame = pd.read_csv(csv, dtype={"id": str})
    pd.concat([frame.assign(id=str(i)) for i in range(9)]).to_csv(csv, index=False)
    plan = a.create_plan(csv, roles=ROLES)
    assert plan["status"] == "blocked" and "8 series" in plan["errors"][0]
    repeated = pd.concat([frame] * 18, ignore_index=True)
    repeated["time"] = pd.date_range("2024-01-01", periods=len(repeated), freq="5min")
    repeated.to_csv(csv, index=False)
    assert a.create_plan(csv, roles=ROLES)["status"] == "blocked"
    reduced = a.create_plan(csv, roles=ROLES, rows=list(range(24)))
    assert reduced["status"] == "ready" and reduced["selected_rows"] == list(range(24))


def test_controlled_timeout_preserves_failure_and_does_not_retry(csv, tmp_path, monkeypatch):
    calls = []

    def timeout(*args, **kwargs):
        calls.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(a.subprocess, "run", timeout)
    output = tmp_path / "timeout"
    result = a.diagnose(csv, a.create_plan(csv, roles=ROLES), output)
    assert result["status"] == "failed" and calls == [60]
    assert (output / "source.csv").read_bytes() == csv.read_bytes()
    assert not (output / "complete.json").exists()
    with pytest.raises(ValueError):
        inspect_analysis(output)


def test_source_change_and_partial_io_leave_evidence(csv, tmp_path, monkeypatch):
    plan = a.create_plan(csv, roles=ROLES)
    before = csv.read_bytes()
    csv.write_bytes(before + b"\n")
    with pytest.raises(ValueError):
        a.diagnose(csv, plan, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()
    csv.write_bytes(before)
    original = Path.write_bytes

    def broken(path, value):
        if path.name == "source.csv" and path.parent.name == "io-failure":
            raise OSError("controlled write failure")
        return original(path, value)

    monkeypatch.setattr(Path, "write_bytes", broken)
    result = a.diagnose(csv, plan, tmp_path / "io-failure")
    assert result["status"] == "failed" and "controlled write failure" in result["error"]
    assert csv.read_bytes() == before
