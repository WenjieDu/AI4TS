"""Forecast covariate evidence, time-grid validation and raw TSFM export."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from ai4ts.cli import cli
from ai4ts.data_protocols import WindowRegistry, prepare_for_pypots
from ai4ts.data_protocols.preprocessing import state_sha256
from ai4ts.data_protocols.registry import file_sha256
from ai4ts.data_protocols.task_data import validate_task_state

OPTIONS = {
    "sample_id_col": "id",
    "timestamp_col": "time",
    "feature_columns": ["weather", "sales", "promo", "region"],
    "target_features": ["sales"],
    "n_pred_steps": 2,
}
EVIDENCE = {"origin": "source", "reference": "Published schedule, rev. 4"}
COVARIATES = {
    "promo": {"role": "known", "kind": "numeric", "evidence": EVIDENCE},
    "region": {"role": "static", "kind": "categorical", "evidence": {"origin": "user-confirmed", "reference": "Map"}},
}
DEFAULT = {"role": "past", "kind": "numeric", "evidence": {"origin": "default"}}


@pytest.fixture
def frames():
    history = pd.DataFrame(
        [
            ("002", "2024-01-01T05:00:00+01:00", 9.5, 320, 1, "NA", "hidden"),
            ("001", "2024-01-01 00:00", np.nan, 120, 0, "001", "hidden"),
            ("002", "2024-01-01 03:00Z", 8.5, 230, 1, "NA", "hidden"),
            ("001", "2024-01-01 01:00Z", 2.5, np.nan, 0, "", "hidden"),
        ],
        columns=["id", "time", *OPTIONS["feature_columns"], "unselected"],
    )
    future = pd.DataFrame(
        [
            ("002", "2024-01-01 06:00Z", np.nan),
            ("001", "2024-01-01 02:00Z", 1),
            ("002", "2024-01-01 05:00Z", 0),
            ("001", "2024-01-01 04:00+01:00", 1),
        ],
        columns=["id", "time", "promo"],
    )
    return history, future


def validate(frames, **overrides):
    from ai4ts.data_protocols import validate_future_covariates

    options = dict(OPTIONS, covariates=COVARIATES)
    options.update(overrides)
    return validate_future_covariates(*frames, **options)


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ([], "mapping"),
        ({"sales": {"role": "past", "kind": "numeric"}}, "non-target"),
        ({"id": {"role": "past", "kind": "numeric"}}, "non-target"),
        ({"time": {"role": "past", "kind": "numeric"}}, "non-target"),
        ({"absent": {"role": "past", "kind": "numeric"}}, "non-target"),
        ({"unselected": {"role": "past", "kind": "numeric"}}, "non-target"),
        ({"promo": "known"}, "role and kind"),
        ({"promo": {"role": "future", "kind": "numeric"}}, "role"),
        ({"promo": {"role": "past", "kind": "float"}}, "kind"),
        ({"promo": {"role": "known", "kind": "numeric"}}, "evidence"),
        ({"promo": {"role": "static", "kind": "numeric"}}, "evidence"),
        ({"promo": {"role": "past", "kind": "numeric", "other": 1}}, "fields"),
        ({"promo": {"role": "past", "kind": "numeric", "evidence": {"origin": "default"}}}, "evidence"),
        ({"promo": {"role": "known", "kind": "numeric", "evidence": {"origin": "source"}}}, "evidence"),
        ({"promo": {"role": "known", "kind": "numeric", "evidence": dict(EVIDENCE, reference="")}}, "reference"),
        ({"promo": {"role": "known", "kind": "numeric", "evidence": dict(EVIDENCE, reference="a\nb")}}, "reference"),
        ({"promo": {"role": "known", "kind": "numeric", "evidence": dict(EVIDENCE, reference="x" * 257)}}, "reference"),
    ],
)
def test_declaration_validation(frames, declaration, message):
    with pytest.raises(ValueError, match=message):
        validate(frames, covariates=declaration)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda f: None, "required"),
        (lambda f: f.drop(columns="id"), "Missing"),
        (lambda f: f.drop(columns="time"), "Missing"),
        (lambda f: f.drop(columns="promo"), "Missing"),
        (lambda f: f.assign(sales=1), "target"),
        (lambda f: f.assign(weather=1), "known"),
        (lambda f: f.assign(region="a"), "known"),
        (lambda f: f.assign(absent=1), "absent from history"),
        (lambda f: f.assign(unselected=1), "known"),
        (lambda f: pd.concat([f, f[["promo"]]], axis=1), "unique"),
        (lambda f: f.assign(id=f.id.replace("002", "003")), "same entities"),
        (lambda f: f[f.id == "001"], "same entities"),
        (lambda f: f.assign(id=np.nan), "identifiers"),
        (lambda f: f.iloc[:-1], "exactly 2"),
        (lambda f: f.assign(time=f.time.mask(f.index == 0, f.time.iloc[2])), "Duplicate"),
        (lambda f: f.assign(time=f.time.mask(f.index == 0, "2024-01-01 07:00Z")), "continue"),
        (lambda f: f.assign(time=f.time.mask(f.index == 2, "2024-01-01 04:00Z")), "continue"),
        (lambda f: f.assign(time="unparseable"), "timestamp"),
        (lambda f: f.assign(time="1704067200"), "epoch"),
        (lambda f: f.assign(promo="wrong"), "numeric"),
        (lambda f: f.assign(promo=float("inf")), "infinite"),
        (lambda f: f.assign(promo=1 + 2j), "real numeric"),
        (lambda f: f.assign(promo=pd.Timestamp("2024-01-01")), "real numeric"),
        (lambda f: f.assign(promo=pd.Timedelta("1h")), "real numeric"),
        (lambda f: f.assign(promo=True), "real numeric"),
    ],
)
def test_future_validation(frames, edit, message):
    history, future = frames
    with pytest.raises(ValueError, match=message):
        validate((history, edit(future)))


@pytest.mark.parametrize(
    ("edit", "options", "message"),
    [
        (lambda h: h.iloc[:0], {}, "nonempty"),
        (lambda h: h.assign(id=""), {}, "identifiers"),
        (lambda h: h.assign(time="2024-01-01 00:00Z"), {}, "Duplicate"),
        (lambda h: h.assign(time=h.time.mask(h.index == 0, "2024-01-01 05:00Z")), {}, "cadence"),
        (lambda h: h.iloc[:2], {}, "frequency"),
        (lambda h: h, {"frequency": "30min"}, "cadence"),
        (lambda h: h, {"frequency": "0min"}, "positive"),
        (lambda h: h, {"frequency": "nonsense"}, "frequency"),
        (lambda h: h, {"n_pred_steps": False}, "positive"),
        (lambda h: h, {"target_features": []}, "target"),
        (lambda h: h, {"target_features": ["absent"]}, "target"),
        (lambda h: h, {"feature_columns": ["sales", "sales"]}, "unique"),
        (lambda h: h, {"sample_id_col": "time"}, "distinct"),
        (lambda h: h.drop(columns="weather"), {}, "Missing"),
        (lambda h: h.assign(sales="wrong"), {}, "numeric"),
    ],
)
def test_history_validation(frames, edit, options, message):
    history, future = frames
    with pytest.raises(ValueError, match=message):
        validate((edit(history), future), **options)


def test_supplied_future_normalized_without_mutating_inputs(frames):
    history, future = frames
    before = history.copy(deep=True), future.copy(deep=True)
    result = validate((history, future[["promo", "time", "id"]]), frequency="60min")
    assert list(result.columns) == ["id", "time", "promo"]
    assert result.id.tolist() == ["001", "001", "002", "002"]
    assert result.time.dt.hour.tolist() == [2, 3, 5, 6]
    assert str(result.time.dt.tz) == "UTC"
    assert pd.isna(result.promo.iloc[-1])
    for actual, expected in zip(frames, before):
        pd.testing.assert_frame_equal(actual, expected)


def test_named_indexes_do_not_interfere_with_entity_columns(frames):
    history, future = frames
    expected = validate(frames)
    indexed = history.set_index("id", drop=False), future.set_index("time", drop=False)
    pd.testing.assert_frame_equal(validate(indexed), expected)
    renamed = history.rename(columns={"id": "self"}), future.rename(columns={"id": "self"})
    pd.testing.assert_frame_equal(validate(renamed, sample_id_col="self"), expected.rename(columns={"id": "self"}))


@pytest.mark.parametrize("timestamps", [["01:00", "02:00"], ["now", "today"]])
def test_history_requires_absolute_timestamps(timestamps):
    from ai4ts.data_protocols import validate_future_covariates

    history = pd.DataFrame({"id": ["a", "a"], "time": timestamps, "sales": [1, 2]})
    with pytest.raises(ValueError, match="absolute"):
        validate_future_covariates(history, None, **dict(OPTIONS, feature_columns=["sales"], frequency="1h"))


def test_empty_known_set_builds_grid_without_promoting_other_covariates(frames):
    history, _ = frames
    future = validate((history, None), covariates={"region": COVARIATES["region"]})
    assert list(future.columns) == ["id", "time"]
    assert future.time.dt.hour.tolist() == [2, 3, 5, 6]
    supplied = validate((history, future), covariates={"region": COVARIATES["region"]})
    pd.testing.assert_frame_equal(supplied, future)
    singleton = validate((history.iloc[:2], None), covariates={"region": COVARIATES["region"]}, frequency="1h")
    assert singleton.time.dt.hour.tolist() == [1, 2, 5, 6]
    with pytest.raises(ValueError, match="cell limit"):
        validate((history, None), covariates={"region": COVARIATES["region"]}, n_pred_steps=10_000_000)


def test_export_raw_frames_and_manifest(frames, tmp_path):
    from ai4ts.data_protocols import export_tsfm_frames

    history, future = frames
    source, ahead, output = (tmp_path / name for name in ("history.csv", "ahead.csv", "output"))
    history.to_csv(source, index=False)
    future.to_csv(ahead, index=False)
    manifest = export_tsfm_frames(source, output, future_csv=ahead, covariates=COVARIATES, **OPTIONS)
    assert manifest == json.loads((output / "manifest.json").read_text())
    assert manifest["schema"] == "ai4ts-tsfm-frames-v1" and manifest["status"] == "success"
    assert manifest["covariates"] == dict(COVARIATES, weather=DEFAULT)
    assert manifest["covariates_sha256"] == state_sha256(manifest["covariates"])
    assert manifest["n_pred_steps"] == 2 and manifest["frequency"] == "1h"
    assert manifest["time_basis"] == "regular_timestamp" and manifest["timezone"] == "UTC"
    assert manifest["source_sha256"] == file_sha256(source) and manifest["future_source_sha256"] == file_sha256(ahead)
    assert manifest["files"] == {name: file_sha256(output / name) for name in ("context.csv", "future.csv")}
    assert {"dropped_rows": 0} in manifest["consequences"]
    assert {"unselected_context_columns": ["unselected"]} in manifest["consequences"]
    ctx = pd.read_csv(output / "context.csv", dtype=str, keep_default_na=False)
    aft = pd.read_csv(output / "future.csv", dtype=str, keep_default_na=False)
    assert list(ctx.columns) == ["id", "time", *OPTIONS["feature_columns"]]
    assert list(aft.columns) == ["id", "time", "promo"]
    assert ctx["id"].tolist() == aft["id"].tolist() == ["001", "001", "002", "002"]
    assert ctx["region"].tolist() == ["001", "", "NA", "NA"]
    assert ctx["sales"].tolist() == ["120.0", "", "230.0", "320.0"]
    assert aft["promo"].tolist() == ["1.0", "1.0", "0.0", ""]
    with pytest.raises(FileExistsError):
        export_tsfm_frames(source, output, future_csv=ahead, covariates=COVARIATES, **OPTIONS)
    assert manifest == json.loads((output / "manifest.json").read_text())


@pytest.mark.parametrize(
    ("alias", "offset", "expected_alias"),
    [
        ("MS", pd.offsets.MonthBegin(), "MS"),
        ("ME", pd.offsets.MonthEnd(), "ME"),
        ("QS", pd.offsets.QuarterBegin(startingMonth=1), "QS-JAN"),
        ("QE", pd.offsets.QuarterEnd(startingMonth=12), "QE-DEC"),
        ("YS", pd.offsets.YearBegin(), "YS-JAN"),
        ("YE", pd.offsets.YearEnd(), "YE-DEC"),
        ("W-MON", pd.offsets.Week(weekday=0), "W-MON"),
        ("B", pd.offsets.BusinessDay(), "B"),
    ],
)
@pytest.mark.parametrize("explicit", [False, True])
def test_export_calendar_frequency_alias_and_future_grid(tmp_path, alias, offset, expected_alias, explicit):
    from ai4ts.data_protocols import export_tsfm_frames

    dates = pd.date_range("2024-01-01", periods=12, freq=offset, tz="UTC")
    source, output = tmp_path / "history.csv", tmp_path / "output"
    pd.DataFrame({"id": "a", "time": dates, "sales": 1}).to_csv(source, index=False)
    options = dict(OPTIONS, feature_columns=["sales"], frequency=alias if explicit else None)
    manifest = export_tsfm_frames(source, output, **options)
    exported_alias = manifest["frequency"]
    if alias == "QS" and not explicit:
        assert exported_alias.startswith("QS-")
        assert dates.equals(pd.date_range(dates[0], periods=len(dates), freq=exported_alias))
    else:
        assert exported_alias == expected_alias
    exported = pd.read_csv(output / "future.csv")
    assert pd.to_datetime(exported.time, utc=True).tolist() == [dates[-1] + k * offset for k in (1, 2)]


def test_calendar_input_validates_each_entity_and_supplied_horizon():
    from ai4ts.data_protocols import validate_future_covariates

    month_start = pd.offsets.MonthBegin()
    a = pd.date_range("2023-12-01", periods=4, freq=month_start)
    b = pd.date_range("2024-01-01", periods=2, freq=month_start)
    history = pd.DataFrame({"id": ["a"] * len(a) + ["b"] * len(b), "time": a.append(b), "sales": 1, "promo": 0})
    future = pd.DataFrame(
        {
            "id": ["a", "a", "b", "b"],
            "time": [last + k * month_start for last in (a[-1], b[-1]) for k in (1, 2)],
            "promo": [1, np.nan, 0, 1],
        },
    )
    options = dict(OPTIONS, feature_columns=["sales", "promo"], covariates={"promo": COVARIATES["promo"]})
    result = validate_future_covariates(history, future, **options)
    assert result.time.tolist() == pd.to_datetime(future.time, utc=True).tolist()
    assert pd.isna(result.promo.iloc[1])
    wrong = future.assign(time=future.time.mask(future.index == 0, future.time.iloc[0] + pd.Timedelta(days=1)))
    with pytest.raises(ValueError, match="continue"):
        validate_future_covariates(history, wrong, **options)
    inconsistent = pd.concat([history, history.assign(id="c", time=history.time + pd.offsets.MonthEnd())])
    inconsistent = inconsistent.drop_duplicates(["id", "time"])
    with pytest.raises(ValueError, match="cadence"):
        validate_future_covariates(inconsistent, future, **options)
    off_grid = history.assign(time=history.time.mask(history.index == 0, pd.Timestamp("2023-11-30")))
    with pytest.raises(ValueError, match="cadence"):
        validate_future_covariates(off_grid, future, **dict(options, frequency="MS"))


@pytest.mark.parametrize(("kind", "constant", "different"), [("numeric", "1.0", "2.0"), ("categorical", "001", "002")])
def test_export_rejects_static_values_that_change_within_an_entity(tmp_path, kind, constant, different):
    from ai4ts.data_protocols import export_tsfm_frames

    source, output = tmp_path / "history.csv", tmp_path / "output"
    history = pd.DataFrame(
        {"id": "a", "time": pd.date_range("2024-01-01", periods=12, freq="h"), "sales": 1,
         "s": [constant] * 11 + [different]},
    )
    history.to_csv(source, index=False)
    options = dict(OPTIONS, feature_columns=["sales", "s"], covariates={
        "s": {"role": "static", "kind": kind, "evidence": EVIDENCE},
    })
    with pytest.raises(ValueError, match=r"Static covariate 's'.*entity 'a'"):
        export_tsfm_frames(source, output, **options)
    assert not output.exists()
    equivalent = "1e0" if kind == "numeric" else constant
    a = history.assign(s=[constant] * 10 + [equivalent, None])
    pd.concat([a, history.assign(id="b", s=different)]).to_csv(source, index=False)
    export_tsfm_frames(source, output, **options)
    context = pd.read_csv(output / "context.csv", keep_default_na=False, dtype=str)
    assert context.loc[context.id == "a", "s"].tolist() == [constant] * 10 + [equivalent, ""]
    assert context.loc[context.id == "b", "s"].tolist() == [different] * 12


def test_export_single_series_with_no_identifier_and_supplied_known_covariates(tmp_path):
    from ai4ts.data_protocols import export_tsfm_frames, validate_future_covariates

    history = pd.DataFrame({"time": ["2024-01-01", "2024-02-01", "2024-03-01"], "sales": [1, 2, 3], "promo": 0})
    future = pd.DataFrame({"time": ["2024-04-01", "2024-05-01"], "promo": [1, 0]})
    opts = dict(
        OPTIONS, sample_id_col=None, feature_columns=["sales", "promo"], covariates={"promo": COVARIATES["promo"]},
    )
    validated = validate_future_covariates(history, future, **opts)
    source, ahead, output = (tmp_path / name for name in ("history.csv", "future.csv", "out"))
    history.to_csv(source, index=False)
    future.to_csv(ahead, index=False)
    manifest = export_tsfm_frames(source, output, future_csv=ahead, **opts)
    assert manifest["sample_id_col"] == "item_id"
    assert {"generated_entity_id": {"column": "item_id", "value": "0"}} in manifest["consequences"]
    assert manifest["future_source_sha256"] == file_sha256(ahead)
    ctx = pd.read_csv(output / "context.csv", dtype=str)
    aft = pd.read_csv(output / "future.csv", dtype=str)
    assert ctx.item_id.tolist() == ["0"] * 3
    assert aft.item_id.tolist() == validated.item_id.tolist() == ["0"] * 2
    assert aft.promo.tolist() == ["1", "0"]
    assert list(history.columns) == ["time", "sales", "promo"] and list(future.columns) == ["time", "promo"]


def test_cli_without_sample_id_generates_an_entity_and_grid(tmp_path):
    source, output = tmp_path / "history.csv", tmp_path / "out"
    source.write_text("time,sales,item_id\n2024-01-01,1,9\n2024-01-02,2,10\n2024-01-03,3,11\n")
    result = CliRunner().invoke(cli, [
        "data", "export-tsfm", "--input", str(source), "--timestamp", "time", "--features", "sales,item_id",
        "--targets", "sales", "--n-pred-steps", "2", "--output", str(output),
    ])
    assert result.exit_code == 0, result.output
    manifest = json.loads(result.stdout)
    assert manifest["sample_id_col"] == "item_id_"
    assert {"generated_entity_id": {"column": "item_id_", "value": "0"}} in manifest["consequences"]
    ctx = pd.read_csv(output / "context.csv", dtype=str)
    aft = pd.read_csv(output / "future.csv", dtype=str)
    assert ctx.item_id.tolist() == ["9", "10", "11"]
    assert aft.item_id_.tolist() == ["0", "0"]
    assert pd.to_datetime(aft.time).dt.day.tolist() == [4, 5]


def test_export_categorical_future_and_generated_grid(frames, tmp_path):
    from ai4ts.data_protocols import export_tsfm_frames

    history, future = frames
    source, ahead, output, generated = (tmp_path / name for name in ("history.csv", "future.csv", "out", "generated"))
    history.to_csv(source, index=False)
    future.assign(region=["NA", "001", "low", ""]).to_csv(ahead, index=False)
    known_region = dict(COVARIATES["region"], role="known")
    declared = dict(COVARIATES, region=known_region)
    manifest = export_tsfm_frames(source, output, future_csv=ahead, covariates=declared, **OPTIONS)
    assert manifest["covariates"]["region"] == known_region
    out = pd.read_csv(output / "future.csv", dtype=str, keep_default_na=False)
    assert out["region"].tolist() == ["001", "", "low", "NA"]

    static_only = {"region": COVARIATES["region"]}
    manifest = export_tsfm_frames(source, generated, covariates=static_only, **OPTIONS)
    assert manifest["future_source_sha256"] is None
    assert {"generated_future_rows": 4} in manifest["consequences"]
    out = pd.read_csv(generated / "future.csv", dtype=str, keep_default_na=False)
    assert list(out.columns) == ["id", "time"] and len(out) == 4


def test_failed_manifest_write_cleans_up_new_export(frames, tmp_path, monkeypatch):
    from ai4ts.data_protocols import covariates

    def fail(*args):
        raise OSError("controlled write failure")

    history, future = frames
    source, ahead, output = (tmp_path / name for name in ("history.csv", "future.csv", "output"))
    history.to_csv(source, index=False)
    future.to_csv(ahead, index=False)
    monkeypatch.setattr(covariates, "write_json", fail)
    with pytest.raises(OSError, match="controlled write failure"):
        covariates.export_tsfm_frames(source, output, future_csv=ahead, covariates=COVARIATES, **OPTIONS)
    assert not output.exists()
    assert source.is_file() and ahead.is_file()


def test_export_rejects_numeric_tokens_with_embedded_null(tmp_path):
    from ai4ts.data_protocols import export_tsfm_frames

    source, output = tmp_path / "history.csv", tmp_path / "output"
    source.write_bytes(b"id,time,sales\n001,2024-01-01,5\x00c\n")
    with pytest.raises(ValueError, match="real numeric"):
        export_tsfm_frames(source, output, **dict(OPTIONS, feature_columns=["sales"], frequency="1h"))
    assert not output.exists()


def test_cli_export_and_invalid_input(frames, tmp_path):
    history, future = frames
    source, ahead, declared, output = (tmp_path / name for name in ("history.csv", "ahead.csv", "roles.json", "output"))
    history.to_csv(source, index=False)
    future.to_csv(ahead, index=False)
    declared.write_text(json.dumps(COVARIATES), encoding="utf-8")
    args = [
        "data", "export-tsfm", "--input", str(source), "--future", str(ahead), "--covariates", str(declared),
        "--sample-id", "id", "--timestamp", "time", "--features", ",".join(OPTIONS["feature_columns"]),
        "--targets", "sales", "--n-pred-steps", "2", "--output", str(output),
    ]
    runner = CliRunner()
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == json.loads((output / "manifest.json").read_text())
    rejected = tmp_path / "rejected"
    future.assign(sales=10).to_csv(ahead, index=False)
    result = runner.invoke(cli, [*args[:-1], str(rejected)])
    assert result.exit_code != 0 and "target" in json.loads(result.stdout)["error"]
    assert not rejected.exists()


def test_cli_reports_missing_required_options_as_json():
    result = CliRunner().invoke(cli, ["data", "export-tsfm"])
    assert result.exit_code != 0
    assert "--input" in json.loads(result.stdout)["error"]


def test_task_binding_defaults_reuse_legacy_and_conflicts(tmp_path):
    source = tmp_path / "series.csv"
    source.write_text("price,y,promo\n" + "".join(f"{i},{i * 10},{i % 2}\n" for i in range(9)))
    options = {"task": "forecasting", "window_size": 3, "n_pred_steps": 2, "target_features": ["y"], "missing_rate": 0}
    default = prepare_for_pypots(source, **options)
    assert default["task_state"]["covariates"] == {"price": DEFAULT, "promo": DEFAULT}
    declaration = {"promo": COVARIATES["promo"]}
    declared = prepare_for_pypots(source, **options, covariates=declaration)
    state, registry = declared["task_state"], declared["registry"]
    assert state["covariates"] == {"price": DEFAULT, **declaration}
    assert state["covariates_sha256"] == state_sha256(state["covariates"])
    assert registry.covariates == state["covariates"]
    np.testing.assert_array_equal(declared["X"][:, :, :2], default["X"][:, :, :2])
    np.testing.assert_array_equal(declared["X"][0, :, 2], [0, 1, 0])
    np.testing.assert_array_equal(declared["X_pred"], default["X_pred"])
    assert registry.feature_columns == ["y", "price", "promo@lead"]
    assert registry.task_state_sha256 == state_sha256(state)
    assert WindowRegistry.from_dict(registry.to_dict()).fingerprint() == registry.fingerprint()
    held = prepare_for_pypots(source, task="forecasting", set_type="test", task_state=state, missing_rate=0)
    assert held["registry"].covariates == state["covariates"]
    held["registry"].validate_task_binding(source, held["registry"].prediction_binding(), state)
    with pytest.raises(ValueError, match="saved training"):
        prepare_for_pypots(source, task="forecasting", set_type="full", task_state=state, covariates={})
    with pytest.raises(ValueError, match="forecasting"):
        prepare_for_pypots(source, task="imputation", covariates=declaration)

    bad_state = copy.deepcopy(state)
    bad_state["covariates"]["promo"]["evidence"]["reference"] = "Changed"
    with pytest.raises(ValueError, match="binding"):
        validate_task_state(bad_state)
    bad_registry = copy.deepcopy(registry.to_dict())
    bad_registry["covariates"]["promo"]["evidence"]["reference"] = "Changed"
    with pytest.raises(ValueError, match="binding"):
        WindowRegistry.from_dict(bad_registry)
    bad_registry["covariates_sha256"] = state_sha256(bad_registry["covariates"])
    altered = WindowRegistry.from_dict(bad_registry)
    with pytest.raises(ValueError, match="binding"):
        altered.validate_task_binding(source, altered.prediction_binding(), state)

    covariate_fields = {"covariates", "covariates_sha256"}
    new_state_fields = covariate_fields | {"input_features", "categorical_encoding"}
    legacy_state = {k: v for k, v in default["task_state"].items() if k not in new_state_fields}
    legacy_registry = {k: v for k, v in default["registry"].to_dict().items() if k not in covariate_fields}
    legacy_registry["feature_columns"] = ["price", "y", "promo"]
    legacy_registry["task_state_sha256"] = state_sha256(legacy_state)
    old = WindowRegistry.from_dict(legacy_registry)
    assert old.to_dict() == legacy_registry
    old.validate_task_binding(source, old.prediction_binding(), legacy_state)
    live = prepare_for_pypots(source, task="forecasting", set_type="full", task_state=legacy_state)
    assert live["registry"].covariates == default["registry"].covariates
    assert default["registry"].feature_columns == ["y", "price", "promo"]
    assert live["registry"].feature_columns == legacy_registry["feature_columns"]
    np.testing.assert_array_equal(live["X"][0], default["X"][0][:, [1, 0, 2]])
    assert live["task_state"] == legacy_state
