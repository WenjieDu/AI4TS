"""Bounded analysis execution and immutable evidence publication."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import contextlib
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .capabilities import available_methods, implementation_identity

SCHEMA = "timeseries-analysis-v1"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path, limit=8 * 1024 * 1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Missing, linked or oversized analysis JSON")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: _bad_constant(value))


def _bad_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
        return digest.hexdigest()


def create_plan(csv_path, **request):
    """Bind the shared preview to this implementation and installed backends."""
    from ai4ts.data_protocols.analysis import create_analysis_plan
    from ai4ts.data_protocols.ingestion import content_sha256

    request["execution_identity"] = implementation_identity()
    result = create_analysis_plan(csv_path, **request)
    methods = request.get("methods") or []
    if request.get("kind") == "forecast-baselines":
        methods = request.get("candidates") or ["last-value", "seasonal-naive"]
    catalog = available_methods()["methods"]
    missing = [name for name in methods if name in catalog and catalog[name]["dependency_status"] == "not_installed"]
    if missing:
        result["status"] = "blocked"
        result["errors"].append("Optional backend missing for: " + ", ".join(missing))
        result.pop("sha256")
        result["sha256"] = content_sha256(result)
    return result


def _validate(plan, source):
    from ai4ts.data_protocols.analysis import validate_analysis_plan

    validate_analysis_plan(plan, source)
    if plan["request"]["execution_identity"] != implementation_identity():
        raise ValueError("Analysis implementation/dependencies differ; regenerate the plan in a matching environment")


def _plots(root, source, plan, result, arrays):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    (root / "figures").mkdir()
    temporal = {g["entity"]: g["time"] for g in plan["groups"]}
    bindings = []
    for index, record in enumerate(result["records"]):
        if record["status"] != "succeeded":
            continue
        method = record["method"]
        if method in ("adf", "kpss"):
            continue
        fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
        used = []
        if method == "profile":
            rows = record["row_indices"]
            time_column = plan["roles"]["timestamp_col"]
            x = (
                pd.to_datetime(source.loc[rows, time_column], format="mixed", utc=True)
                if temporal[record["entity"]]["valid"]
                else rows
            )
            ax.plot(x, source.loc[rows, record["feature"]].to_numpy(), marker=".", markersize=2)
            ax.set_xlabel("Time (UTC)" if temporal[record["entity"]]["valid"] else "Original source row")
        elif method in ("acf", "pacf"):
            values = np.asarray(record["result"]["values"], dtype=float)
            ax.stem(np.arange(len(values)), values)
            ax.set_xlabel("Lag (observations)")
            used = list(record["arrays"].values())
        elif method == "stl":
            for name, key in record["arrays"].items():
                ax.plot(record["row_indices"], arrays[key], label=name)
            ax.set_xlabel("Original source row")
            ax.legend()
            used = list(record["arrays"].values())
        elif method == "lombscargle":
            omega = arrays[record["arrays"]["angular_frequency"]]
            ax.plot(2 * np.pi / omega, arrays[record["arrays"]["power"]])
            ax.set_xlabel("Candidate period (seconds); significance not estimated")
            used = list(record["arrays"].values())
        else:
            group = next(g for g in plan["groups"] if g["entity"] == record["entity"])
            for split, fh_name in (("val", "validation_fh"), ("test", "test_fh")):
                key = record["arrays"].get(split + "_prediction")
                if key:
                    ax.plot(group[fh_name], arrays[key], label=split + " prediction")
                    ax.plot(group[fh_name], arrays[record["arrays"][split + "_truth"]], ".", label=split + " truth")
                    used.extend([key, record["arrays"][split + "_truth"]])
            ax.set_xlabel("Forecast steps from fixed training cutoff")
            ax.legend()
        ax.set_title(f"{record['entity']} / {record['feature']} / {method}", parse_math=False)
        ax.set_ylabel(
            record["unit"] if method in ("profile", "stl", "last-value", "seasonal-naive") else "Dimensionless",
            parse_math=False,
        )
        relative = f"figures/{index:03d}.png"
        fig.savefig(root / relative, dpi=140)
        plt.close(fig)
        record["figures"] = [relative]
        bindings.append(
            dict(
                path=relative,
                entity=record["entity"],
                feature=record["feature"],
                method=method,
                arrays=used,
                source_rows=record["row_indices"],
            )
        )
    return bindings


def _report(root, plan, result, meta):
    import html

    lines = [
        "# Time-series analysis",
        "",
        f"Status: {meta['status']}",
        "",
        "Original observations were not filled, resampled, sorted or overwritten.",
        "Statistics are descriptive evidence. Non-rejection is not proof; correlation is not causation.",
        "Fixed-origin baseline results must not be ranked against rolling-window neural results.",
        "Saved-result inspection checks integrity/source semantics; it does not rerun statistical tests.",
        "",
    ]
    for group in plan["groups"]:
        temporal = {k: v for k, v in group["time"].items() if k not in ("rows", "seconds")}
        lines += ["Time evidence for " + html.escape(group["entity"]) + ": " + html.escape(json.dumps(temporal)), ""]
    for item in result["records"]:
        title = html.escape(f"{item['entity']} / {item['feature']} / {item['method']}")
        lines += [
            "## " + title,
            "",
            f"Status: {item['status']}; backend: {item['backend']} {item.get('version', 'unknown')}; "
            f"unit: {html.escape(item['unit'])}",
            "",
        ]
        if item.get("selected"):
            lines += ["Selected using validation raw MAE. Final test: " + json.dumps(item["test"], allow_nan=False), ""]
        elif item.get("validation"):
            lines += ["Validation: " + json.dumps(item["validation"], allow_nan=False), ""]
        elif item.get("result"):
            lines += [
                "Recorded evidence: " + html.escape(json.dumps(item["result"], ensure_ascii=False, allow_nan=False)),
                "",
            ]
        if item.get("reason"):
            lines += ["Limit: " + html.escape(item["reason"]), ""]
        for warning in item.get("warnings", []):
            lines += ["Warning: " + html.escape(warning), ""]
        for image in item.get("figures", []):
            if item["method"] == "profile":
                evidence = item["result"]
                caption = (
                    f"{evidence['missing_cells']} empty cells among {evidence['observations']} source observations "
                    f"(rate {evidence['missing_rate']:.3g}); finite observations: {evidence['finite_observations']}. "
                    "Missing cells remain gaps; unknown sampling schedules do not imply missing timestamps."
                )
            elif item["method"] in ("last-value", "seasonal-naive"):
                caption = (
                    f"Training cutoff is source row {item['origin_row']}; validation MAE is "
                    f"{item['validation']['mae']:.6g} {item['unit']} over "
                    f"{item['validation']['scored_positions']} finite targets. "
                    "The test horizon continues from that cutoff; the estimator was not updated with validation truth."
                )
            elif item["method"] == "lombscargle":
                caption = (
                    "Largest saved power occurs at a candidate period of "
                    f"{item['result']['candidate_period_seconds']:.6g} seconds "
                    "within the declared search range. Significance was not estimated."
                )
            elif item["method"] == "stl":
                caption = (
                    f"Robust STL uses the declared {plan['request']['period']}-observation period. "
                    f"Maximum component reconstruction error: {item['result']['reconstruction_max_error']:.3g}. "
                    "These components describe this series; they do not establish a cause."
                )
            else:
                caption = (
                    "Saved lag estimates use observation distances on the original regular grid; "
                    "no confidence intervals are claimed."
                )
                if item["method"] == "acf":
                    caption += (
                        " Per-lag finite-pair counts are recorded; fewer than eight pairs "
                        "leaves the displayed estimate unknown."
                    )
            lines += [f"![Recorded {item['method']} evidence]({image})", "", html.escape(caption), ""]
        if item.get("reference"):
            lines += [f"[Method documentation]({item['reference']})", ""]
    lines += [
        "## Scope",
        "",
        f"Analysis scope: {plan['scope']}; kind: {plan['request']['kind']}.",
        "Unknown units remain unknown. No hidden truth, significance claim or model ranking was added.",
        "[Structured results](results.json) · [Plan](plan.json) · [Operation ledger](calls.jsonl)",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _execute(csv_path, plan, output_dir, expected_kind):
    _validate(plan, csv_path)
    if plan["request"]["kind"] != expected_kind:
        raise ValueError("Plan kind does not match the selected analysis operation")
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a new analysis output directory")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    meta = dict(
        schema=SCHEMA,
        status="running",
        stage="snapshot",
        kind=expected_kind,
        plan_sha256=plan["sha256"],
        source_sha256=plan["source"]["sha256"],
        execution_identity=plan["request"]["execution_identity"],
        started_utc=datetime.now(timezone.utc).isoformat(),
        budget=plan["budget"],
        production_supported=False,
    )
    write_json(output / "analysis.json", meta)
    try:
        from ai4ts.data_protocols.ingestion import MAX_CSV_BYTES

        with Path(csv_path).open("rb") as stream:
            snapshot = stream.read(MAX_CSV_BYTES + 1)
        if len(snapshot) > MAX_CSV_BYTES:
            raise ValueError("Source exceeds the bounded snapshot limit")
        if hashlib.sha256(snapshot).hexdigest() != meta["source_sha256"]:
            raise ValueError("Source changed before snapshot")
        (output / "source.csv").write_bytes(snapshot)
        write_json(output / "plan.json", plan)
        (output / "calls.jsonl").touch()
        meta["stage"] = "computation"
        write_json(output / "analysis.json", meta)
        with (output / "compute.stdout.log").open("w") as stdout, (output / "compute.stderr.log").open("w") as stderr:
            process = subprocess.run(
                [sys.executable, "-m", "ai4ts.analysis.worker", str(output)],
                stdout=stdout,
                stderr=stderr,
                timeout=plan["budget"]["compute_seconds"],
            )
        if process.returncode:
            raise RuntimeError("Analysis worker failed; inspect saved compute.stderr.log; no automatic retry")
        meta = read_json(output / "analysis.json")
        meta.update(stage="complete", elapsed_seconds=time.monotonic() - started)
        write_json(output / "analysis.json", meta)
        files = {
            p.relative_to(output).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        }
        write_json(output / "complete.json", dict(schema=SCHEMA, files=files, status=meta["status"]))
        from .verify import inspect_analysis

        return {**inspect_analysis(output), "status": meta["status"]}
    except Exception as error:
        with contextlib.suppress(ValueError, OSError):
            meta = read_json(output / "analysis.json")
        meta.update(
            status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started
        )
        with contextlib.suppress(OSError):
            write_json(output / "analysis.json", meta)
        return dict(
            status="failed",
            analysis=str(output),
            stage=meta["stage"],
            error=meta["error"],
            next_action="Inspect saved evidence and use a new explicitly planned run; no automatic retry",
        )


def diagnose(csv_path, plan, output_dir):
    """Run a ready diagnostic plan into a new evidence directory.

    Parameters
    ----------
    csv_path : str or Path
        Original source CSV.
    plan : dict
        Ready implementation-bound diagnostic plan.
    output_dir : str or Path
        New directory; existing output is never overwritten.

    Returns
    -------
    dict
        Operation outcome and checked evidence links.
    """
    return _execute(csv_path, plan, output_dir, "diagnostics")


def compare_baselines(csv_path, plan, output_dir):
    """Execute a ready fixed-origin baseline plan.

    Parameters
    ----------
    csv_path : str or Path
        Original source CSV.
    plan : dict
        Ready implementation-bound baseline plan.
    output_dir : str or Path
        New directory for evidence and the operation ledger.

    Returns
    -------
    dict
        Outcome, validation selections, final test facts, and evidence links.
    """
    return _execute(csv_path, plan, output_dir, "forecast-baselines")
