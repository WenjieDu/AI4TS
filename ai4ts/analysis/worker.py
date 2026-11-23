"""Internal bounded worker; accepts no arbitrary executable command."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import contextlib
import sys
from pathlib import Path


def main():
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        from ai4ts.data_protocols.analysis import read_analysis_source
        from .artifacts import _plots, _report, _validate, read_json, write_json

        root = Path(sys.argv[1])
        plan = read_json(root / "plan.json")
        _validate(plan, root / "source.csv")
        source, _, _ = read_analysis_source(root / "source.csv", plan["roles"])
        if plan["request"]["kind"] == "diagnostics":
            from .diagnostics import compute_diagnostics

            result = compute_diagnostics(source, plan, root / "calls.jsonl")
        else:
            from .baselines import compute_baselines

            result = compute_baselines(source, plan, root / "calls.jsonl")
        arrays = result.pop("arrays")
        np.savez(root / "arrays.npz", **arrays)
        write_json(root / "results.json", result)
        meta = read_json(root / "analysis.json")
        meta["stage"] = "publication"
        write_json(root / "analysis.json", meta)
        result["figures"] = _plots(root, source, plan, result, arrays)
        write_json(root / "results.json", result)
        meta["status"] = "success" if all(r["status"] == "succeeded" for r in result["records"]) else "partial"
        write_json(root / "analysis.json", meta)
        _report(root, plan, result, meta)


if __name__ == "__main__":
    main()
