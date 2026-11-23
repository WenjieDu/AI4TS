"""Small explicit method catalog; querying it never imports a backend."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import hashlib
from importlib.util import find_spec
import importlib.metadata as metadata
from pathlib import Path

CATALOG = {
    "profile": ("numpy", "original rows", "empty cells retained", "descriptive", "https://numpy.org/doc/"),
    "acf": (
        "statsmodels",
        "regular timestamps",
        "conservative; keep grid",
        "statistical_estimation",
        "https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html",
    ),
    "pacf": (
        "statsmodels",
        "regular timestamps",
        "complete",
        "statistical_estimation",
        "https://www.statsmodels.org/stable/tsa.html",
    ),
    "adf": (
        "statsmodels",
        "regular timestamps",
        "complete",
        "statistical_estimation",
        "https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html",
    ),
    "kpss": (
        "statsmodels",
        "regular timestamps",
        "complete",
        "statistical_estimation",
        "https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html",
    ),
    "stl": (
        "statsmodels",
        "regular timestamps; declared period",
        "complete",
        "statistical_estimation",
        "https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html",
    ),
    "lombscargle": (
        "scipy",
        "strictly increasing actual timestamps",
        "finite observations retain real times",
        "statistical_estimation",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lombscargle.html",
    ),
    "last-value": (
        "numpy",
        "regular timestamps; fixed training cutoff",
        "finite last training value",
        "baseline_prediction",
        "Repeat the final training cycle from the fixed forecast origin",
    ),
    "seasonal-naive": (
        "numpy",
        "regular timestamps; declared period",
        "complete final training cycle",
        "baseline_prediction",
        "Repeat the final training cycle from the fixed forecast origin",
    ),
}


def available_methods():
    """Describe supported adapters and installed, unprobed dependencies.

    Returns
    -------
    dict
        Method contracts and package metadata, without running/importing backends.
    """
    methods = {}
    for name, (backend, time, missing, computation, reference) in CATALOG.items():
        try:
            version = metadata.version(backend)
        except metadata.PackageNotFoundError:
            version = None
        methods[name] = dict(
            backend=backend,
            version=version,
            dependency_status="installed_not_probed" if version else "not_installed",
            time_requirement=time,
            missing_policy=missing,
            computation=computation,
            reference=reference,
        )
    return dict(status="success", methods=methods, backend_imports=False, backend_calls=0)


def implementation_identity():
    """Identify active code and dependency records without importing data backends."""
    spec = find_spec("ai4ts")
    if spec is None or spec.origin is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("Cannot locate the active ai4ts package")
    protocol_root = Path(spec.origin).parent / "data_protocols"

    files = {}
    roots = [("ai4ts.analysis", Path(__file__).parent), ("ai4ts.data_protocols", protocol_root)]
    for namespace, root in roots:
        paths = (
            root.glob("*.py")
            if namespace == "ai4ts.analysis"
            else (
                root / name
                for name in (
                    "analysis.py",
                    "ingestion.py",
                    "pipeline.py",
                    "profile.py",
                    "semantics.py",
                    "preprocessing.py",
                    "registry.py",
                )
            )
        )
        for path in sorted(paths):
            files[namespace + "." + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    packages = {}
    for name in ("numpy", "pandas", "scipy", "statsmodels"):
        try:
            dist = metadata.distribution(name)
            record = dist.read_text("RECORD") or ""
            packages[name] = dict(version=dist.version, record_sha256=hashlib.sha256(record.encode()).hexdigest())
        except metadata.PackageNotFoundError:
            packages[name] = None
    return dict(schema="analysis-implementation-v1", code=files, packages=packages)
