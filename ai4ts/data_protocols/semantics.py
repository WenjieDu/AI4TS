"""Optional source-bound units, independent of numerical preprocessing."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import unicodedata

from .preprocessing import state_sha256


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid feature semantics source fingerprint")


def _label(value, limit=64):
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > limit
        or any(unicodedata.category(c).startswith("C") or c in "\n\r" for c in value)
    ):
        raise ValueError("Unit labels must be nonempty, bounded, single-line strings")


def validate_feature_semantics(state, feature_columns=None, source_sha256=None):
    """Validate a portable description before reading its declared units.

    Parameters
    ----------
    state : dict
        Versioned ordered units and their source/declaration identities.
    feature_columns : list of str or None
        Expected feature order, when known.
    source_sha256 : str or None
        Expected prepared input source identity, when known.

    Returns
    -------
    dict
        The validated description, without modifying it.
    """
    if not isinstance(state, dict) or type(state.get("version")) is not int or state["version"] != 1:
        raise ValueError("Unsupported feature semantics")
    features = state.get("feature_columns")
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(f, str) or not f for f in features)
        or len(set(features)) != len(features)
        or (feature_columns is not None and features != list(feature_columns))
    ):
        raise ValueError("Feature semantics order mismatch")
    _hash(state.get("source_sha256"))
    if source_sha256 is not None and state["source_sha256"] != source_sha256:
        raise ValueError("Feature semantics/source binding mismatch")
    units = state.get("units")
    if not isinstance(units, dict) or set(units) != set(features):
        raise ValueError("Feature semantics units must match selected features")
    for label in units.values():
        _label(label)
    declaration = state.get("declaration", {})
    if not isinstance(declaration, dict) or declaration.get("origin") not in ("source", "user-confirmed"):
        raise ValueError("Units require source evidence or user confirmation")
    _hash(declaration.get("source_sha256"))
    if declaration.get("origin") == "source":
        _label(declaration.get("reference"), 256)
    elif "reference" in declaration:
        _label(declaration["reference"], 256)
    inherited = state.get("inherited_from_sha256")
    if inherited is not None:
        _hash(inherited)
    elif declaration["source_sha256"] != state["source_sha256"]:
        raise ValueError("Unit evidence belongs to another source")
    state_sha256(state)
    return state


def prepare_feature_semantics(feature_units, features, source_sha256, unit_state=None):
    """Bind optional declarations or saved training units to a prepared input.

    Parameters
    ----------
    feature_units : dict or None
        A feature-to-unit mapping confirms units explicitly. Source evidence
        instead uses ``units``, ``origin``, ``source_sha256`` and ``reference``.
    features : list of str
        Confirmed feature order.
    source_sha256 : str
        Current source fingerprint.
    unit_state : dict or None
        Saved training description to inherit or check against.

    Returns
    -------
    dict or None
        Bound semantic state, or None for legacy undeclared data.
    """
    _hash(source_sha256)
    if unit_state is not None:
        validate_feature_semantics(unit_state, features)
    if feature_units is None:
        if unit_state is None:
            return None
        result = copy.deepcopy(unit_state)
        result.update(source_sha256=source_sha256, inherited_from_sha256=state_sha256(unit_state))
        return validate_feature_semantics(result, features, source_sha256)
    if not isinstance(feature_units, dict):
        raise ValueError("feature_units must be a mapping")
    if isinstance(feature_units.get("units"), dict):
        if set(feature_units) - {"units", "origin", "source_sha256", "reference"}:
            raise ValueError("Unknown unit declaration fields")
        declared = feature_units["units"]
        declaration = {k: v for k, v in feature_units.items() if k != "units"}
        declaration.setdefault("origin", "user-confirmed")
        if declaration["origin"] == "source" and "source_sha256" not in declaration:
            raise ValueError("Source unit evidence requires its fingerprint")
        declaration.setdefault("source_sha256", source_sha256)
    else:
        declared = feature_units
        declaration = {"origin": "user-confirmed", "source_sha256": source_sha256}
    if set(declared) - set(features):
        raise ValueError("Unit declaration names an unselected feature")
    for feature, label in declared.items():
        _label(label)
        if unit_state is not None and unit_state["units"][feature] not in ("unknown", label):
            raise ValueError("Explicit unit differs from the saved training unit")
    result = {
        "version": 1,
        "source_sha256": source_sha256,
        "feature_columns": list(features),
        "units": {f: declared.get(f, unit_state["units"][f] if unit_state else "unknown") for f in features},
        "declaration": declaration,
    }
    return validate_feature_semantics(result, features, source_sha256)


def validate_semantic_binding(state, digest, features=None, source_sha256=None):
    """Validate a description/hash pair; both absent denotes legacy data.

    Parameters
    ----------
    state : dict or None
        Optional semantic description.
    digest : str or None
        Its bound canonical identity.
    features : list of str or None
        Expected feature order.
    source_sha256 : str or None
        Expected input source identity.

    Returns
    -------
    dict or None
        Validated state, or None for a legacy input.
    """
    if state is None and digest is None:
        return None
    validate_feature_semantics(state, features, source_sha256)
    if state_sha256(state) != digest:
        raise ValueError("Feature semantics binding mismatch")
    return state
