"""The statistical analysis tools: unsupervised statistics of catalog features on the data.

Each tool takes a list of catalog ``feature_name`` strings, encodes them exactly as a
team's committed features are encoded for scoring (``Knowledge/features.py``, fitted on
the same rows), and returns a JSON object, so an agent analyses the very columns XGBoost
would be trained on. The rows are ``ToolResources.dataset``: the dev split, the rows the
CV folds are cut from; the holdout never reaches an agent. The target is never analysed.

Every tool returns ``"not_in_catalog"``: the requested names the catalog does not have,
which are skipped (an agent's mistake, answered so it can correct itself, not an error).
Results are keyed by feature_name, then by encoded column: one column for most
encodings, one per category for ``one_hot`` / ``bin``. A statistic that is undefined
(e.g. the mean of a column with no values) is ``null``.

  distribution_analysis    n, mean, std, variance, median, iqr per column
  missing_values_analysis  missing_pct, missing, n per column
  correlation_analysis     Pearson r between every pair of columns
  prevalence_analysis      share present (0/1 columns) or share measured (others)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain_core.tools import BaseTool, tool
from typeguard import typechecked

from ...Knowledge.features import build_feature_frame_with_sources
from .tool_resources import ToolResources

DISTRIBUTION_ANALYSIS = "distribution_analysis"
MISSING_VALUES_ANALYSIS = "missing_values_analysis"
CORRELATION_ANALYSIS = "correlation_analysis"
PREVALENCE_ANALYSIS = "prevalence_analysis"
STATS_TOOL_NAMES = (DISTRIBUTION_ANALYSIS, MISSING_VALUES_ANALYSIS, CORRELATION_ANALYSIS,
                    PREVALENCE_ANALYSIS)

# Pairs at or above this |r| are reported as redundant.
REDUNDANT_ABS_R = 0.5
# Pairs reported when none is redundant, strongest first.
STRONGEST_PAIRS = 5
DIGITS = 4

_ARG = ("features: a list of feature_name strings copied verbatim from pending_features, "
        "accepted_features or rejected_features.")


def _number(value: Any) -> Optional[float]:
    """A JSON-safe rounded number; NaN and infinities become None (``null``)."""
    value = float(value)
    return round(value, DIGITS) if math.isfinite(value) else None


@typechecked
def _encode(resources: ToolResources, features: List[str]
            ) -> Tuple[Optional[pd.DataFrame], Dict[str, str], List[str], List[str]]:
    """Encode the catalog names among ``features`` on the dataset.

    Returns (numeric frame without the target or None, {column: feature_name}, names
    encoded, names not in the catalog). Repeated names are encoded once.
    """
    names = list(dict.fromkeys(f.strip() for f in features if f.strip()))
    known = [n for n in names if n in resources.catalog]
    unknown = [n for n in names if n not in resources.catalog]
    if not known:
        return None, {}, known, unknown
    data = resources.dataset
    frame, sources = build_feature_frame_with_sources(
        data, known, dict(resources.catalog), resources.target_column, fit_df=data)
    frame = frame.drop(columns=[resources.target_column])
    return frame.apply(pd.to_numeric, errors="coerce"), sources, known, unknown


def _per_feature(sources: Dict[str, str], known: List[str],
                 stats: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group per-column statistics under their feature, in the order asked."""
    out: Dict[str, Dict[str, Any]] = {name: {} for name in known}
    for column, values in stats.items():
        out[sources[column]][column] = values
    return out


def _answer(known: List[str], unknown: List[str], body: Dict[str, Any]) -> Dict[str, Any]:
    answer = dict(body, not_in_catalog=unknown)
    if not known:
        answer["note"] = ("no catalog feature_name in the list; copy names verbatim from "
                          "pending_features, accepted_features or rejected_features.")
    return answer


def _distribution(column: pd.Series) -> Dict[str, Any]:
    if not column.notna().any():    # undefined; computing it would only warn
        return {"n": 0, "mean": None, "std": None, "variance": None, "median": None,
                "iqr": None}
    return {"n": int(column.notna().sum()), "mean": _number(column.mean()),
            "std": _number(column.std()), "variance": _number(column.var()),
            "median": _number(column.median()),
            "iqr": _number(column.quantile(0.75) - column.quantile(0.25))}


def _missing(column: pd.Series) -> Dict[str, Any]:
    missing = int(column.isna().sum())
    return {"missing_pct": _number(100.0 * missing / len(column)), "missing": missing,
            "n": len(column)}


def _prevalence(column: pd.Series) -> Dict[str, Any]:
    values = set(column.dropna().unique())
    n = len(column)
    if values and values <= {0.0, 1.0}:
        present = int((column == 1).sum())
        return {"kind": "binary", "prevalence": _number(present / n), "present": present,
                "n": n}
    measured = int(column.notna().sum())
    return {"kind": "continuous", "measured": _number(measured / n), "non_null": measured,
            "n": n}


@typechecked
def stats_tools(resources: ToolResources) -> List[BaseTool]:
    """The four analysis tools over ``resources.dataset`` (see the module docstring)."""

    def per_column(features: List[str], stat: Any) -> Dict[str, Any]:
        frame, sources, known, unknown = _encode(resources, features)
        stats = {} if frame is None else {str(c): stat(frame[c]) for c in frame.columns}
        return _answer(known, unknown, {"features": _per_feature(sources, known, stats)})

    @tool(DISTRIBUTION_ANALYSIS, description=(
        "Descriptive statistics of each feature, for continuous features and outliers. "
        + _ARG + " Returns {\"features\": {feature_name: {column: {n, mean, std, variance, "
        "median, iqr}}}, \"not_in_catalog\": [...]}; null where undefined."))
    def distribution_analysis(features: List[str]) -> Dict[str, Any]:
        return per_column(features, _distribution)

    @tool(MISSING_VALUES_ANALYSIS, description=(
        "Missingness of each feature. " + _ARG + " Returns {\"features\": {feature_name: "
        "{column: {missing_pct, missing, n}}}, \"not_in_catalog\": [...]}; missing_pct is "
        "0-100."))
    def missing_values_analysis(features: List[str]) -> Dict[str, Any]:
        return per_column(features, _missing)

    @tool(PREVALENCE_ANALYSIS, description=(
        "Prevalence of each feature, for non-continuous features. " + _ARG + " Returns "
        "{\"features\": {feature_name: {column: {...}}}, \"not_in_catalog\": [...]}; a 0/1 "
        "column gives {kind: \"binary\", prevalence, present, n} (share equal to 1), any "
        "other {kind: \"continuous\", measured, non_null, n} (share with a value)."))
    def prevalence_analysis(features: List[str]) -> Dict[str, Any]:
        return per_column(features, _prevalence)

    @tool(CORRELATION_ANALYSIS, description=(
        "Pairwise Pearson correlation between the features' columns, for redundancy. "
        + _ARG + f" Returns {{\"redundant_pairs\": [[column_a, column_b, r], ...] (|r| >= "
        f"{REDUNDANT_ABS_R}, strongest first), \"strongest_pairs\": the {STRONGEST_PAIRS} "
        "strongest pairs when none is redundant, else [], \"columns\": {column: "
        "feature_name}, \"not_in_catalog\": [...]}; needs at least two columns."))
    def correlation_analysis(features: List[str]) -> Dict[str, Any]:
        frame, sources, known, unknown = _encode(resources, features)
        body: Dict[str, Any] = {"redundant_pairs": [], "strongest_pairs": [],
                                "columns": sources}
        if frame is None:
            return _answer(known, unknown, body)
        if frame.shape[1] < 2:
            body["note"] = "need at least two columns to correlate; add features."
            return _answer(known, unknown, body)
        corr = frame.corr()
        columns = [str(c) for c in corr.columns]
        pairs = [(columns[i], columns[j], float(corr.iat[i, j]))
                 for i in range(len(columns)) for j in range(i + 1, len(columns))
                 if pd.notna(corr.iat[i, j])]
        pairs.sort(key=lambda p: abs(p[2]), reverse=True)
        body["redundant_pairs"] = [[a, b, _number(r)] for a, b, r in pairs
                                   if abs(r) >= REDUNDANT_ABS_R]
        if not body["redundant_pairs"]:
            body["strongest_pairs"] = [[a, b, _number(r)] for a, b, r in
                                       pairs[:STRONGEST_PAIRS]]
        return _answer(known, unknown, body)

    return [distribution_analysis, missing_values_analysis, correlation_analysis,
            prevalence_analysis]
