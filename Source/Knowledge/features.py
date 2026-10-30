"""Encode a team's committed features into the numeric frame XGBoost is scored on.

A team commits catalog ``feature_name``s. Each is encoded exactly as its catalog entry
says (``Knowledge/catalog.py``); nothing here infers an encoding from a column's dtype or
looks a feature up anywhere but the catalog.

Every problem raises; no feature is skipped. A name the catalog does not have, a missing
column or an encoding that fails are our bugs or a bad catalog, and dropping the feature
would score the team on a different set than it committed. The run-start check
``assert_catalog_matches`` catches the dataset problems (missing or non-numeric columns,
malformed lab lists) before any team runs, so these errors should only ever mean a bug.
Whether a team committed any features at all is decided by the evaluator before calling.

Data-dependent encodings are fitted on ``fit_df``: the one-hot / bin category set and the
standardize mean and std. Scoring on the dev split passes the dev frame for both ``df``
and ``fit_df``; the holdout stage (``EA.holdout_evaluation``) passes the dev frame as
``fit_df``, so the holdout gets the columns and scaling the model was trained with, not
ones fitted on the test rows.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from typeguard import typechecked

from .catalog import (ENCODING_PARAMS, NUMERIC_ENCODINGS, CatalogEntry,
                                 normalize_med, parse_list_cell, source_columns)


@typechecked
def build_feature_frame(df: pd.DataFrame, feature_names: List[str],
                        catalog: Dict[str, CatalogEntry], target_column: str,
                        fit_df: pd.DataFrame) -> pd.DataFrame:
    """Encode ``feature_names`` from ``df`` and append ``target_column``.

    Args:
        df: the rows to encode (the dev split when scoring a team).
        feature_names: the team's committed catalog names; non-empty and unique.
        catalog: from ``load_catalog``, already checked against the dataset.
        target_column: the 0/1 label column of ``df``, copied over unchanged.
        fit_df: the rows the data-dependent encodings are fitted on (see module
            docstring). Pass ``df`` itself when scoring on the dev split.

    Returns:
        The encoded feature columns in ``feature_names`` order (one column per feature,
        one per category for ``one_hot`` / ``bin``), then the target, on ``df``'s index.
        This is what ``scoring.cross_validate`` takes.

    Raises:
        AssertionError: no features, duplicate names, a name not in the catalog, a
            missing column, or two features encoding to the same column name.
    """
    return build_feature_frame_with_sources(df, feature_names, catalog, target_column,
                                            fit_df)[0]


@typechecked
def build_feature_frame_with_sources(df: pd.DataFrame, feature_names: List[str],
                                     catalog: Dict[str, CatalogEntry], target_column: str,
                                     fit_df: pd.DataFrame
                                     ) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """``build_feature_frame``, plus each encoded column's feature: ``{column: feature_name}``.

    A ``one_hot`` / ``bin`` feature has several columns, and their names cannot be
    traced back by prefix (one feature name may start another's), so the mapping is
    recorded as the frame is built. Same arguments and checks as ``build_feature_frame``.
    """
    assert feature_names, "no features to encode"
    duplicates = sorted({n for n in feature_names if feature_names.count(n) > 1})
    assert not duplicates, f"features committed more than once: {duplicates}"
    unknown = [n for n in feature_names if n not in catalog]
    assert not unknown, f"{len(unknown)} features are not in the catalog: {unknown[:5]}"
    assert target_column in df.columns, f"target column '{target_column}' is not in the dataset"

    columns = [encode_feature(catalog[name], df, fit_df) for name in feature_names]
    frame = pd.concat(columns + [df[target_column]], axis=1)
    clashes = frame.columns[frame.columns.duplicated()].tolist()
    assert not clashes, f"two features encode to the same column name: {clashes}"
    sources = {str(column): name
               for name, encoded in zip(feature_names, columns) for column in encoded.columns}
    return frame, sources


@typechecked
def encode_feature(entry: CatalogEntry, df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.DataFrame:
    """Encode one catalog feature from ``df`` as its entry says; columns fitted on ``fit_df``.

    One column named ``feature_name``, or for ``one_hot`` / ``bin`` one column per category
    named ``{feature_name}_{category}``.
    """
    for column in source_columns(entry):
        assert column in df.columns and column in fit_df.columns, \
            f"'{entry.feature_name}' needs column '{column}', which is not in the dataset"
        if entry.encoding in NUMERIC_ENCODINGS:
            assert pd.api.types.is_numeric_dtype(df[column]), \
                (f"'{entry.feature_name}' ({entry.encoding}) needs a numeric column; "
                 f"'{column}' is {df[column].dtype}")
    encoded = _ENCODERS[entry.encoding](entry, df, fit_df)
    if isinstance(encoded, pd.Series):
        encoded = encoded.rename(entry.feature_name).to_frame()
    return encoded


# --- one function per encoding --------------------------------------------------------
# Each takes (entry, df, fit_df) and returns a Series (named by the caller) or, for the
# multi-column encodings, a DataFrame with its own column names.

def _passthrough(entry: CatalogEntry, df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.Series:
    return df[entry.source_col]


def _aligned_dummies(series: pd.Series, fit_series: pd.Series, prefix: str) -> pd.DataFrame:
    """One-hot ``series`` with exactly the categories present in ``fit_series``.

    ``pd.get_dummies`` only emits the categories it sees, so encoding two frames
    separately would give them different columns. Reindexing onto the fit frame's columns
    makes a fit category absent from ``series`` an all-zero column and drops a category
    the fit frame never had (the model has no weight for it). A missing value is all zeros.
    """
    reference = pd.get_dummies(fit_series, prefix=prefix).columns
    dummies = pd.get_dummies(series, prefix=prefix)
    return dummies.reindex(columns=reference, fill_value=0).astype(int)


def _one_hot(entry: CatalogEntry, df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.DataFrame:
    return _aligned_dummies(df[entry.source_col], fit_df[entry.source_col],
                            prefix=entry.feature_name)


def _list_presence(entry: CatalogEntry, df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.Series:
    value = entry.encode_params["value"]
    return df[entry.source_col].map(lambda cell: int(value in parse_list_cell(cell)))


def _med_presence(entry: CatalogEntry, df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.Series:
    """1 when any of the patient's medication strings normalizes to the entry's drug."""
    drug = entry.encode_params["drug"]
    return df[entry.source_col].map(
        lambda cell: int(any(normalize_med(m) == drug for m in parse_list_cell(cell))))


def _lab_value(entry: CatalogEntry, df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.Series:
    """The patient's value for this lab; NaN when it was not measured.

    ``components`` are the raw lab names merged into this feature. The first position
    whose name is one of them gives the value. The run-start check guarantees the two lists
    line up and hold only numbers (Case A / B in ``Knowledge/catalog.py``); the assert here
    only guards against being handed a frame that was never checked.
    """
    params = entry.encode_params
    name_column, value_column = params["name_column"], params["value_column"]
    wanted = set(params["components"])

    def value_of(row: object, name_cell: object, value_cell: object) -> float:
        names, values = parse_list_cell(name_cell), parse_list_cell(value_cell)
        assert len(names) == len(values), \
            (f"Case A: row {row} has {len(names)} '{name_column}' and {len(values)} "
             f"'{value_column}' items; run assert_catalog_matches on this frame")
        return next((float(v) for n, v in zip(names, values) if n in wanted), np.nan)

    return pd.Series([value_of(row, n, v) for row, n, v
                      in zip(df.index, df[name_column], df[value_column])],
                     index=df.index, dtype=float)


def _bin(entry: CatalogEntry, df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.DataFrame:
    """One-hot of the bin each value falls in; the cut points come from the catalog."""
    params = entry.encode_params

    def cut(frame: pd.DataFrame) -> pd.Series:
        return pd.cut(frame[entry.source_col], bins=params["bins"],
                      labels=params.get("labels"), include_lowest=True)

    return _aligned_dummies(cut(df), cut(fit_df), prefix=entry.feature_name)


def _standardize(entry: CatalogEntry, df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.Series:
    """Centre and scale with ``fit_df``'s mean and std, never the encoded rows' own."""
    fit = fit_df[entry.source_col].astype(float)
    return (df[entry.source_col].astype(float) - fit.mean()) / (fit.std() + 1e-8)


def _log(entry: CatalogEntry, df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.Series:
    """``log1p`` of the column, which must have no negative values.

    ``log1p`` is undefined below -1 and meaningless for a negative count or measurement;
    clipping to 0 (what the old encoder did) would silently merge every negative value
    into "zero". A column with negatives needs a different encoding.
    """
    column = df[entry.source_col].astype(float)
    negative = column[column.lt(0)]
    assert negative.empty, \
        (f"'{entry.feature_name}' (log) needs non-negative values; '{entry.source_col}' has "
         f"{len(negative)}, e.g. {dict(list(negative.head(5).items()))}")
    return pd.Series(np.log1p(column.to_numpy()), index=column.index)


def _interaction(entry: CatalogEntry, df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.Series:
    first, second = entry.encode_params["columns"]
    return df[first].astype(float) * df[second].astype(float)


_ENCODERS: Dict[str, Callable[[CatalogEntry, pd.DataFrame, pd.DataFrame], object]] = {
    "passthrough": _passthrough,
    "one_hot": _one_hot,
    "list_presence": _list_presence,
    "med_presence": _med_presence,
    "lab_value": _lab_value,
    "bin": _bin,
    "standardize": _standardize,
    "log": _log,
    "interaction": _interaction,
}
assert set(_ENCODERS) == set(ENCODING_PARAMS), \
    f"encoders and catalog encodings differ: {set(_ENCODERS) ^ set(ENCODING_PARAMS)}"
