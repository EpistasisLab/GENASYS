"""The feature catalog: every concrete feature an agent can commit, and how to encode it.

The catalog is built offline from the cohort, per dataset, and its path is handed to the
EA. Each ``CatalogEntry`` pairs a verbatim ``feature_name`` (the key agents commit and the
semantic matcher resolves concepts to) with the encoding that turns it into numbers. ``features.py``
only applies what the catalog says; it never guesses an encoding.

This module is read by both sides of a run: the agent tools that search the catalog and the
feature encoder. ``parse_list_cell`` and ``normalize_med`` live here, once, because the
catalog was built with them and the encoder must split cells and normalize drug names the
same way, or a catalog feature would silently encode as all zeros.

Everything is checked up front, so a bad catalog or cohort stops the run at startup rather
than crashing the evaluation of a team mid-run (after its LLM calls were paid for):

- ``load_catalog`` checks the catalog itself: unique names, a known encoding, and the
  parameters that encoding needs.
- ``assert_catalog_matches`` checks the catalog against the dataset: every source column
  exists, and the lab columns are well formed (see below).

Lab values
----------
A patient's labs are two pipe-delimited columns aligned by position::

    GENERAL_LAB_COMPONENTS_LIST:  ABSOLUTE BASOS YAABA | ABSOLUTE EOS YHAEO | ABSOLUTE LYMPHS
    GENERAL_LAB_VALUES_LIST:      43                   | 78                 | 2823

The value of a lab is the item at the same position as its name. Two problems can break
that, and ``assert_catalog_matches`` fails on both rather than turning them into NaN (which
is what the old encoder did, silently, making the patient look unmeasured):

Case A, the two lists have different lengths in a row. Then no value can be trusted to
belong to its name: every value after the gap may be shifted onto the wrong lab. It arises
when the cohort extract builds the two columns separately and they drift apart, e.g.

- an aggregation that skips NULL values but keeps their names (``LISTAGG`` / ``STRING_AGG``
  drop NULLs), or one that orders the two lists differently;
- an empty item (``43||78``), which ``parse_list_cell`` drops, shifting what follows;
- a ``|`` inside a lab name or value, splitting one item into two;
- a cell truncated on export (e.g. Excel's 32,767-character limit) or one column empty
  where the other is not.

There is no right answer to recover, so the cohort has to be fixed at its source.

Case B, a value is not a finite number. Lab systems report more than numbers:

- censored results below or above the detection limit, e.g. ``<20``, ``<0.5``, ``>1000``;
- qualitative results, e.g. ``POSITIVE``, ``NEGATIVE``, ``DETECTED``;
- free text, e.g. ``see note``, ``cancelled``, ``hemolyzed``;
- a unit glued to the number, e.g. ``5.2 mg/dL``, or the literal text ``nan`` / ``inf``.

Dropping these would be wrong, not just lossy: ``<20`` is a real, very low measurement, and
NaN would make it read as "not measured". How to represent them (``<20`` → ``20``, a
separate below-limit flag column, a qualitative lab as its own 0/1 column) is a modelling
decision for whoever prepares the cohort, so the run fails and names the rows instead of
choosing one silently.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd
from typeguard import typechecked

# Every encoding the encoder implements, with the ``encode_params`` keys it requires. The
# offline builder only produces the first five; the rest are kept so a catalog can use
# them later, and are validated the same way if it does.
ENCODING_PARAMS: Dict[str, Tuple[str, ...]] = {
    "passthrough": (),                   # numeric column as is
    "one_hot": (),                       # categorical column -> one 0/1 column per category
    "list_presence": ("value",),         # 0/1: is ``value`` in the patient's list cell
    "med_presence": ("drug",),           # 0/1: does any dose/form of ``drug`` appear
    "lab_value": ("name_column", "value_column", "components"),  # the lab's numeric value
    "bin": ("bins",),                    # numeric column cut at ``bins`` -> one-hot of bins
    "standardize": (),                   # numeric column -> mean 0, std 1
    "log": (),                           # numeric column -> log1p
    "interaction": ("columns",),         # product of the two numeric ``columns``
}

# Encodings that do arithmetic on their source columns, which must therefore be numeric.
NUMERIC_ENCODINGS = frozenset({"passthrough", "bin", "standardize", "log", "interaction"})

# How many offending rows / features an error message lists.
_EXAMPLES = 5


@dataclass(frozen=True)
class CatalogEntry:
    """One feature an agent can commit.

    ``feature_name`` is the verbatim key, e.g. ``"E11.9 | Type 2 diabetes"``.
    ``feature_type`` is one of ICD-10, procedure, medication, lab, vital, demographic.
    ``source_col`` is the column the value lives in; ``match_text`` is the clinical text
    the semantic matcher embeds. ``prevalence`` and ``count`` are the fraction and number
    of patients with the feature present in the cohort the catalog was built from.
    ``encoding`` and ``encode_params`` say how ``features.py`` encodes it.
    """

    feature_name: str
    feature_type: str
    source_col: str
    match_text: str
    prevalence: float
    count: int
    encoding: str
    encode_params: Dict[str, Any] = field(default_factory=dict)


def source_columns(entry: CatalogEntry) -> List[str]:
    """The dataset columns ``entry`` is encoded from."""
    if entry.encoding == "lab_value":
        return [entry.encode_params["name_column"], entry.encode_params["value_column"]]
    if entry.encoding == "interaction":
        return list(entry.encode_params["columns"])
    return [entry.source_col]


def _check_entry(entry: CatalogEntry) -> None:
    """Fail unless ``entry`` names a known encoding and carries the params it needs."""
    name = entry.feature_name
    assert name.strip(), f"catalog entry with an empty feature_name: {entry}"
    assert entry.source_col.strip(), f"catalog entry '{name}' has an empty source_col"
    assert entry.encoding in ENCODING_PARAMS, \
        (f"catalog entry '{name}' has unknown encoding '{entry.encoding}'; "
         f"expected one of {sorted(ENCODING_PARAMS)}")
    missing = [p for p in ENCODING_PARAMS[entry.encoding] if not entry.encode_params.get(p)]
    assert not missing, \
        f"catalog entry '{name}' ({entry.encoding}) is missing encode_params {missing}"
    if entry.encoding == "interaction":
        columns = entry.encode_params["columns"]
        assert isinstance(columns, list) and len(columns) == 2, \
            f"catalog entry '{name}' (interaction) needs exactly two columns: {columns}"


@typechecked
def load_catalog(path: str) -> Dict[str, CatalogEntry]:
    """Load and check the catalog at ``path``, keyed by ``feature_name``.

    Every entry must have every ``CatalogEntry`` field (no defaults are filled in), a
    known encoding and that encoding's ``encode_params``, and names must be unique.

    Args:
        path: the catalog JSON written by the offline builder.

    Raises:
        AssertionError: missing file, duplicate names, or an invalid entry.
        TypeError: an entry with a missing or unexpected field.
    """
    assert os.path.isfile(path), f"feature catalog not found: {path}"
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, list) and payload, \
        f"feature catalog {path} must be a non-empty JSON list of entries"

    catalog: Dict[str, CatalogEntry] = {}
    duplicates: List[str] = []
    for record in payload:
        entry = CatalogEntry(**record)
        _check_entry(entry)
        if entry.feature_name in catalog:
            duplicates.append(entry.feature_name)
        catalog[entry.feature_name] = entry
    assert not duplicates, \
        f"feature catalog {path} has {len(duplicates)} duplicate names, e.g. {duplicates[:_EXAMPLES]}"
    return catalog


@typechecked
def assert_catalog_matches(catalog: Dict[str, CatalogEntry], df: pd.DataFrame) -> None:
    """Fail unless every catalog feature can be encoded from ``df``.

    The EA calls this once at startup. It checks that every source column the catalog
    refers to is in ``df``, that the columns of ``NUMERIC_ENCODINGS`` features are numeric
    (a text column would otherwise only fail when a team commits that feature), and that
    each lab name/value column pair passes the Case A and Case B checks described in the
    module docstring.

    Args:
        catalog: from ``load_catalog``.
        df: the dataset the features will be encoded from.
    """
    needed: Dict[str, List[str]] = {}
    lab_pairs = set()
    for entry in catalog.values():
        for column in source_columns(entry):
            needed.setdefault(column, []).append(entry.feature_name)
        if entry.encoding == "lab_value":
            lab_pairs.add((entry.encode_params["name_column"],
                           entry.encode_params["value_column"]))

    missing = {column: names for column, names in needed.items() if column not in df.columns}
    assert not missing, \
        ("the feature catalog does not match the dataset; columns missing from the dataset "
         "(with features that need them): "
         + "; ".join(f"'{c}' ({len(n)}, e.g. {n[:3]})" for c, n in sorted(missing.items())))

    not_numeric: Dict[str, List[str]] = {}
    for entry in catalog.values():
        if entry.encoding in NUMERIC_ENCODINGS:
            for column in source_columns(entry):
                if not pd.api.types.is_numeric_dtype(df[column]):
                    not_numeric.setdefault(column, []).append(entry.feature_name)
    assert not not_numeric, \
        ("these columns must be numeric for the catalog encodings that use them: "
         + "; ".join(f"'{c}' is {df[c].dtype} (e.g. {n[:3]})"
                     for c, n in sorted(not_numeric.items())))

    for name_column, value_column in sorted(lab_pairs):
        _assert_lab_columns(df, name_column, value_column)


def _assert_lab_columns(df: pd.DataFrame, name_column: str, value_column: str) -> None:
    """Fail on any row of ``df`` where the lab lists misalign (A) or a value is not a number (B)."""
    misaligned: List[str] = []
    not_numeric: List[str] = []
    n_misaligned = n_not_numeric = 0
    for row, name_cell, value_cell in zip(df.index, df[name_column], df[value_column]):
        names, values = parse_list_cell(name_cell), parse_list_cell(value_cell)
        if len(names) != len(values):
            n_misaligned += 1
            if len(misaligned) < _EXAMPLES:
                misaligned.append(f"row {row}: {len(names)} names, {len(values)} values")
            continue
        for lab, value in zip(names, values):
            if not _is_finite_number(value):
                n_not_numeric += 1
                if len(not_numeric) < _EXAMPLES:
                    not_numeric.append(f"row {row}: {lab} = {value!r}")

    assert not n_misaligned, \
        (f"Case A: '{name_column}' and '{value_column}' have different lengths in "
         f"{n_misaligned} rows, so lab values cannot be matched to their names; fix the "
         f"cohort extract. e.g. {misaligned}")
    assert not n_not_numeric, \
        (f"Case B: '{value_column}' has {n_not_numeric} values that are not finite numbers; "
         f"decide how to represent them (e.g. '<20' -> 20) when preparing the cohort. "
         f"e.g. {not_numeric}")


def _is_finite_number(text: str) -> bool:
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def parse_list_cell(cell: Any) -> List[str]:
    """Split one pipe-delimited list cell into its stripped, non-empty items.

    An empty cell (NaN) is an empty list. Only ``|`` separates items: commas occur inside
    names (``HDL CHOLESTEROL, DIRECT``), so they never do.
    """
    if pd.isna(cell):
        return []
    return [item.strip() for item in str(cell).split("|") if item.strip()]


# Dose, form, route and unit tokens dropped when normalizing a medication string.
_MED_FORM_TOKENS = {
    "po", "iv", "in", "im", "sc", "sq", "tp", "pr", "sl", "na", "misc",
    "tab", "tabs", "tablet", "tablets", "cap", "caps", "capsule", "capsules",
    "oral", "soln", "solr", "solution", "susp", "suspension", "injection", "inj",
    "cream", "ointment", "patch", "spray", "drop", "drops", "liqd", "liquid",
    "syrg", "syringe", "cpsr", "tbsr", "ptmd", "csmc", "pca", "er", "xr", "sr",
    "mg", "ml", "mcg", "unit", "units", "gm", "g", "meq", "gram", "hcl", "%",
}


def normalize_med(name: str) -> str:
    """Reduce a dose-level medication string to its ingredient(s), keeping combinations.

    Strips dose, form and route but keeps every ingredient in order, so combination
    products stay distinct:

    - ``"ATENOLOL 25 MG PO TABS"`` -> ``"atenolol"``
    - ``"diphenhydrAMINE (BENADRYL) capsule"`` -> ``"diphenhydramine"``
    - ``"EMTRICITABINE 200 MG-TENOFOVIR ALAFENAMIDE FUMARATE 25 MG TABLET"``
      -> ``"emtricitabine tenofovir alafenamide fumarate"``

    Tokens starting with a digit (doses) are dropped; name tokens containing a digit
    (``b12``, ``d3``) are kept.
    """
    text = re.sub(r"\([^)]*\)", " ", str(name).lower())   # parentheticals (brands, codes)
    tokens = re.split(r"[\s\-/,]+", text)
    keep = [t for t in tokens if t and not t[0].isdigit() and t not in _MED_FORM_TOKENS]
    return " ".join(keep).strip()