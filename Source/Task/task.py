"""The prediction task: one CSV, its target, and its dev / holdout split.

The EA builds one Task at construction and holds it for the run. The CSV is read
and split once, there; every later reader (the per-generation evaluation, the
final holdout stage, Ray workers via ``ray.put``) shares the same two frames,
so they can never disagree about which rows are held out.

``get_dataframe`` returns the **dev** frame, so everything downstream of a Task —
including ``get_schema_description`` — is confined to the part of the data the
search is allowed to see. The holdout is reachable only through the separately
named ``get_holdout_dataframe``, called once after evolution finishes.

``render_schema_description`` renders the leads' ``print_dataset_schema`` tool
(``Agents/Tools/dataset_tools.py``) and ``get_schema_description``. It is a
module-level function rather than only a Task method because the agents never hold a
Task; they hold the dev frame.
"""

from __future__ import annotations

import os

import pandas as pd
from typeguard import typechecked

from .data_split import split_dev_holdout

EXAMPLE_VALUE_CHARS = 120


def _clip(value: str, limit: int = EXAMPLE_VALUE_CHARS) -> str:
    """Shorten one cell to ``limit`` characters, marking that it was cut."""
    # pandas >= 3.0 keeps NaN as a float through .astype(str), so a cell can
    # arrive here as a non-string; coerce defensively before string ops.
    value = str(value).replace("\n", " ")
    return value if len(value) <= limit else value[:limit] + " …(truncated)"


def render_schema_description(df: pd.DataFrame, target_column: str,
                              task_name: str = "") -> str:
    """Describe ``df`` as name/dtype/example per column plus three sample rows.

    Always pass the DEV frame. The target column is omitted: an agent that saw it
    listed as an available column would reasonably propose it as a feature.
    ``task_name`` is optional because the agent-facing caller has the working
    dataset but no Task to read it off.

    Example values are clipped. One EHR list column (``DX_NAME_LIST``) holds a
    pipe-delimited history thousands of characters long, and printing three of them
    in full made this ~10k tokens — which, returned from the Supervisor's tool, would
    sit in the message history for the rest of the run. The point of the sample is to
    show the *shape* of a value, and a clipped one still does that.
    """
    lines = []
    if task_name:
        lines.append(f"Task name: {task_name}")
    lines += [
        f"Sample size: {len(df)} (development set; a held-out test set is "
        f"reserved and is not available here)",
        "",
        "Features:",
    ]

    # For each column, show name, dtype, and a sample value.
    for col in df.columns:
        if col == target_column:
            continue
        dtype = str(df[col].dtype)
        sample = _clip(str(df[col].iloc[0])) if len(df) else ""
        lines.append(f"  - {col} ({dtype}) [example: {sample}]")

    lines.append("")
    lines.append("Sample rows (first 3):")
    # Drop the target here too. Listing it under "Features:" would invite an agent
    # to propose it; printing three of its values in the sample rows would do the
    # same thing less obviously.
    sample_cols = [c for c in df.columns if c != target_column]
    sample_rows = df.head(3)[sample_cols].astype(str).map(lambda v: _clip(v, 40))
    lines.append(sample_rows.to_string(index=False))

    return "\n".join(lines)


def _binary_target(target: pd.Series) -> pd.Series:
    """Return ``target`` as int 0/1; fail if any value is NaN, text or not 0/1.

    The run hard-fails here rather than dropping bad rows: a label that is not a
    clean 0/1 means the CSV is not ready for prediction, and silently scoring on
    fewer rows would hide that. The error names the first offending rows (by CSV
    row index) so they can be found and fixed. Both classes must be present.
    """
    name = target.name
    nan_rows = target.index[target.isna()].tolist()
    assert not nan_rows, \
        f"target column '{name}' has {len(nan_rows)} empty/NaN values, e.g. rows {nan_rows[:5]}"
    numeric = pd.to_numeric(target, errors="coerce")
    bad = target[numeric.isna() | ~numeric.isin([0, 1])]
    assert bad.empty, \
        (f"target column '{name}' must contain only 0/1; {len(bad)} rows do not, e.g. "
         f"{dict(list(bad.head(5).items()))}")
    labels = numeric.astype(int)
    assert labels.nunique() == 2, \
        f"target column '{name}' has a single class: {sorted(labels.unique())}"
    return labels


@typechecked
class Task:
    """A prediction task: the CSV at ``csv_path``, split once into dev / holdout."""

    def __init__(self, task_name: str, csv_path: str, target_column: str,
                 holdout_fraction: float, split_seed: int) -> None:
        """Read ``csv_path`` and split it.

        Fails if the file or target column is missing, if the target is not a
        clean 0/1 label with both classes present, or if the holdout gets only one class.

        Args:
            task_name: name of the task, used in the schema description.
            csv_path: path to the cohort CSV.
            target_column: column to predict; must be in the CSV and only 0/1.
            holdout_fraction: proportion of rows held out, in (0, 1).
            split_seed: seed for the split, independent of the EA's seed.
        """
        assert task_name.strip(), "task_name cannot be empty."
        assert target_column.strip(), "target_column cannot be empty."
        assert os.path.isfile(csv_path), f"CSV file not found: {csv_path}"
        assert csv_path.lower().endswith(".csv"), f"Not a .csv file: {csv_path}"

        self.task_name = task_name
        self.csv_path = csv_path
        self.target_column = target_column
        self.holdout_fraction = holdout_fraction
        self.split_seed = split_seed

        df = pd.read_csv(csv_path)
        assert target_column in df.columns, \
            f"target column '{target_column}' is not in the dataset"
        df[target_column] = _binary_target(df[target_column])
        self._dev, self._holdout = split_dev_holdout(
            df, target_column=target_column,
            holdout_fraction=holdout_fraction, seed=split_seed)
        # The holdout is scored once, with ROC-AUC, after evolution. A one-class holdout
        # (a tiny cohort or a rare outcome) would only fail then, after the whole run.
        holdout_classes = sorted(self._holdout[target_column].unique().tolist())
        assert holdout_classes == [0, 1], \
            (f"the {len(self._holdout)}-row holdout has a single class {holdout_classes}; "
             f"raise holdout_fraction or use a larger cohort")

    def get_dataframe(self) -> pd.DataFrame:
        """Return the DEV frame — the part of the data the search is allowed to see."""
        return self._dev

    def get_holdout_dataframe(self) -> pd.DataFrame:
        """Return the untouched holdout frame.

        Deliberately not named ``get_dataframe``-anything: every agent-facing
        caller reaches for that one, and this must stay a call a human had to
        write on purpose. Only the post-evolution holdout stage uses it.
        """
        return self._holdout

    def get_schema_description(self) -> str:
        """Return a human-readable description of the dev frame's schema."""
        # Dev frame only: the dtypes, example values and sample rows below are
        # shown to the agents through print_dataset_schema, so sourcing them from
        # the full CSV would leak the holdout's contents into the feature search.
        return render_schema_description(self._dev, target_column=self.target_column,
                                         task_name=self.task_name)
