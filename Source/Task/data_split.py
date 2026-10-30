"""Deterministic dev / holdout split of the raw cohort.

The dataset is cut once, up front, into a **dev** frame and a **holdout** frame.
Everything GENASYS does during evolution — the schema description the agents
read, every statistician tool, feature building and the cross-validated XGBoost
score — sees the dev frame only. The holdout is touched exactly once: after
evolution finishes, to score the Pareto-front teams.

That separation is the whole point. If the holdout were visible to the agents,
the final number would be reporting on data the feature search had already been
steered by, and it would no longer estimate performance on unseen patients.

The split is a pure function of ``(row count, target values, fraction, seed)``,
so every process that calls it — the driver, a Ray worker, a later re-run of the
same command — reconstructs the identical partition without having to share it.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from typeguard import typechecked

SPLIT_MANIFEST = "data_split.json"


@typechecked
def split_dev_holdout(df: pd.DataFrame, target_column: str, holdout_fraction: float,
                      seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into ``(dev, holdout)``, stratified on the target.

    Args:
        df: the full raw cohort, exactly as read from the CSV.
        target_column: column to stratify on; must be in ``df`` and already
            validated as 0/1 (``Task`` does this).
        holdout_fraction: proportion held out. Must be strictly between 0 and 1 —
            a 0 holdout would make the final evaluation vacuous and a 1 holdout
            would leave nothing to evolve on.
        seed: fixed split seed, independent of the EA's ``--seed`` so that
            changing the evolutionary seed does not change which patients are
            held out and make two runs incomparable.

    Returns:
        ``(dev, holdout)``. Both keep the original row index, so a row can be
        traced back to its line in the source CSV.
    """
    assert target_column in df.columns, \
        f"target column '{target_column}' is not in the dataset"
    assert 0.0 < holdout_fraction < 1.0, \
        f"holdout_fraction must be strictly between 0 and 1: {holdout_fraction}"

    dev, holdout = train_test_split(
        df,
        test_size=holdout_fraction,
        random_state=seed,
        shuffle=True,
        stratify=df[target_column],
    )
    return dev, holdout


def _positive_rate(df: pd.DataFrame, target_column: str) -> float:
    """Fraction of rows labelled 1."""
    return float(df[target_column].mean())


def describe_split(dev: pd.DataFrame, holdout: pd.DataFrame, target_column: str) -> str:
    """One-line summary for the run log."""
    total = len(dev) + len(holdout)
    pct = (100.0 * len(holdout) / total) if total else 0.0
    return (f"dev {len(dev)} rows (positive rate {_positive_rate(dev, target_column):.4f}) | "
            f"holdout {len(holdout)} rows ({pct:.1f}%, positive rate "
            f"{_positive_rate(holdout, target_column):.4f}) — holdout unseen until the "
            f"final evaluation")


@typechecked
def write_split_manifest(out_dir: str, dev: pd.DataFrame, holdout: pd.DataFrame,
                         target_column: str, holdout_fraction: float, seed: int,
                         csv_path: str) -> str:
    """Record which rows were held out in ``out_dir``; returns the file's path.

    Written so the claim "-GP never saw these rows" is checkable after the
    fact rather than taken on trust, and so a later analysis can reproduce the
    exact partition without re-deriving it.
    """
    path = os.path.join(out_dir, SPLIT_MANIFEST)
    manifest: Dict[str, Any] = {
        "csv_path": csv_path,
        "target_column": target_column,
        "holdout_fraction": holdout_fraction,
        "split_seed": seed,
        "n_total": len(dev) + len(holdout),
        "n_dev": len(dev),
        "n_holdout": len(holdout),
        "dev_positive_rate": _positive_rate(dev, target_column),
        "holdout_positive_rate": _positive_rate(holdout, target_column),
        "holdout_row_index": [int(i) for i in holdout.index],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path
