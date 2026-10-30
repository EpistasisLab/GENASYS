"""XGBoost scoring of a team's encoded feature frame.

A team is scored by **stratified k-fold cross-validation on the dev split**: every
reported number is the mean across folds, with its standard deviation alongside so a
score that only holds on one split is visible as such. The holdout reaches this module
only through ``score_holdout``, called once per Pareto-front team after evolution: one
model fitted on all of dev, scored once on the holdout.

The objective is ``roc_auc``: bounded in ``[0, 1]`` with a no-skill value of 0.5 that
does not move with the class balance, so scores compare across runs and datasets.
Everything else is reported for interpretation, never optimized:

- ``pr_auc``, the stricter ranking view on an imbalanced cohort. Its no-skill value is
  the positive rate, so ``positive_rate`` is reported next to it.
- accuracy / precision / recall / F1 / balanced accuracy at two thresholds. At ~12%
  prevalence a 0.5 cutoff predicts almost no positives, so the ``_at_prev`` variants
  cut at the positive rate instead; ``_at_50`` is kept because published numbers use it.

Feature encoding is fitted on the whole dev frame before this module is called, so
only the classifier is cross-validated. That admits a small transductive leak (each
validation fold contributed to its own standardize moments and one-hot categories);
the holdout score is the number without that caveat.

Nothing here decides a team's fate. Every input problem is either our bug (an
encoding that produced text or ``inf``) or a property of the dataset (too few
positives for the folds), so it raises rather than scoring the team 0. Whether a team
committed any features at all is decided by the evaluator before calling in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from typeguard import typechecked
from xgboost import XGBClassifier

# Metrics computed per fold and then averaged. One tuple so the aggregation and the
# ``_std`` companions cannot drift from what is computed.
FOLD_METRICS: Tuple[str, ...] = (
    "roc_auc", "pr_auc", "positive_rate", "threshold_prev",
    "accuracy_at_50", "precision_at_50", "recall_at_50",
    "f1_at_50", "balanced_accuracy_at_50",
    "accuracy_at_prev", "precision_at_prev", "recall_at_prev",
    "f1_at_prev", "balanced_accuracy_at_prev",
)
# Characters XGBoost refuses in a feature name.
XGBOOST_FORBIDDEN = ("[", "]", "<")


@dataclass(frozen=True)
class CVScore:
    """The result of one cross-validation.

    ``metrics`` holds the mean and ``_std`` of every ``FOLD_METRICS`` entry, plus
    ``n_folds`` and ``n_rows``. Values are unrounded; round only for display.
    ``feature_importances`` maps each feature column to its mean XGBoost importance
    across folds. ``fold_feature_importances`` holds one such map per fold, in fold
    order, so the spread of importances across folds is visible.
    """

    metrics: Dict[str, float]
    feature_importances: Dict[str, float]
    fold_feature_importances: List[Dict[str, float]]


@dataclass(frozen=True)
class HoldoutScore:
    """The result of the one holdout fit.

    ``metrics`` holds every ``FOLD_METRICS`` entry measured once on the holdout (no
    ``_std``: there is one model and one test set), plus ``n_train_rows`` (dev) and
    ``n_rows`` (holdout). Values are unrounded. ``feature_importances`` maps each
    feature column to its importance in the model fitted on all of dev.
    """

    metrics: Dict[str, float]
    feature_importances: Dict[str, float]


def _new_model(seed: int) -> XGBClassifier:
    """The one classifier configuration, so every fold and the holdout fit match.
    Using default hyperparameters except for the random seed, which is fixed for reproducibility."""
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=seed,
        eval_metric="logloss",
    )


def xgboost_columns(columns: List[str]) -> List[str]:
    """The names to fit XGBoost on: ``columns`` as they are, or ``f0, f1, ...`` when any
    has a character XGBoost refuses (``XGBOOST_FORBIDDEN``), as some catalog names do
    (e.g. 'DNA [FARR] ANTIBODY'). Scores do not depend on the names; importances are
    mapped back to ``columns`` by position either way.
    """
    if any(ch in c for c in columns for ch in XGBOOST_FORBIDDEN):
        return [f"f{i}" for i in range(len(columns))]
    return list(columns)


def _metrics_from_proba(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """The full metric suite for one set of held-out predictions.

    Label metrics are cut from ``y_proba`` at an explicit threshold rather than taken
    from ``model.predict()``, which would silently imply 0.5.
    """
    # Trapezoidal area under the precision-recall curve, not average_precision_score;
    # the two differ by a few percent, so only compare runs scored the same way.
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    positive_rate = float(y_true.mean())

    out: Dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(auc(recall, precision)),
        "positive_rate": positive_rate,
        "threshold_prev": positive_rate,
    }
    for suffix, threshold in (("50", 0.5), ("prev", positive_rate)):
        y_hat = (y_proba >= threshold).astype(int)
        # zero_division=0: at 0.5 on an imbalanced cohort the model can predict no
        # positives, leaving precision undefined.
        out[f"accuracy_at_{suffix}"] = float(accuracy_score(y_true, y_hat))
        out[f"precision_at_{suffix}"] = float(precision_score(y_true, y_hat, zero_division=0))
        out[f"recall_at_{suffix}"] = float(recall_score(y_true, y_hat, zero_division=0))
        out[f"f1_at_{suffix}"] = float(f1_score(y_true, y_hat, zero_division=0))
        out[f"balanced_accuracy_at_{suffix}"] = float(balanced_accuracy_score(y_true, y_hat))
    return out


def _aggregate(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean and population standard deviation of each metric across folds."""
    out: Dict[str, float] = {}
    for key in FOLD_METRICS:
        values = [m[key] for m in fold_metrics]
        out[key] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    return out


def assert_enough_per_class(y: np.ndarray, n_folds: int) -> None:
    """Fail unless ``y`` is all 0/1 with at least ``n_folds`` rows of each class.

    Stratified k-fold puts at least one row of each class in every validation fold.
    The EA calls this once at startup on the dev labels, so a dataset too small for
    ``cv_folds`` stops the run before any team is evaluated.
    """
    values = set(np.unique(y).tolist())
    assert values <= {0, 1}, f"labels must be 0/1: {sorted(values, key=str)[:5]}"
    counts = {label: int((y == label).sum()) for label in (0, 1)}
    assert min(counts.values()) >= n_folds, \
        f"{n_folds}-fold CV needs at least {n_folds} rows of each class: {counts}"


@typechecked
def cross_validate(frame: pd.DataFrame, target_column: str, n_folds: int,
                   seed: int) -> CVScore:
    """Stratified ``n_folds``-fold XGBoost cross-validation of ``frame``.

    Args:
        frame: the encoded feature columns plus the target column, built from the
            dev split. Every feature column must be numeric and finite or NaN.
        target_column: the 0/1 label column in ``frame``.
        n_folds: number of folds. Both classes need at least this many rows
            (``assert_enough_per_class``); it is never silently reduced.
        seed: the EA's ``seed``. Fixes the fold assignment and XGBoost's seed, so
            every team in a run is scored on the same folds and a re-run with the
            same seed reproduces every score.

    Raises:
        AssertionError: no feature columns, a target that is not 0/1, or too few rows
            of a class for ``n_folds``.
        Any XGBoost error, e.g. a text or ``inf`` column: an encoding bug, not a team
            outcome.
    """
    assert target_column in frame.columns, \
        f"target column '{target_column}' is not in the feature frame"
    feature_columns = [c for c in frame.columns if c != target_column]
    assert feature_columns, "the feature frame has no feature columns"

    y = frame[target_column].to_numpy()
    assert_enough_per_class(y, n_folds)

    # todo: @ag check to make sure this is even needed. Weird edge case found
    X = frame[feature_columns].set_axis(xgboost_columns(feature_columns), axis=1)

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics: List[Dict[str, float]] = []
    importances: List[np.ndarray] = []
    for train_idx, val_idx in splitter.split(X, y):
        model = _new_model(seed)
        model.fit(X.iloc[train_idx], y[train_idx], verbose=False)
        y_proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        fold_metrics.append(_metrics_from_proba(y[val_idx], y_proba))
        importances.append(np.asarray(model.feature_importances_, dtype=float))

    metrics = _aggregate(fold_metrics)
    metrics["n_folds"] = n_folds
    metrics["n_rows"] = len(y)
    mean_importance = np.mean(np.vstack(importances), axis=0)
    fold_importances = [{name: float(v) for name, v in zip(feature_columns, importance)}
                        for importance in importances]
    return CVScore(
        metrics=metrics,
        feature_importances={name: float(v)
                             for name, v in zip(feature_columns, mean_importance)},
        fold_feature_importances=fold_importances,
    )


@typechecked
def score_holdout(dev_frame: pd.DataFrame, holdout_frame: pd.DataFrame, target_column: str,
                  seed: int) -> HoldoutScore:
    """Fit one XGBoost model on all of ``dev_frame`` and score it once on ``holdout_frame``.

    Args:
        dev_frame: the team's encoded features plus the target, built from the dev split.
        holdout_frame: the same features built from the holdout with ``fit_df`` set to
            the dev split, so its columns and scaling are the ones the model trains on.
        target_column: the 0/1 label column of both frames.
        seed: the EA's ``seed``, the same XGBoost seed cross-validation used.

    Raises:
        AssertionError: the frames' columns differ, there are no feature columns, or
            either frame's labels are not both 0 and 1.
        Any XGBoost error: an encoding bug, as in ``cross_validate``.
    """
    assert target_column in dev_frame.columns, \
        f"target column '{target_column}' is not in the dev feature frame"
    # build_feature_frame fits the categories on dev for both frames, so a difference
    # here is a bug; reindexing would silently shift the columns the model reads.
    assert list(holdout_frame.columns) == list(dev_frame.columns), \
        (f"the holdout frame's columns differ from the dev frame's: "
         f"{list(holdout_frame.columns)} vs {list(dev_frame.columns)}")
    feature_columns = [c for c in dev_frame.columns if c != target_column]
    assert feature_columns, "the feature frame has no feature columns"

    y_dev = dev_frame[target_column].to_numpy()
    y_holdout = holdout_frame[target_column].to_numpy()
    for name, y in (("dev", y_dev), ("holdout", y_holdout)):
        labels = sorted(np.unique(y).tolist())
        assert labels == [0, 1], f"the {name} labels must be both 0 and 1: {labels}"

    names = xgboost_columns(feature_columns)
    model = _new_model(seed)
    model.fit(dev_frame[feature_columns].set_axis(names, axis=1), y_dev, verbose=False)
    y_proba = model.predict_proba(holdout_frame[feature_columns].set_axis(names, axis=1))[:, 1]

    metrics = _metrics_from_proba(y_holdout, y_proba)
    metrics["n_train_rows"] = len(y_dev)
    metrics["n_rows"] = len(y_holdout)
    return HoldoutScore(
        metrics=metrics,
        feature_importances={name: float(v) for name, v
                             in zip(feature_columns, model.feature_importances_)},
    )
