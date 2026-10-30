"""The record of one evaluation of one team, and the numbers its logs report.

Everything a log, a CSV or the archive shows about an evaluation is computed here, so
``Reporting/`` only formats and writes. Nothing in this module does I/O.

A team is scored only when its run FINALIZED with at least one feature (see
``evaluator.py``); every other evaluation has no ``score`` and a performance of 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from typeguard import typechecked

from ..Knowledge.catalog import CatalogEntry
from .outcome import FeatureRecord, RunOutcome, RunStatus
from .scoring import CVScore
from .usage import RunCost

# column_kind values: a 0/1 column, or anything else.
BINARY = "binary"
CONTINUOUS = "continuous"


@dataclass(frozen=True)
class FeatureColumn:
    """One encoded column of a scored team's feature frame (a row of a feature_*_details.csv).

    A ``one_hot`` / ``bin`` feature has one column per category, all sharing the
    ``feature_name``; that is the level XGBoost importances exist at. ``prevalence`` is
    the fraction of rows equal to 1 for a binary column and None for a continuous one,
    where ``non_null_frac`` carries the "was it measured" signal instead.
    ``xgb_importance`` and ``used_by_model`` come from the importances the row was built
    with: the mean across folds, or one fold's model.
    """

    feature_name: str
    encoded_column: str
    encoding: str
    feature_type: str
    similarity_score: float
    added_by: str
    suggested_by: str
    column_kind: str
    prevalence: Optional[float]
    non_null_frac: float
    xgb_importance: float
    used_by_model: bool


@typechecked
def feature_columns(frame: pd.DataFrame, target_column: str, sources: Dict[str, str],
                    records: List[FeatureRecord], catalog: Dict[str, CatalogEntry],
                    importances: Dict[str, float]) -> List[FeatureColumn]:
    """One ``FeatureColumn`` per encoded column of ``frame``, in frame order.

    Args:
        frame: the feature frame the team was scored on (target included).
        sources: ``{encoded column: feature_name}`` from ``build_feature_frame_with_sources``.
        records: the team's committed features.
        importances: ``{encoded column: XGBoost importance}``, from ``CVScore``: either
            ``feature_importances`` (the mean across folds) or one entry of
            ``fold_feature_importances``.
    """
    by_name = {r.feature_name: r for r in records}
    columns = [c for c in frame.columns if c != target_column]
    assert set(map(str, columns)) == set(sources), \
        "the column sources do not match the frame's feature columns"
    assert set(sources.values()) == set(by_name), \
        "the frame's features do not match the committed records"
    out: List[FeatureColumn] = []
    for column in columns:
        name = sources[str(column)]
        record = by_name[name]
        series = frame[column]
        values = series.dropna().to_numpy(dtype=float)
        binary = values.size > 0 and bool(np.isin(values, (0.0, 1.0)).all())
        importance = importances[column]
        out.append(FeatureColumn(
            feature_name=name, encoded_column=str(column),
            encoding=catalog[name].encoding, feature_type=record.feature_type,
            similarity_score=record.similarity_score, added_by=record.added_by,
            suggested_by=record.suggested_by,
            column_kind=BINARY if binary else CONTINUOUS,
            prevalence=float((series == 1).mean()) if binary else None,
            non_null_frac=float(series.notna().mean()),
            xgb_importance=float(importance), used_by_model=importance > 0))
    return out


@dataclass(frozen=True)
class EvalResult:
    """One evaluation of one team: how its run ended, its score and what it cost.

    ``attempts`` counts every run of the team in this evaluation; all but the last were
    lost to the provider, and ``provider_failures`` holds their one-line errors, oldest
    first. Everything else describes the last attempt. ``score``, ``columns`` and
    ``fold_columns`` are empty unless that attempt was scored. ``columns`` carries the
    mean importances; ``fold_columns[k]`` carries the same columns with fold ``k + 1``'s.
    ``traceback`` is the last attempt's, if its graph raised.
    """

    team_id: int
    generation: int
    eval_index: int
    attempts: int
    provider_failures: List[str]
    outcome: RunOutcome
    score: Optional[CVScore]
    cost: RunCost
    eval_seconds: float
    traceback: Optional[str]
    columns: List[FeatureColumn]
    fold_columns: List[List[FeatureColumn]]

    def __post_init__(self) -> None:
        assert self.attempts == len(self.provider_failures) + 1, \
            f"{self.attempts} attempts but {len(self.provider_failures)} earlier failures"
        assert (self.score is not None) == (self.outcome.status is RunStatus.FINALIZED
                                            and bool(self.outcome.features)), \
            "a result is scored exactly when its run finalized with features"
        assert self.score is not None or not self.columns, "an unscored result has no columns"
        assert self.score is not None or not self.fold_columns, \
            "an unscored result has no fold columns"

    @property
    def performance(self) -> float:
        """Mean cross-validated ROC-AUC on the dev split; 0.0 when not scored."""
        return float(self.score.metrics["roc_auc"]) if self.score is not None else 0.0

    @property
    def token_count(self) -> int:
        return self.cost.usage.total_tokens

    def team_kwargs(self) -> Dict[str, Any]:
        """The arguments of ``Team.set_evaluation_result`` for this evaluation."""
        return {"performance": self.performance,
                "token_count": self.token_count,
                "eval_seconds": self.eval_seconds,
                "metrics": dict(self.score.metrics) if self.score is not None else {},
                "features": [asdict(r) for r in self.outcome.features],
                "cost": self.cost.total_usd}
