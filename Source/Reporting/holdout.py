"""The final holdout evaluation: one entry per Pareto-front team, written to
holdout_results.json and printed to stdout (the SLURM log).

The EA scores every team (EA.holdout_evaluation); this module only formats and writes
the results, so the console and the JSON always show the same values. Errors are not
caught here.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from typeguard import typechecked

from ..Evaluation.scoring import HoldoutScore
from ..Optimizer.objectives import (OBJECTIVE_2_LABELS, OBJECTIVE_COST_USD, PERFORMANCE_LABEL,
                                    format_objective_2)

HOLDOUT_RESULTS = "holdout_results.json"

# Cost in the console carries more places than on the plots, as in the generation summary.
_COST_DECIMALS = 4


@dataclass(frozen=True)
class HoldoutResult:
    """One front team's holdout score, next to the dev evaluation it was rebuilt from.

    ``evaluation`` is the team's median evaluation (``Team.median_evaluation``):
    ``features`` and ``dev_metrics`` are that evaluation's, so ``dev_metrics["roc_auc"]``
    is the cross-validated score of exactly the feature set the holdout scored.
    ``median_*`` are the team's medians across all its evaluations, the values NSGA-II
    ranked it on.
    """

    team_id: int
    evaluation: int
    median_roc_auc: float
    median_token_count: int
    median_cost_usd: float
    dev_metrics: Dict[str, Any]
    features: List[Dict[str, Any]]
    score: HoldoutScore

    @property
    def roc_auc_gap(self) -> float:
        """Holdout minus dev ROC-AUC: how much of the selected score held up on unseen rows."""
        return float(self.score.metrics["roc_auc"]) - float(self.dev_metrics["roc_auc"])

    def to_dict(self) -> Dict[str, Any]:
        return {"team_id": self.team_id,
                "evaluation": self.evaluation,
                "median_roc_auc": self.median_roc_auc,
                "median_token_count": self.median_token_count,
                "median_cost_usd": self.median_cost_usd,
                "roc_auc_gap": self.roc_auc_gap,
                "dev_metrics": self.dev_metrics,
                "holdout_metrics": self.score.metrics,
                "holdout_feature_importances": self.score.feature_importances,
                "features": self.features}


@typechecked
def write_holdout_results(path: str, task_name: str, objective_2: str, seed: int,
                          n_dev_rows: int, n_holdout_rows: int,
                          results: List[HoldoutResult]) -> None:
    """Write every front team's result to ``path``, in front order (cheapest first).

    ``team_id`` is the team's id in ``archive.json``, which holds its roster.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"task_name": task_name,
               "objective_2": objective_2,
               "seed": seed,
               "n_dev_rows": n_dev_rows,
               "n_holdout_rows": n_holdout_rows,
               "teams": [r.to_dict() for r in results]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return


@typechecked
def print_holdout_summary(results: List[HoldoutResult], objective_2: str,
                          n_holdout_rows: int, path: str) -> None:
    """Print the final holdout block: dev and holdout ROC-AUC side by side, with the gap.

    The dev column is the cross-validated score of the evaluation whose features were
    scored, so the gap compares the same feature set on the two splits.
    """
    def objective_2_text(r: HoldoutResult) -> str:
        value = r.median_cost_usd if objective_2 == OBJECTIVE_COST_USD else r.median_token_count
        return format_objective_2(float(value), objective_2, _COST_DECIMALS)

    lines = ["=" * 80,
             f"  FINAL HOLDOUT EVALUATION: Pareto front on the {n_holdout_rows}-row holdout",
             "=" * 80,
             f"  {'team':>5}  {OBJECTIVE_2_LABELS[objective_2]:>12}  {'feats':>5}  {'dev ' + PERFORMANCE_LABEL:>12}  "
             f"{'holdout':>8}  {'gap':>7}  {'PR-AUC':>7}  {'F1@prev':>7}"]
    for r in results:
        m = r.score.metrics
        lines.append(f"  {r.team_id:>5}  {objective_2_text(r):>12}  {len(r.features):>5}  "
                     f"{float(r.dev_metrics['roc_auc']):>12.4f}  {float(m['roc_auc']):>8.4f}  "
                     f"{r.roc_auc_gap:>+7.4f}  {float(m['pr_auc']):>7.4f}  "
                     f"{float(m['f1_at_prev']):>7.4f}")
    lines += [f"  results             : {path}", "=" * 80]
    print("\n" + "\n".join(lines) + "\n", flush=True)
    return
