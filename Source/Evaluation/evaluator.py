"""Evaluate one team: run its graph, score what it committed, price what it spent.

One evaluation of a team is:

1. ``team.txt`` is written before anything runs, so the team is on disk even if the run
   later crashes.
2. The team's graph runs from its kickoff message (``invoke_graph``). A run the model
   provider stopped (``RunStatus.PROVIDER``) says nothing about the team, so it is run
   again after ``team_retry_delay_secs``, up to ``team_retry_attempts`` more times. Each
   lost attempt keeps its own ``Provider_Failures/Attempt_{k}/run_history.log``.
3. The last attempt is scored only if it FINALIZED with at least one feature: XGBoost is
   cross-validated on the committed features, encoded from the dev split. Anything
   else scores 0.
4. The last attempt's tokens are priced per agent, at each agent's own model's rates.
5. Its ``run_history.log``, ``feature_decisions.json``, one ``{role}_transcripts.json``
   per worker and (when scored) the ``feature_*_details.csv`` files are written to
   ``Team_Results/Team_{id}/Eval_{i}/``.

An evaluation runs in whichever process calls it: a Ray worker in a run (``worker.py``),
or the test process. It writes only into its own ``Eval_{i}`` folder, so concurrent
evaluations never share a file. The driver passes each returned ``EvalResult`` to an
``EvaluationRecorder``, which appends the run's ``team_evaluations.csv`` row and counts
the evaluations lost to the provider (``ProviderAbort``).

Errors: a team whose graph raised scores 0 and keeps the traceback in its log. An
evaluation still lost to the provider after its retries scores 0 too. Anything raised by
our own code (encoding, scoring, pricing, writing) is not caught: it is a bug, and it
stops the run.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, Tuple

from typeguard import typechecked

from ..Task.task import Task
from ..Knowledge.catalog import CatalogEntry
from ..Reporting.team_logs import (TEAM_EVALUATIONS_CSV, append_team_evaluation,
                                   attempt_dir, eval_dir, print_evaluation,
                                   write_failed_attempt, write_feature_decisions,
                                   write_feature_details, write_run_history,
                                   write_team_roster, write_worker_transcripts)
from ..Team_Builder.graph import GraphBuilder
from ..Team_Builder.topology import SUPERVISOR
from ..Team_Builder.team import Team
from ..Models.model_config import ModelConfig
from ..Knowledge.features import build_feature_frame_with_sources
from .outcome import RunOutcome, RunStatus, describe_outcome
from .result import EvalResult, FeatureColumn, feature_columns
from .run_graph import GraphRun, invoke_graph
from .scoring import CVScore, cross_validate
from .usage import aggregate_usage, price_usage


class ProviderAbort(RuntimeError):
    """Too many evaluations in a row were lost to the model provider."""


class Evaluator:
    """Evaluates one team, once per call, on the dev split. Holds no state between calls,
    so one Evaluator can run in any process; the driver's EvaluationRecorder keeps the
    run-wide record."""

    @typechecked
    def __init__(self, task: Task, catalog: Dict[str, CatalogEntry],
                 model_config: ModelConfig, cv_folds: int, seed: int, run_dir: str,
                 build_graph: GraphBuilder,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        """
        Args:
            catalog: from ``load_catalog``, already checked against the dataset.
            cv_folds / seed: passed to ``cross_validate``; the EA's ``seed``, so every
                team is scored on the same folds.
            run_dir: ``{save_directory}/runs/{task_name}``.
            build_graph: compiles a team (``Team_Builder/graph.py``).
            sleep: waits out ``team_retry_delay_secs``; replaced in tests.
        """
        assert cv_folds >= 2, f"Cross-validation needs at least 2 folds: {cv_folds}."
        self.task = task
        self.catalog = catalog
        self.model_config = model_config
        self.cv_folds = cv_folds
        self.seed = seed
        self.run_dir = run_dir
        self.build_graph = build_graph
        self.sleep = sleep

    @typechecked
    def evaluate(self, team: Team, team_id: int, eval_index: int) -> EvalResult:
        """Evaluate ``team`` once (see the module docstring) and write its logs."""
        out_dir = eval_dir(self.run_dir, team_id, eval_index)
        os.makedirs(out_dir, exist_ok=True)
        write_team_roster(os.path.join(out_dir, "team.txt"), team, team_id,
                          self.model_config.recursion_limit)

        max_attempts = self.model_config.team_retry_attempts + 1
        workers = sorted(role for role in team.get_roster() if role != SUPERVISOR)
        header = {"team_id": team_id, "generation": team.get_generation(),
                  "eval_index": eval_index}
        provider_failures: List[str] = []
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            graph = self.build_graph(team)
            run = invoke_graph(graph.app, graph.kickoff_message,
                               recursion_limit=self.model_config.recursion_limit,
                               time_limit_s=self.model_config.team_time_limit_s)
            outcome = describe_outcome(run, graph.review_fields)
            if outcome.status is not RunStatus.PROVIDER or attempt == max_attempts:
                break
            # Keep the lost attempt's transcript apart, so every run_history.log at the
            # Eval_{i} level is the attempt that counted.
            lost_dir = attempt_dir(out_dir, attempt)
            write_failed_attempt(lost_dir, run, outcome, attempt, max_attempts)
            write_worker_transcripts(lost_dir, run, workers, {**header, "attempt": attempt})
            provider_failures.append(str(run.error))
            delay = self.model_config.team_retry_delay_secs
            print(f"    Team {team_id} eval {eval_index}: attempt {attempt}/{max_attempts} "
                  f"lost to the provider; retrying in {delay:g}s", flush=True)
            self.sleep(delay)

        score, columns, fold_columns = self._score(outcome)
        cost = price_usage(aggregate_usage(run.all_messages()), graph.role_to_model, self.model_config)
        result = EvalResult(team_id=team_id, generation=team.get_generation(),
                            eval_index=eval_index, attempts=attempt,
                            provider_failures=provider_failures, outcome=outcome,
                            score=score, cost=cost,
                            eval_seconds=time.perf_counter() - started,
                            traceback=run.traceback, columns=columns,
                            fold_columns=fold_columns)

        self._write_logs(out_dir, run, result)
        write_worker_transcripts(out_dir, run, workers, header)
        print_evaluation(result)
        return result

    def _score(self, outcome: RunOutcome) -> Tuple[Optional[CVScore], List[FeatureColumn],
                                                   List[List[FeatureColumn]]]:
        """Cross-validate the committed features; (None, [], []) when the gate says 0."""
        if outcome.status is not RunStatus.FINALIZED or not outcome.features:
            return None, [], []
        dev = self.task.get_dataframe()
        target = self.task.target_column
        names = [f.feature_name for f in outcome.features]
        frame, sources = build_feature_frame_with_sources(dev, names, self.catalog, target,
                                                          fit_df=dev)
        score = cross_validate(frame, target, n_folds=self.cv_folds, seed=self.seed)

        def columns_for(importances: Dict[str, float]) -> List[FeatureColumn]:
            return feature_columns(frame, target, sources, outcome.features, self.catalog,
                                   importances)

        return (score, columns_for(score.feature_importances),
                [columns_for(fold) for fold in score.fold_feature_importances])

    def _write_logs(self, out_dir: str, run: GraphRun, result: EvalResult) -> None:
        write_run_history(os.path.join(out_dir, "run_history.log"), run, result)
        write_feature_decisions(os.path.join(out_dir, "feature_decisions.json"),
                                run.state, result)
        if result.columns:
            write_feature_details(out_dir, result)


class EvaluationRecorder:
    """The driver's record of every evaluation in a run.

    ``record`` appends each result as one row of the run's ``team_evaluations.csv``, and
    counts the evaluations lost to the model provider in a row. Once ``abort_after`` have
    been lost back to back, it raises ``ProviderAbort``: the key, quota or endpoint is
    broken, and every later team would score 0 for nothing.
    """

    def __init__(self, run_dir: str, abort_after: int) -> None:
        assert abort_after >= 1, f"abort_after must be at least 1: {abort_after}."
        self.csv_path = os.path.join(run_dir, TEAM_EVALUATIONS_CSV)
        self.abort_after = abort_after
        # Evaluations in a row lost to the provider; any other outcome resets it.
        self.consecutive_provider_failures = 0

    def record(self, result: EvalResult) -> None:
        """Log ``result`` and count it. Called once the evaluation's own folder is written,
        so the evidence for an abort is on disk."""
        append_team_evaluation(self.csv_path, result)
        if result.outcome.status is not RunStatus.PROVIDER:
            self.consecutive_provider_failures = 0
            return
        self.consecutive_provider_failures += 1
        if self.consecutive_provider_failures >= self.abort_after:
            raise ProviderAbort(
                f"{self.consecutive_provider_failures} evaluations in a row were lost to "
                f"the model provider, each after {result.attempts} attempt(s) (last: "
                f"{result.outcome.detail}). Check the API keys, quota and endpoints; the "
                f"evidence is in each team's run_history.log.")
