"""A team: its roster of agent blueprints and the record of every evaluation it received.

Member variables:
  - roster (Dict[str, BluePrint]): the full team for this team — a mapping of
      agent job to blueprint. Always includes "supervisor"; the remaining keys are
      the workers the Supervisor may delegate to. Membership IS the team topology.
  - performance (List[float]): a list of the performance of the pipeline for each evaluation.
  - token_count (List[int]): a list of the token count of the pipeline for each evaluation.
  - cost (List[float]): USD spent on each evaluation, priced per agent at its own
      model's rates. Recorded alongside token_count rather than instead of it: once a
      population mixes models the two stop being interchangeable (a gpt-4o token and a
      gpt-4o-mini token differ ~17x in price), and which one NSGA-II optimizes is a
      config switch. Always 0.0 for local models, which are free.
  - eval_seconds (List[float]): wall-clock seconds each evaluation took.
  - metrics (List[Dict[str, Any]]): the full reported metric suite from each evaluation.
      performance carries only the optimized objective; this carries everything else
      (PR-AUC, accuracy, precision, recall, F1, balanced accuracy) so reporting can
      read it off a survivor without re-deriving the evaluation order.
  - features (List[List[Dict[str, Any]]]): the feature records the Supervisor committed
      in each evaluation. Kept because the post-evolution holdout stage has to rebuild
      a front pipeline's frame on the test split, and re-running the agents to recover
      the feature list would neither be cheap nor return the same answer.
"""

from typing import Any, Dict, List, Optional, Tuple
from ..Agents.blueprint import BluePrint
from typeguard import typechecked
import copy as cp
import numpy as np

# The variation operators an offspring can come from: mutation, crossover only, or
# crossover followed by mutation.
VARIATIONS = ("m", "c", "cm")


@typechecked
class Team:
    def __init__(self,
                 roster: Dict[str, BluePrint],
                 parent_id0: int,
                 parent_id1: int,
                 generation: int,
                 variation: Optional[str]) -> None:
        """
        - roster:
            The full team for this Team: a mapping of agent job to blueprint.
            Must include "supervisor"; every other key is a worker the Supervisor
            may delegate to. The set of keys IS the team topology — there is no
            separate connected-workers set.
        - parent_id0:
            The ID of the first parent pipeline.
        - parent_id1:
            The ID of the second parent pipeline.
        - generation:
            The generation this pipeline was created in.
        - variation:
            The variation operators that produced this team: 'm' (mutation),
            'c' (crossover only) or 'cm' (crossover, then mutation). None for an
            initial-population team (generation -1), which has no parents.
        """

        assert len(roster) > 0, "Roster dictionary cannot be empty."
        assert "supervisor" in roster, "Team must include a 'supervisor' agent."
        if generation == -1:
            assert variation is None, (
                f"An initial-population team (generation -1) has no variation: {variation!r}.")
        else:
            assert variation in VARIATIONS, (
                f"An offspring (generation {generation}) needs a variation in "
                f"{VARIATIONS}: {variation!r}.")

        self.roster = cp.deepcopy(roster)
        self.performance = []
        self.token_count = []
        self.cost = []
        self.eval_seconds = []
        self.metrics = []
        self.features = []
        self.parent_id0 = parent_id0
        self.parent_id1 = parent_id1
        self.generation = generation
        self.variation = variation
        return

    def set_evaluation_result(self,
                              performance: float,
                              token_count: int,
                              eval_seconds: float,
                              metrics: Optional[Dict[str, Any]] = None,
                              features: Optional[List[Dict[str, Any]]] = None,
                              cost: float = float(0.0)) -> None:
        """
        Will append to self.performance and self.token_count to keep track of multiple generation evaluations

        Args:
            performance (float): the performance of the pipeline for a given evaluation
            token_count (int): the token count of the pipeline for a given evaluation
            eval_seconds (float): wall-clock seconds this evaluation took
            metrics (Optional[Dict[str, Any]]): the full reported metric suite for this
                evaluation. Defaults to an empty dict, which is what a gated or crashed
                run has — no model was trained, so there is nothing to report.
            features (Optional[List[Dict[str, Any]]]): the feature records the Supervisor
                committed. Empty for a gated or crashed run, matching metrics.
            cost (float): USD spent on this evaluation. Trailing and defaulted so
                existing positional callers stay valid. 0.0 is legitimate, not a
                sentinel: local models are free and a crashed run spends nothing.
        """
        assert performance >= 0.0 and performance <= 1.0, f"Performance must be between 0 and 1: {performance}."
        assert token_count >= 0, f"Token count must be non-negative: {token_count}."
        assert eval_seconds >= 0.0, f"Evaluation seconds must be non-negative: {eval_seconds}."
        assert cost >= 0.0, f"Cost must be non-negative: {cost}."

        self.token_count.append(token_count)
        self.cost.append(float(cost))
        self.performance.append(performance)
        self.eval_seconds.append(float(eval_seconds))
        self.metrics.append(dict(metrics or {}))
        self.features.append(cp.deepcopy(list(features or [])))
        assert len(self.token_count) == len(self.cost) == len(self.performance) == len(self.metrics) == len(self.features), f"Length of token_count {len(self.token_count)}, cost {len(self.cost)}, performance {len(self.performance)}, metrics {len(self.metrics)}, and features {len(self.features)} must be the same."
        return

    def get_roster(self) -> Dict[str, BluePrint]:
        return cp.deepcopy(self.roster)

    def get_teammate(self, agent_job: str) -> BluePrint:
        assert agent_job in self.roster, f"Agent job {agent_job} not found in roster dictionary: {list(self.roster.keys())}."
        return cp.deepcopy(self.roster[agent_job])

    def is_evaluated(self) -> bool:
        # True once at least one evaluation has been recorded (or replayed).
        return len(self.performance) > 0

    def get_all_performance(self) -> List[float]:
        assert len(self.performance) > 0, "Performance list cannot be empty."
        return cp.deepcopy(self.performance)

    def get_median_performance(self) -> float:
        """The median performance across this pipeline's evaluations.

        The single value NSGA-II ranks a pipeline's performance on: the median is
        robust to a lone unlucky (or lucky) evaluation in a way the latest attempt
        is not, so a pipeline is judged on its typical behaviour across its
        ``eval_cnt`` runs rather than whichever attempt happened to finish last.
        """
        assert len(self.performance) > 0, "Performance list cannot be empty."
        return float(np.median(self.performance))

    def get_all_token_count(self) -> List[int]:
        assert len(self.token_count) > 0, "Token count list cannot be empty."
        return cp.deepcopy(self.token_count)

    def get_median_token_count(self) -> int:
        """The median token count across this pipeline's evaluations.

        The representative token cost NSGA-II uses as objective 2 for a local run,
        matching get_median_performance so both objectives describe the pipeline's
        typical behaviour. Rounded to the nearest int for an even number of
        evaluations, where the median falls between two counts.
        """
        assert len(self.token_count) > 0, "Token count list cannot be empty."
        return int(round(float(np.median(self.token_count))))

    def get_all_cost(self) -> List[float]:
        assert len(self.cost) > 0, "Cost list cannot be empty."
        return cp.deepcopy(self.cost)

    def get_median_cost(self) -> float:
        """The median USD cost across this pipeline's evaluations.

        The representative objective 2 NSGA-II uses once models are mixed, matching
        get_median_performance so both objectives describe the pipeline's typical
        behaviour rather than its most recent attempt.
        """
        assert len(self.cost) > 0, "Cost list cannot be empty."
        return float(np.median(self.cost))

    def get_all_eval_seconds(self) -> List[float]:
        return cp.deepcopy(self.eval_seconds)

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Every evaluation's metric suite, in evaluation order.

        Parallel to performance/token_count/cost, so index i is the same
        evaluation across all of them. The archive reads this to persist and to
        replay the full multi-evaluation record onto a reused pipeline.
        """
        return cp.deepcopy(self.metrics)

    def get_all_features(self) -> List[List[Dict[str, Any]]]:
        """Every evaluation's committed feature records, in evaluation order.

        Parallel to the other per-evaluation lists. The archive persists this and
        replays it when a genome is reused so the holdout stage can rebuild the
        frame without re-running the agents.
        """
        return cp.deepcopy(self.features)

    def median_evaluation(self) -> int:
        """Index of the evaluation whose performance is the median.

        For an even number of evaluations ``get_median_performance`` falls between
        two of them; this is the lower one, so the index always names an evaluation
        that actually ran. Ties go to the earlier evaluation.
        """
        assert len(self.performance) > 0, "Performance list cannot be empty."
        order = np.argsort(self.performance, kind="stable")
        return int(order[(len(order) - 1) // 2])

    def get_median_features(self) -> List[Dict[str, Any]]:
        """The feature records committed in the median evaluation (``median_evaluation``).

        The holdout stage rebuilds a front team's frame from these: the feature set
        behind its typical score, rather than whichever evaluation ran last.
        """
        return cp.deepcopy(self.features[self.median_evaluation()])

    def get_median_metrics(self) -> Dict[str, Any]:
        """The metric suite of the median evaluation, matching ``get_median_features``."""
        return cp.deepcopy(self.metrics[self.median_evaluation()])

    def get_parent_ids(self) -> Tuple[int, int]:
        """Return the IDs of the two parent pipelines.

        Returns:
            A tuple containing the ID of the first parent and the ID of the second parent.
        """
        return (self.parent_id0, self.parent_id1)

    def get_variation(self) -> Optional[str]:
        """Return the variation operators that produced this team ('m', 'c' or
        'cm'), or None for an initial-population team."""
        return self.variation

    def get_generation(self) -> int:
        """Return the generation this pipeline was created in.

        Returns:
            The generation number.
        """
        return self.generation

    def print_team(self) -> None:
        print("Agents and their blueprints:")
        for agent_job, blueprint in self.roster.items():
            print(f"{agent_job}:")
            print(f"  Model: {blueprint.model}")
            print(f"  Parameters: {blueprint.parameters}")
        return