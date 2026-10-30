"""How a team's run ended, and the features it committed.

Only a ``FINALIZED`` run is scored: ``return_final_feature_set`` sets the state's
``final_feature_set`` (None until then), and that list is exactly what was committed.
Accepting a feature does not submit it, so a run that stopped, ran out of steps or
failed scores 0 however good its ``accepted_features`` were. Acceptance is final
(nothing removes a feature from ``accepted_features``), so the finalized list is the
accepted set at the moment of finalizing.

The feature entries are written by our own tools, so a missing field, a review field
the team does not have, or a finalized name with no accepted entry raises: each is a
bug in those tools, not something a team did.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Collection, Dict, List, Mapping, Optional, Tuple

from langgraph.errors import GraphRecursionError
from typeguard import typechecked

from ..Agents.state import pending_entries
from .run_graph import GraphRun, TeamTimeLimit

# What LangGraph's prebuilt ReAct agent (which runs the Supervisor) says instead of
# calling another tool when its remaining_steps run out. It ends the run cleanly, so
# without this check a truncated run would look like one that just stopped.
STEP_BUDGET_MESSAGE = "Sorry, need more steps to process this request."

# Fields every accepted entry carries besides its *_review lists: copied from the
# pending entry the tools staged, plus the Supervisor's ``verdict``.
_ENTRY_FIELDS = ("concept", "feature_type", "similarity_score", "added_by",
                 "suggested_by", "added_by_rationale", "verdict")


class RunStatus(str, Enum):
    """How a run ended. Only FINALIZED runs are scored."""

    FINALIZED = "finalized"   # return_final_feature_set ran
    TRUNCATED = "truncated"   # ran out of steps (recursion limit or step budget) or time
    STOPPED = "stopped"       # ended without finalizing, e.g. a turn with only text
    ERROR = "error"           # the graph raised; traceback in run_history.log
    PROVIDER = "provider"     # the model provider failed; retried by the evaluator


@dataclass(frozen=True)
class FeatureRecord:
    """One committed feature, as the team recorded it.

    ``reviews`` holds every review field on the team (``team_review_fields``), each a
    list of reviews oldest first; a field nobody wrote to is an empty list.
    ``added_by`` staged the feature, ``suggested_by`` is credited with the idea, and
    ``added_by_rationale`` is why it was staged and ``verdict`` the Supervisor's
    summarized reasoning for accepting it.
    """

    feature_name: str
    concept: str
    feature_type: str
    similarity_score: float
    added_by: str
    suggested_by: str
    added_by_rationale: str
    verdict: str
    reviews: Dict[str, List[str]]


# Where a feature stood when the run ended.
DECISIONS = ("accepted", "rejected", "pending")


@dataclass(frozen=True)
class FeatureDecision:
    """One feature in the final state, whatever became of it.

    A feature rejected and later re-staged and accepted has one decision per round.
    ``committed`` is True only for an accepted feature the finalized run submitted.
    A pending feature has no verdict yet, so its ``record.verdict`` is "".
    """

    decision: str
    committed: bool
    record: FeatureRecord


@dataclass(frozen=True)
class RunOutcome:
    """``features`` is empty unless the run finalized. ``decisions`` holds every
    accepted, rejected and still-pending feature in the final state, finalized or not,
    so a failed run's work is still on record. The counts are of the final state's
    ``accepted_features``, ``rejected_features`` and undecided ``pending_features``."""

    status: RunStatus
    detail: str
    features: List[FeatureRecord]
    decisions: List[FeatureDecision]
    n_accepted: int
    n_rejected: int
    n_pending: int


@typechecked
def run_status(run: GraphRun) -> Tuple[RunStatus, str]:
    """Classify how ``run`` ended, with a one-line detail for the logs and console."""
    if run.infrastructure_error:
        return RunStatus.PROVIDER, f"the model provider failed: {run.error}"
    if run.error_type == GraphRecursionError.__name__:
        return RunStatus.TRUNCATED, ("hit recursion_limit before return_final_feature_set; "
                                     "raise recursion_limit")
    if run.error_type == TeamTimeLimit.__name__:
        return RunStatus.TRUNCATED, (f"{run.error} before return_final_feature_set; "
                                     f"raise team_time_limit_s")
    if run.error is not None:
        return RunStatus.ERROR, f"the graph raised {run.error}"

    final = run.state["final_feature_set"]
    if final is not None:
        return RunStatus.FINALIZED, f"return_final_feature_set submitted {len(final)} features"
    messages = run.state["messages"]
    if messages and messages[-1].content == STEP_BUDGET_MESSAGE:
        return RunStatus.TRUNCATED, ("an agent ran out of remaining_steps before "
                                     "return_final_feature_set; raise recursion_limit")
    return RunStatus.STOPPED, ("ended without calling return_final_feature_set "
                               "(e.g. a turn with only text)")


@typechecked
def committed_features(state: Mapping[str, Any],
                       review_fields: Collection[str]) -> List[FeatureRecord]:
    """The finalized features as records, in ``final_feature_set`` order.

    Args:
        state: the final state of a finalized run.
        review_fields: the team's review fields (``team_review_fields(team)``); each
            entry must carry exactly these.
    """
    final = state["final_feature_set"]
    assert final is not None, "the run did not finalize, so nothing was committed"
    accepted = state["accepted_features"]
    missing = [name for name in final if name not in accepted]
    assert not missing, \
        f"return_final_feature_set submitted {missing}, which are not in accepted_features"
    return [_record(name, accepted[name], set(review_fields), "accepted") for name in final]


@typechecked
def feature_decisions(state: Mapping[str, Any],
                      review_fields: Collection[str]) -> List[FeatureDecision]:
    """Every accepted, rejected and undecided pending feature in ``state``, in that
    order, each in the order it was recorded. Works on any run's final state."""
    final: Optional[List[str]] = state["final_feature_set"]
    committed = set(final or [])
    fields = set(review_fields)
    decisions = [FeatureDecision("accepted", name in committed,
                                 _record(name, entry, fields, "accepted"))
                 for name, entry in state["accepted_features"].items()]
    decisions += [FeatureDecision("rejected", False, _record(name, entry, fields, "rejected"))
                  for name, entry in state["rejected_features"].items()]
    decisions += [FeatureDecision("pending", False, _record(name, entry, fields, "pending"))
                  for name, entry in pending_entries(state).items()]
    return decisions


def _record(name: str, entry: Mapping[str, Any], review_fields: set,
            decision: str) -> FeatureRecord:
    """``entry`` as a record. A pending entry has no verdict yet; it gets ""."""
    assert decision in DECISIONS, f"unknown decision '{decision}'"
    expected = _ENTRY_FIELDS if decision != "pending" else _ENTRY_FIELDS[:-1]
    reviews = {k: v for k, v in entry.items() if k.endswith("_review")}
    lacking = [k for k in expected if k not in entry]
    assert not lacking, f"{decision} feature '{name}' has no {lacking}"
    unknown = sorted(set(entry) - set(expected) - set(reviews))
    assert not unknown, f"{decision} feature '{name}' has unknown fields {unknown}"
    assert set(reviews) == review_fields, \
        (f"{decision} feature '{name}' has review fields {sorted(reviews)}, but the team's "
         f"are {sorted(review_fields)}")
    for field, values in reviews.items():
        assert isinstance(values, list), \
            f"{decision} feature '{name}': {field} is {type(values).__name__}, not a list"
    return FeatureRecord(feature_name=name,
                         concept=entry["concept"],
                         feature_type=entry["feature_type"],
                         similarity_score=float(entry["similarity_score"]),
                         added_by=entry["added_by"],
                         suggested_by=entry["suggested_by"],
                         added_by_rationale=entry["added_by_rationale"],
                         verdict=entry.get("verdict", ""),
                         reviews={k: list(v) for k, v in reviews.items()})


@typechecked
def describe_outcome(run: GraphRun, review_fields: Collection[str]) -> RunOutcome:
    """How ``run`` ended and, if it finalized, what it committed.

    A FINALIZED run with no features is still FINALIZED; the evaluator scores it 0.
    """
    status, detail = run_status(run)
    features = (committed_features(run.state, review_fields)
                if status is RunStatus.FINALIZED else [])
    return RunOutcome(status=status, detail=detail, features=features,
                      decisions=feature_decisions(run.state, review_fields),
                      n_accepted=len(run.state["accepted_features"]),
                      n_rejected=len(run.state["rejected_features"]),
                      n_pending=len(pending_entries(run.state)))
