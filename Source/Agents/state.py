"""LangGraph state schemas shared by the Supervisor graph and its worker subgraphs.

The agents that raise concepts stage candidates into ``pending_features`` and record
the concepts they drop in ``rejected_concepts``; only the lead (the Supervisor, or the
single agent) moves a feature out of ``pending_features``, into ``accepted_features``
or ``rejected_features``, via the accept/reject/finalize writers. Workers append their opinion
to their own ``*_review`` field on a pending entry, so the Supervisor can collect
several reviews on one candidate before deciding. The reducers below define how
those concurrent writes merge.
"""

from __future__ import annotations

import copy
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from typing_extensions import TypedDict


def _merge_dict(cur: Optional[Dict], upd: Optional[Dict]) -> Dict:
    """Reducer: shallow add/overwrite by key (accepted_features, rejected_features,
    rejected_concepts)."""
    out = dict(cur or {})
    out.update(upd or {})
    return out


def _merge_pending(cur: Optional[Dict], upd: Optional[Dict]) -> Dict:
    """Reducer: merge each pending entry's fields so agents update in place.

    A tool that writes ``{feature_name: {rag_review: [...]}}`` updates only that
    field on the existing entry rather than clobbering the whole record.

    ``*_review`` fields APPEND instead of replacing. The Supervisor can send one
    pending feature to several agents, and back to the same agent after new
    evidence; with replace semantics each review would erase the last and the
    Supervisor would decide on a single opinion it never asked for. Every other
    field keeps replace semantics.

    A key mapped to ``None`` marks that feature decided: this is how it leaves the
    queue once the Supervisor accepts or rejects it (see :func:`resolve_feature`).
    The key is kept, not popped, because the handoff tools, return_final_feature_set
    and a subgraph that ends all send the parent graph their WHOLE state; a popped key
    would simply be absent from that replay, which merges as "no change", and the
    parent would keep the feature as pending. A ``None`` travels with the replay.
    Staging the feature again starts a fresh entry. Read the queue through
    :func:`pending_entries`, which skips decided features.
    """
    out = {k: (None if v is None else dict(v)) for k, v in (cur or {}).items()}
    for key, fields in (upd or {}).items():
        if fields is None:
            out[key] = None
            continue
        merged = dict(out.get(key) or {})
        for field, value in (fields or {}).items():
            if field.endswith("_review"):
                merged[field] = _merge_reviews(merged.get(field), value)
            else:
                merged[field] = value
        out[key] = merged
    return out


def _merge_reviews(cur: Any, upd: Any) -> List:
    """Append new reviews, without double-counting a replayed history.

    Two different kinds of update arrive on the same field. A review tool sends a
    delta — one new review — which must be appended. But a worker subgraph and
    ``return_final_feature_set`` hand the parent graph the WHOLE entry, so the
    reviews already recorded arrive a second time; appending those blindly would
    duplicate every review on every handoff.

    Both are distinguishable: a replay always starts with what we already have.
    So an update that extends the current list is taken as-is, and anything else
    is treated as new and appended.
    """
    cur, upd = _as_list(cur), _as_list(upd)
    if upd[:len(cur)] == cur:
        return upd
    return cur + upd


def _as_list(value: Any) -> List:
    """Coerce a review field to a list, tolerating a bare string or None."""
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


class WorkerState(TypedDict):
    """State schema for worker subgraphs built by ``langchain.agents.create_agent``.

    ``create_agent`` adds ``remaining_steps`` itself and rejects a user-supplied
    schema that already contains it, so workers get this schema while the
    Supervisor uses :class:`TeamState`. All the feature dicts live here so the
    worker ``print_*``, ``add_pending_feature`` and ``add_pending_feature_review``
    tools resolve them via ``InjectedState`` when the Supervisor hands off.
    """

    messages: Annotated[list, add_messages]
    pending_features: Annotated[Dict[str, Dict[str, Any]], _merge_pending]
    accepted_features: Annotated[Dict[str, Dict[str, Any]], _merge_dict]
    # Pending features the Supervisor reviewed and turned down; the counterpart of
    # accepted_features, keyed by feature_name.
    rejected_features: Annotated[Dict[str, Dict[str, Any]], _merge_dict]
    rejected_concepts: Annotated[Dict[str, Dict[str, Any]], _merge_dict]


class TeamState(WorkerState):
    """The whole team's state: what the Supervisor graph holds and a run returns.

    Extends :class:`WorkerState` with ``remaining_steps``, which the Supervisor's
    internal ReAct agent requires, and ``final_feature_set``. ``pending_features`` is written by the
    staging and review tools and ``rejected_concepts`` by ``add_rejected_concept``;
    ``accepted_features`` and ``rejected_features`` only by the lead's accept/reject
    writers.
    """

    # Managed value: langgraph auto-decrements it per step.
    remaining_steps: RemainingSteps
    # None until return_final_feature_set runs; then the accepted feature_names it
    # submitted, in acceptance order. Whether it is None is how the evaluator tells a
    # finalized run from one that stopped, and the list is exactly what gets scored.
    # Written once, by that tool only, so it needs no reducer.
    final_feature_set: Optional[List[str]]


_DECISIONS = ("accepted_features", "rejected_features")


def resolve_feature(state: Dict[str, Any], feature_name: str,
                    destination: str, verdict: str) -> Dict[str, Any]:
    """The state update recording the Supervisor's decision on ``feature_name``.

    ``destination`` is ``accepted_features`` or ``rejected_features``. The pending
    feature moves there with its whole record (reviews, origin, staging rationale)
    plus the Supervisor's ``verdict``, the summarized reasoning for moving it there
    (required and non-empty for both decisions). It is removed from
    ``pending_features`` in the same update, so the queue only ever holds undecided
    features.

    Only pending features can be decided, and acceptance is final: nothing removes a
    feature from ``accepted_features``. A rejected feature may be staged again and
    accepted later; it then sits in both dicts, each entry carrying the reviews and
    verdict of its own round. Raises ``KeyError`` when the feature is not pending.
    """
    assert destination in _DECISIONS, (
        f"destination must be one of {_DECISIONS}, got '{destination}'")
    assert verdict.strip(), f"a verdict is required to move '{feature_name}' to {destination}"
    pending = pending_entries(state)
    if feature_name not in pending:
        raise KeyError(f"'{feature_name}' is not in pending_features")

    entry = {**copy.deepcopy(pending[feature_name]), "verdict": verdict}
    return {destination: {feature_name: entry},
            "pending_features": {feature_name: None}}


def pending_entries(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The undecided features in ``pending_features``, skipping the ``None`` markers
    that decided ones leave behind (see :func:`_merge_pending`)."""
    return {name: entry for name, entry in state["pending_features"].items()
            if entry is not None}


def initial_state(user_message: str) -> Dict[str, Any]:
    """The seed state for one run: the kickoff message and empty feature dicts.

    Empty dicts (not missing keys) give the merge reducers a stable starting point.
    Kept beside the schema so a new state field is seeded in the same place it is
    declared.
    """
    return {
        "messages": [HumanMessage(content=user_message)],
        "pending_features": {},
        "accepted_features": {},
        "rejected_features": {},
        "rejected_concepts": {},
        "final_feature_set": None,
    }
