"""The tools agents use to read and change the feature dicts in the team state.

Who gets what (assigned by the factories' tool flags, see ``AgentFactory.tools``):

    Tool                                   Workers   Single agent   Supervisor
    -------------------------------------  --------  -------------  ----------
    add_pending_feature                    RAG, KG,  yes            -
                                           Domain
                                           Expert
    <field>_review (e.g. rag_review)       own one   all four       -
    add_rejected_concept                   RAG, KG,  yes            -
                                           Domain
                                           Expert
    add_accepted_feature                   -         yes            yes
    add_rejected_feature                   -         yes            yes
    return_final_feature_set               -         yes            yes
    print_* inspectors                     yes       yes            yes

The agents that raise concepts from knowledge (RAG, KG, Domain Expert, single agent)
stage their own candidates and drop the concepts that do not map; only a lead (the
Supervisor, or the single agent) decides on them. The Supervisor of a team only
manages, so it neither stages nor reviews. The Statistician only reviews: it analyzes
features already pending, so it neither stages nor has concepts to drop.

Staging records two roles. ``added_by`` is the calling role, fixed when the tool is
built. ``suggested_by`` is the credit field, chosen by the model from the team's roles:
the teammate whose analysis led to the feature, which need not be the one staging it
(the KG agent may stage a concept the RAG agent asked it to look up). Whether the
credit is given correctly is up to the agents; a role not on the team is refused. A
staged entry carries every review field of the team, each starting as [].
``add_rejected_concept`` records ``added_by`` the same way.

Model mistakes (a name not in the catalog, a feature that is not pending, an empty
verdict) are answered with a ToolMessage telling the model what to do instead; the
state is left unchanged. Only our own bugs raise.

Writers return ``Command(update=...)``; the reducers in ``Agents/state.py`` merge
the update. Every Command carries a ToolMessage answering its own tool call, or
LangGraph's ToolNode aborts the run. Decided features stay in ``pending_features``
as ``None`` markers, so every tool reads the queue through ``pending_entries``.
"""

from __future__ import annotations

import difflib
import json
from typing import Annotated, Any, Dict, List, Mapping, Sequence

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.graph import END
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typeguard import typechecked

from ...Knowledge.catalog import CatalogEntry
from ..state import pending_entries, resolve_feature

# How many close catalog names to suggest when a staged name is not in the catalog.
_SUGGESTIONS = 3


# ------------------------------------- staging ----------------------------------------------

@typechecked
def staging_tool(role: str, team_roles: Sequence[str], review_fields: Sequence[str],
                 catalog: Mapping[str, CatalogEntry]) -> BaseTool:
    """``add_pending_feature`` for ``role``, on a team of ``team_roles`` whose review
    fields are ``review_fields``.

    The feature's ``feature_type`` is taken from the catalog, which is the authority
    on it. Refused (answered, not raised): a name not in the catalog, a
    ``suggested_by`` that is not one of ``team_roles``, a feature already accepted
    (acceptance is final), a feature already pending (its reviews belong to the
    staging already open), and an empty concept or rationale.
    """
    assert role in team_roles, f"role '{role}' is not on the team {list(team_roles)}"
    assert all(f.endswith("_review") for f in review_fields), \
        f"review fields must end in '_review': {list(review_fields)}"
    roles, fields = list(team_roles), list(review_fields)

    @tool(description=(
        "Suggest a feature by adding it to pending_features for review and a decision. "
        "feature_name: a feature_name returned by Semantic_Matcher, copied verbatim. "
        "concept: the concept you passed to Semantic_Matcher. similarity_score: the "
        "similarity_score Semantic_Matcher returned for it. added_by_rationale: why you "
        "are suggesting this feature. suggested_by: the teammate whose analysis led to "
        f"this feature, yourself included; one of {', '.join(roles)}."))
    def add_pending_feature(feature_name: str, concept: str, similarity_score: float,
                            added_by_rationale: str, suggested_by: str,
                            state: Annotated[Dict[str, Any], InjectedState],
                            tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        if feature_name not in catalog:
            return _reply(tool_call_id, _not_in_catalog(feature_name, catalog))
        if not concept.strip() or not added_by_rationale.strip():
            return _reply(tool_call_id, f"NOT STAGED: '{feature_name}' needs a non-empty "
                                        f"concept and added_by_rationale.")
        if suggested_by not in roles:
            return _reply(tool_call_id, (
                f"NOT STAGED: suggested_by must be one of {', '.join(roles)} (the teammate "
                f"whose analysis led to '{feature_name}'), got '{suggested_by}'."))
        if feature_name in state["accepted_features"]:
            return _reply(tool_call_id, f"NOT STAGED: '{feature_name}' is already accepted; "
                                        f"acceptance is final.")
        pending = pending_entries(state)
        if feature_name in pending:
            return _reply(tool_call_id, (
                f"NOT STAGED: '{feature_name}' is already in pending_features, staged by "
                f"{pending[feature_name]['added_by']}. Add a review to it instead."))

        entry = {"concept": concept,
                 "feature_type": catalog[feature_name].feature_type,
                 "similarity_score": similarity_score,
                 "added_by": role,
                 "suggested_by": suggested_by,
                 "added_by_rationale": added_by_rationale,
                 **{field: [] for field in fields}}
        return Command(update={"pending_features": {feature_name: entry},
                               "messages": _ack(tool_call_id,
                                                f"pending_features += {feature_name}")})

    return add_pending_feature


def _not_in_catalog(feature_name: str, catalog: Mapping[str, CatalogEntry]) -> str:
    """The reply to a staged name that is not a catalog feature, with the closest names."""
    close = difflib.get_close_matches(feature_name, list(catalog), n=_SUGGESTIONS)
    hint = (f" Closest catalog names: {json.dumps(close)}." if close else "")
    return (f"NOT STAGED: '{feature_name}' is not a feature in the dataset catalog.{hint} "
            f"Call Semantic_Matcher and copy a returned feature_name character for "
            f"character.")


# ------------------------------------- reviews ----------------------------------------------

@typechecked
def review_tool(field: str) -> BaseTool:
    """The tool that appends a review to ``field`` of a pending feature, named after
    the field (``rag_review``, ``kg_review``, ...). A pending feature is required:
    a review of anything else is refused."""
    assert field.endswith("_review"), f"review field '{field}' must end in '_review'"

    @tool(field, description=(
        f"Add your review to {field} of a feature in pending_features. It is appended "
        f"to that feature's {field} list; it never replaces an earlier review."))
    def write_review(feature_name: str, review: str,
                     state: Annotated[Dict[str, Any], InjectedState],
                     tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        if feature_name not in pending_entries(state):
            return _reply(tool_call_id, (
                f"'{feature_name}' is not in pending_features. Call "
                f"print_pending_features_list to see the exact names available, or "
                f"add_pending_feature if you mean to suggest it as a new feature."))
        if not review.strip():
            return _reply(tool_call_id, f"NOT RECORDED: the review of '{feature_name}' "
                                        f"is empty.")
        return Command(update={"pending_features": {feature_name: {field: [review]}},
                               "messages": _ack(tool_call_id,
                                                f"{field} recorded for {feature_name}")})

    return write_review


# ------------------------------------- decisions --------------------------------------------

@typechecked
def decision_tools() -> List[BaseTool]:
    """The lead's tools: accept or reject a pending feature, and finalize the run."""
    return [_decide_tool("add_accepted_feature", "accepted_features"),
            _decide_tool("add_rejected_feature", "rejected_features"),
            _finalize_tool()]


def _decide_tool(name: str, destination: str) -> BaseTool:
    """``name`` moves a pending feature to ``destination`` with the lead's verdict."""
    action = {"accepted_features": "Accept a feature for the downstream predictive model. "
                                   "Acceptance is final.",
              "rejected_features": "Reject a feature. It may be staged again later if "
                                   "new evidence turns up."}[destination]

    @tool(name, description=(
        f"{action} The feature must be in pending_features; it moves to {destination} "
        f"with its reviews and your verdict."))
    def decide(feature_name: str, verdict: str,
               state: Annotated[Dict[str, Any], InjectedState],
               tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        if feature_name not in pending_entries(state):
            return _reply(tool_call_id, (
                f"NOTHING CHANGED: '{feature_name}' is not in pending_features. Only a "
                f"staged feature can be decided. Call print_pending_features_list to see "
                f"what is waiting on your decision."))
        if not verdict.strip():
            return _reply(tool_call_id, f"NOTHING CHANGED: a verdict is required to move "
                                        f"'{feature_name}' to {destination}.")
        return Command(update={**resolve_feature(state, feature_name, destination, verdict),
                               "messages": _ack(tool_call_id,
                                                f"{destination} += {feature_name}")})

    return decide


@typechecked
def reject_concept_tool(role: str) -> BaseTool:
    """``add_rejected_concept`` for ``role``: record a concept dropped before it was ever
    staged. ``added_by`` is ``role``, fixed when the tool is built, as in staging."""
    @tool
    def add_rejected_concept(concept: str, rationale: str,
                             tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Record a concept you are dropping that was never staged as a feature, e.g. one
        Semantic_Matcher found no match for. To turn down a staged feature, use
        add_rejected_feature.

        Args:
            concept: the concept being rejected (as passed to Semantic_Matcher).
            rationale: why the concept is rejected.
        """
        if not concept.strip() or not rationale.strip():
            return _reply(tool_call_id, "NOTHING CHANGED: a concept and a non-empty "
                                        "rationale are required.")
        entry = {"added_by": role, "rationale": rationale}
        return Command(update={"rejected_concepts": {concept: entry},
                               "messages": _ack(tool_call_id,
                                                f"rejected_concepts += {concept}")})

    return add_rejected_concept


def _finalize_tool() -> BaseTool:
    @tool
    def return_final_feature_set(state: Annotated[Dict[str, Any], InjectedState],
                                 tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Submit accepted_features as the final feature set and end the run. Without
        this call the run scores zero."""
        final = list(state["accepted_features"])
        # The lead runs as a subgraph of the supervisor workflow, so ending only its own
        # loop would let the workflow call it again: Command.PARENT ends the run. The
        # parent then never receives the subgraph's state the normal way, so the whole
        # state is replayed here, as the handoff tools do. remaining_steps is a managed
        # value of the subgraph that the parent has no channel for.
        replay = {k: v for k, v in state.items() if k != "remaining_steps"}
        return Command(update={**replay, "final_feature_set": final,
                               "messages": [*state["messages"], *_ack(
                                   tool_call_id,
                                   f"FINAL_FEATURE_SET: {json.dumps(final)}")]},
                       goto=END, graph=Command.PARENT)

    return return_final_feature_set


# ------------------------------------- inspectors -------------------------------------------

@typechecked
def inspector_tools() -> List[BaseTool]:
    """Read-only views of the feature dicts, for every agent."""
    return [_list_tool("print_pending_features_list", "pending_features"),
            _list_tool("print_accepted_features_list", "accepted_features"),
            _list_tool("print_rejected_features_list", "rejected_features"),
            _list_tool("print_rejected_concepts_list", "rejected_concepts"),
            _properties_tool("print_pending_feature_properties", "pending_features"),
            _properties_tool("print_accepted_feature_properties", "accepted_features"),
            _properties_tool("print_rejected_feature_properties", "rejected_features"),
            _properties_tool("print_rejected_concept_properties", "rejected_concepts")]


def _entries(state: Dict[str, Any], key: str) -> Dict[str, Dict[str, Any]]:
    return pending_entries(state) if key == "pending_features" else state[key]


def _list_tool(name: str, key: str) -> BaseTool:
    noun = "concept" if key == "rejected_concepts" else "feature_name"

    @tool(name, description=f"Return a JSON list of the {noun} strings in {key}.")
    def print_list(state: Annotated[Dict[str, Any], InjectedState]) -> str:
        return json.dumps(list(_entries(state, key)))

    return print_list


def _properties_tool(name: str, key: str) -> BaseTool:
    description = (f"Return the properties of one "
                   f"{'concept' if key == 'rejected_concepts' else 'feature_name'} in "
                   f"{key}, as JSON.")
    # The argument is named as the prompts describe it.
    if key == "rejected_concepts":
        @tool(name, description=description)
        def print_concept(concept: str, state: Annotated[Dict[str, Any], InjectedState]) -> str:
            return _properties(state, key, concept)
        return print_concept

    @tool(name, description=description)
    def print_feature(feature_name: str,
                      state: Annotated[Dict[str, Any], InjectedState]) -> str:
        return _properties(state, key, feature_name)
    return print_feature


def _properties(state: Dict[str, Any], key: str, wanted: str) -> str:
    entry = _entries(state, key).get(wanted)
    if entry is None:
        return json.dumps({"error": f"'{wanted}' is not in {key}; call the matching "
                                    f"print_*_list tool for the exact names"})
    return json.dumps(entry, indent=2)


# -------------------------------------- helpers ---------------------------------------------

def _ack(tool_call_id: str, text: str) -> List[ToolMessage]:
    """The ToolMessage every Command must carry for its own tool call."""
    return [ToolMessage(content=text, tool_call_id=tool_call_id)]


def _reply(tool_call_id: str, text: str) -> Command:
    """Answer the model without changing the feature dicts."""
    return Command(update={"messages": _ack(tool_call_id, text)})
