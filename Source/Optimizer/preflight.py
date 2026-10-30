"""Startup checks the EA runs before any team is evaluated.

Each check fails hard on something that would otherwise surface mid-run, after money
has been spent: a matcher, retriever or knowledge graph that breaks its contract or is
missing for a role that needs it, or a topology that allows a team the factories cannot
build. Unused inputs (a factory no allowed team fields, a retriever no role searches)
are warned about, not refused.

``check_models`` is the one opt-in check (runner.py ``--check_models``): it sends a few
small paid calls to learn what each model's API accepts, which nothing offline can tell.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.tools import tool

from ..Agents.blueprint import BluePrint
from ..Factories.base_factory import AgentFactory
from ..Factories.factory_set import FactorySet
from ..Knowledge.catalog import CatalogEntry
from ..Knowledge.knowledge_graph import KnowledgeGraph, check_kg
from ..Knowledge.retriever import Retriever, check_passages
from ..Knowledge.semantic_matcher import SemanticMatcher, check_matches
from ..Models.chat_model import knob_settings
from ..Models.model_config import ModelConfig
from ..Team_Builder.graph import ModelBuilder
from ..Team_Builder.topology import TopologyRules
from ..Utils.log import warn

# check_models: the message each model answers, and the tool bound to it.
HELLO_WORLD_PROMPT = "Reply with the words: hello world."


@tool
def hello_world() -> str:
    """Return the words hello world."""
    return "hello world"


def factories_in_use(factories: FactorySet, topology: TopologyRules) -> List[AgentFactory]:
    """The factories at least one allowed team is built from."""
    in_use = [factories.workers[r] for r in sorted(topology.roles_in_use())
              if r in factories.workers]
    if topology.single_agent_allowed() and factories.single_agent:
        in_use.append(factories.single_agent)
    if any(len(team) > 1 for team in topology.allowed_teams) and factories.supervisor:
        in_use.append(factories.supervisor)
    return in_use


def check_factories_cover_topology(factories: FactorySet, topology: TopologyRules) -> None:
    """Every allowed team must be buildable from the supplied factories.

    Worker roles are already checked against the factories when the topology
    is built; this covers the Supervisor slot, which takes a SingleAgentFactory
    on a team with no workers and a SupervisorFactory on any other.
    """
    if topology.single_agent_allowed():
        assert factories.single_agent is not None, (
            "topology_rules allow the single-agent team ('supervisor' alone), so "
            "a SingleAgentFactory is required. Supply one, or add {'supervisor'} "
            "to restricted_teams.")
    if any(len(team) > 1 for team in topology.allowed_teams):
        assert factories.supervisor is not None, (
            "topology_rules allow teams with workers, so a SupervisorFactory is "
            "required.")

    unused = sorted(set(factories.workers) - topology.roles_in_use())
    if unused:
        warn(f"No allowed team fields {unused}; their factories are unused.",
             source="topology")
    if factories.single_agent is not None and not topology.single_agent_allowed():
        warn("A SingleAgentFactory was supplied, but topology_rules never allow "
             "the single-agent team; it is unused.", source="topology")
    if factories.supervisor is not None and not any(
            len(team) > 1 for team in topology.allowed_teams):
        warn("A SupervisorFactory was supplied, but every allowed team is the "
             "single agent; it is unused.", source="topology")


def probe_matcher(matcher: SemanticMatcher, catalog: Dict[str, CatalogEntry],
                  catalog_path: str) -> None:
    """Fail now, not at a team's first ``Semantic_Matcher`` call, if the matcher
    breaks the contract (``MatcherContractError``) or cannot find a catalog entry
    from that entry's own ``match_text``."""
    entry = next(iter(catalog.values()))
    concept = entry.match_text.strip()
    matches = check_matches(matcher.match(concept), concept, catalog, matcher)
    assert matches, (
        f"{type(matcher).__name__}.match({concept!r}) found nothing, though it is the "
        f"match_text of catalog entry '{entry.feature_name}'. Is the matcher built on "
        f"{catalog_path}?")


def probe_retriever(retriever: Optional[Retriever], catalog: Dict[str, CatalogEntry],
                    in_use: List[AgentFactory]) -> None:
    """A retriever must be given when a role in use searches the literature, and must
    answer a query (the first catalog entry's ``match_text``) within its contract."""
    retrieving = sorted(f.role_name for f in in_use if f.retrieves)
    if retriever is None:
        assert not retrieving, (f"{retrieving} search the literature, so a Retriever "
                                f"is required.")
        return
    if not retrieving:
        warn("A Retriever was supplied, but no role in use searches the literature; "
             "it is unused.", source="retriever")
        return
    query = next(iter(catalog.values())).match_text.strip()
    passages = check_passages(retriever.retrieve(query, 1), query, 1, retriever)
    assert passages, (f"{type(retriever).__name__}.retrieve({query!r}, k=1) found "
                      f"nothing; is its corpus empty?")


def probe_kg(kg: Optional[KnowledgeGraph], catalog: Dict[str, CatalogEntry],
             in_use: List[AgentFactory]) -> None:
    """A knowledge graph must be given when a role in use searches it, and must
    answer a probe (``check_kg``) within its contract."""
    searching = sorted(f.role_name for f in in_use if f.searches_kg)
    if kg is None:
        assert not searching, (f"{searching} search the knowledge graph, so a "
                               f"KnowledgeGraph is required.")
        return
    if not searching:
        warn("A KnowledgeGraph was supplied, but no role in use searches it; it is "
             "unused.", source="kg")
        return
    check_kg(kg, catalog)


def check_models(models: List[str], model_config: ModelConfig,
                 build_model: ModelBuilder) -> None:
    """Ask each of ``models`` to say hello world, once per ``knob_settings`` of its
    param_space, with a tool bound as every agent has. Prints each outcome and raises
    listing every setting the API refused.

    Costs a few small calls per model, so name only the models not yet known to
    work. ``build_model`` is the one the teams use.
    """
    assert models, "check_models needs at least one model name."
    unknown = sorted(set(models) - set(model_config.model_pool()))
    assert not unknown, (f"check_models: {unknown} have no entry in "
                         f"{model_config.source}; configured models are "
                         f"{model_config.model_pool()}.")
    print("Live model check (one hello world per param_space setting):", flush=True)
    failures: List[str] = []
    for model in dict.fromkeys(models):
        for params in knob_settings(model_config.param_space(model)):
            try:
                chat = build_model(BluePrint(model=model, parameters=params))
                chat.bind_tools([hello_world]).invoke(HELLO_WORLD_PROMPT)
            except Exception as exc:  # noqa: BLE001 - every refused setting is reported
                failures.append(f"'{model}' with {params}: {type(exc).__name__}: {exc}")
                print(f"  [FAIL] {model} {params}: {type(exc).__name__}: {exc}", flush=True)
                continue
            print(f"  [ ok ] {model} {params}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} model setting(s) failed the live check; fix "
                           f"them in {model_config.source}:\n  "
                           + "\n  ".join(failures))
    print(flush=True)
