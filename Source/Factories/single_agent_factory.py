"""Factory for the single-agent baseline: one agent doing the whole team's job."""

from __future__ import annotations

from typing import ClassVar, Dict, FrozenSet

from typeguard import typechecked

from .base_factory import AgentFactory


@typechecked
class SingleAgentFactory(AgentFactory):
    """A lone agent with every worker's resources and tools and the Supervisor's
    decision writers: it stages, reviews, decides and finalizes by itself.

    It fills the Supervisor's slot (``role_name`` is ``"supervisor"``) because the
    team, pricing and the graph all key the hub on that name; a single-agent team
    is a Supervisor with no workers. The same factory serves the standalone
    baseline and any EA team that loses all its workers to mutation or crossover.
    """

    role_name: ClassVar[str] = "supervisor"
    display_name: ClassVar[str] = "Single Agent"
    # Every worker's perspective, written by this one agent: its own reasoning goes
    # in domain_expert_review, and its reading of each kind of tool output in that
    # tool's field. Same field names as the workers', so records compare across teams.
    review_fields: ClassVar[Dict[str, str]] = {
        "domain_expert_review": "your reviews from your own clinical and biomedical "
                                "reasoning, without tools",
        "rag_review": "your reviews of what the literature (RAG) search tools returned",
        "kg_review": "your reviews of what the knowledge graph (KG) tools returned",
        "stats_review": "your reviews of what the statistical analysis tools returned",
    }
    stages: ClassVar[bool] = True
    rejects_concepts: ClassVar[bool] = True
    decides: ClassVar[bool] = True
    describes_data: ClassVar[bool] = True
    matches_concepts: ClassVar[bool] = True
    analyzes_data: ClassVar[bool] = True
    retrieves: ClassVar[bool] = True
    searches_kg: ClassVar[bool] = True
    default_description: ClassVar[str] = (
        "proposes, reviews and commits every feature itself, with the retrieval, "
        "knowledge graph and statistical tools of the specialist agents")

    # It owns the KG tools, so, like the KG agent, it must be told what the graph
    # contains. There is no one to delegate to, so no roster is required.
    required_placeholders: ClassVar[Dict[str, FrozenSet[str]]] = {
        "system": frozenset({"target_variable", "kg_schema_summary",
                             "pending_review_fields"}),
        "user": frozenset({"target_variable"}),
    }
    role_placeholders: ClassVar[FrozenSet[str]] = frozenset({"kg_schema_summary"})
