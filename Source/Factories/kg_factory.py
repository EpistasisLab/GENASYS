"""Factory for the Knowledge Graph Specialist, backed by the user's KnowledgeGraph."""

from __future__ import annotations

from typing import ClassVar, Dict, FrozenSet

from typeguard import typechecked

from .base_factory import WORKER_SYSTEM_PLACEHOLDERS, AgentFactory


@typechecked
class KGFactory(AgentFactory):
    role_name: ClassVar[str] = "kg_search"
    display_name: ClassVar[str] = "KG"
    call_tool_name: ClassVar[str] = "call_kg_agent"
    review_fields: ClassVar[Dict[str, str]] = {
        "kg_review": "every review given so far by the KG agent"}
    stages: ClassVar[bool] = True
    rejects_concepts: ClassVar[bool] = True
    matches_concepts: ClassVar[bool] = True
    searches_kg: ClassVar[bool] = True
    default_description: ClassVar[str] = (
        "knowledge graph search for reviewing existing and generating new clinical "
        "concepts and features")

    # The KG agent is only useful if it is told what the graph contains.
    required_placeholders: ClassVar[Dict[str, FrozenSet[str]]] = {
        "system": WORKER_SYSTEM_PLACEHOLDERS | {"kg_schema_summary"},
    }
    role_placeholders: ClassVar[FrozenSet[str]] = frozenset({"kg_schema_summary"})
