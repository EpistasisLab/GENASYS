"""Factory for the RAG Specialist, which searches the literature with the user's Retriever."""

from __future__ import annotations

from typing import ClassVar, Dict, FrozenSet

from typeguard import typechecked

from .base_factory import WORKER_SYSTEM_PLACEHOLDERS, AgentFactory

@typechecked
class RAGFactory(AgentFactory):
    role_name: ClassVar[str] = "rag_search"
    display_name: ClassVar[str] = "RAG"
    call_tool_name: ClassVar[str] = "call_rag_agent"
    review_fields: ClassVar[Dict[str, str]] = {
        "rag_review": "every review given so far by the RAG agent"}
    stages: ClassVar[bool] = True
    rejects_concepts: ClassVar[bool] = True
    matches_concepts: ClassVar[bool] = True
    retrieves: ClassVar[bool] = True
    default_description: ClassVar[str] = (
        "literature retrieval for evidence-based review of existing and generation "
        "of new clinical concepts and features")

    required_placeholders: ClassVar[Dict[str, FrozenSet[str]]] = {
        "system": WORKER_SYSTEM_PLACEHOLDERS,
    }
