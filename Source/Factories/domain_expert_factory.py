"""Factory for the Domain Expert, which reasons from the model's own knowledge."""

from __future__ import annotations

from typing import ClassVar, Dict, FrozenSet

from typeguard import typechecked

from .base_factory import WORKER_SYSTEM_PLACEHOLDERS, AgentFactory


@typechecked
class DomainExpertFactory(AgentFactory):
    role_name: ClassVar[str] = "domain_expert"
    display_name: ClassVar[str] = "Domain Expert"
    call_tool_name: ClassVar[str] = "call_domain_expert_agent"
    review_fields: ClassVar[Dict[str, str]] = {
        "domain_expert_review": "every review given so far by the Biomedical Domain Expert"}
    stages: ClassVar[bool] = True
    rejects_concepts: ClassVar[bool] = True
    matches_concepts: ClassVar[bool] = True
    default_description: ClassVar[str] = (
        "biomedical domain expertise for reviewing existing and generating new "
        "clinical concepts and features using internal clinical knowledge")

    required_placeholders: ClassVar[Dict[str, FrozenSet[str]]] = {
        "system": WORKER_SYSTEM_PLACEHOLDERS,
    }
