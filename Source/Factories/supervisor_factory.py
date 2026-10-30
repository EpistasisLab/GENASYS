"""Factory for the Supervisor, the hub every team is built around."""

from __future__ import annotations

from typing import ClassVar, Dict, FrozenSet

from typeguard import typechecked

from .base_factory import AgentFactory


@typechecked
class SupervisorFactory(AgentFactory):
    """The Supervisor of a team with workers. A team with none runs a
    ``SingleAgentFactory`` in this slot instead.
    """

    role_name: ClassVar[str] = "supervisor"
    display_name: ClassVar[str] = "Supervisor"
    # None: in a team the Supervisor only manages; its reasoning goes in the
    # accept / reject verdict, and reviews come from the workers. It does not map
    # concepts either (matches_concepts stays off): every feature it decides on was
    # mapped and staged by a worker.
    review_fields: ClassVar[Dict[str, str]] = {}
    decides: ClassVar[bool] = True
    describes_data: ClassVar[bool] = True
    default_description: ClassVar[str] = (
        "consolidates team proposals and emits the final feature spec")

    # The Supervisor has to be told who it can delegate to and how.
    required_placeholders: ClassVar[Dict[str, FrozenSet[str]]] = {
        "system": frozenset({"target_variable", "agent_roster", "agent_call_tools",
                             "origin_roles", "pending_review_fields"}),
        "user": frozenset({"target_variable", "agent_roster"}),
    }
